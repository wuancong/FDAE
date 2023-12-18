import numpy as np
import os
import gin
import json
from time import strftime, gmtime
import time
from data.ground_truth import named_data
from evaluation.metrics.utils import _cluster_discretize
import argparse
import torch


def write_text(metric, result_dict, print_txt, file):
    file = file.replace('model', 'eval')
    if os.path.exists(file):
        with open(file, 'r') as f:
            new_dict = json.load(f)
    else:
        new_dict = {}
    new_dict[metric] = result_dict
    if print_txt:
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        with open(file, 'w') as f:
            json.dump(new_dict, f)


def evaluate(net, args=None, cluster_random_state=0,
             dataset=None,
             beta_VAE_score=False,
             dci_score=False,
             factor_VAE_score=False,
             MIG=False,
             print_txt=False,
             txt_name="metric.json",
             log_dir='',
             group_label=None,
             ):

    def _representation(x):
        x = x * 2 - 1.0
        x = torch.from_numpy(x)
        x = x.float().cuda()
        x = x.permute(0, 3, 1, 2)
        out = net(x.contiguous())
        if not isinstance(out, dict):
            out = {'feature': out}
        z = out[args.feature_name].squeeze()
        z = z.view(z.shape[0], -1)
        return z.detach().cpu().numpy()

    if beta_VAE_score:
        t_begin = time.time()
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(dataset, _representation,
                                               random_state=np.random.RandomState(args.test_seed), artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        write_text("beta_VAE_score", result_dict, print_txt, os.path.join(log_dir, txt_name))
        gin.clear_config()
        print('Time elapsed: ' + strftime("%H:%M:%S", gmtime(time.time() - t_begin)))
    if MIG:
        t_begin = time.time()
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            gin.bind_parameter("mig.num_train", 10000)
            if group_label is not None:
                gin.bind_parameter("mig.group_label", group_label)
            gin.bind_parameter("discretizer.discretizer_fn", _cluster_discretize)
            gin.bind_parameter("discretizer.random_state", cluster_random_state)
            gin.bind_parameter("discretizer.pca_dim", args.postprocessing_pca_dim)

            gin.bind_parameter("discretizer.num_bins", 20)

        result_dict = compute_mig(dataset, _representation, random_state=np.random.RandomState(args.test_seed))
        print("MIG score:" + str(result_dict))
        write_text("MIG", result_dict, print_txt, os.path.join(log_dir, txt_name))
        gin.clear_config()
        print('Time elapsed: ' + strftime("%H:%M:%S", gmtime(time.time() - t_begin)))
    if factor_VAE_score:
        t_begin = time.time()
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate", 10000)
            gin.bind_parameter("factor_vae_score.num_train", 10000)
            gin.bind_parameter("factor_vae_score.num_eval", 5000)
            gin.bind_parameter("factor_vae_score.batch_size", 64)
            gin.bind_parameter("factor_vae_score.group_label", group_label)
        result_dict = compute_factor_vae(dataset, _representation, random_state=np.random.RandomState(args.test_seed),
                                         pca_dim=args.postprocessing_pca_dim)
        print("factor VAE score:" + str(result_dict))
        write_text("factor_VAE_score", result_dict, print_txt, os.path.join(log_dir, txt_name))
        gin.clear_config()
        print('Time elapsed: ' + strftime("%H:%M:%S", gmtime(time.time() - t_begin)))
    if dci_score:
        t_begin = time.time()
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
            gin.bind_parameter("dci.lr", args.dci_lr)
        if group_label is not None:
            gin.bind_parameter("dci.group_label", group_label)
        result_dict, importance_matrix = compute_dci(dataset, _representation, random_state=np.random.RandomState(args.test_seed),
                                  pca_dim=args.postprocessing_pca_dim, factor_num=args.factor_num)
        print("dci score:" + str(result_dict))
        write_text("dci_score", result_dict, print_txt, os.path.join(log_dir, txt_name))
        if importance_matrix is not None:
            npy_path = os.path.join(log_dir, 'importance_matrix.npy')
            np.save(npy_path, importance_matrix)
        gin.clear_config()
        print('Time elapsed: ' + strftime("%H:%M:%S", gmtime(time.time() - t_begin)))


def default_args():
    default = dict(
        dataset=-1,
        test_seed=0,
        factor_num=-1,
        cluster_random_state=0,
        dci_lr=0.1,
        use_group_label=True,
        mig=True,
        postprocessing_pca_dim=3,
        vae=True,
        dci=True,
        feature_name='feature',
    )
    args = argparse.Namespace(**default)
    return args


def evaluate_diffusion(dataset_name, train_args, log_dir, encoder, step, feature_dim=36):
    args = default_args()
    assert dataset_name in ["shapes3d", "mpi3d_real_complex", "mpi3d_real", "cars3d"]
    args.dataset = dataset_name
    args.postprocessing_pca_dim = feature_dim // (train_args.semantic_group_num * 2)

    with gin.unlock_config():
        gin.bind_parameter("dataset.name", args.dataset)
    dataset_ = named_data.get_named_ground_truth_data()

    group_label = []
    for i in range(train_args.semantic_group_num):
        group_label += [i] * train_args.semantic_code_dim
    for i in range(train_args.semantic_group_num, train_args.semantic_group_num * 2):
        group_label += [i] * train_args.mask_code_dim
    group_label = np.array(group_label)

    if train_args.semantic_group_num == 1:
        group_label = None
        args.use_group_label = False
        args.cluster_method = None

    encoder.eval()
    with torch.no_grad():
        evaluate(encoder, args,
                 dataset=dataset_,
                 beta_VAE_score=False,
                 dci_score=args.dci,
                 factor_VAE_score=args.vae,
                 MIG=args.mig,
                 print_txt=True,
                 group_label=group_label,
                 log_dir=log_dir,
                 txt_name=f'metric_step{step}.json',
                 )
    encoder.train()
