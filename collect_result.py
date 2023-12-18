import os
import json
import numpy as np


metric_type_list = ['mig', 'dci_d', 'dci_c', 'dci_i_train', 'dci_i_test',
                    'fvae_train', 'fvae_test']

model_list = [
    [
        '2023-12-15-21-52-21-cars3d_FDAE_seed0_rank0',
        '2023-12-18-01-34-25-cars3d_FDAE_seed8_rank0',
        '2023-12-17-13-05-23-cars3d_FDAE_seed7_rank0',
        '2023-12-18-01-21-51-cars3d_FDAE_seed4_rank0',
        '2023-12-17-12-57-52-cars3d_FDAE_seed3_rank0',
        '2023-12-17-00-36-31-cars3d_FDAE_seed6_rank0',
        '2023-12-17-00-34-38-cars3d_FDAE_seed2_rank0',
        '2023-12-16-12-11-28-cars3d_FDAE_seed9_rank0',
        '2023-12-16-12-08-19-cars3d_FDAE_seed5_rank0',
        '2023-12-16-12-11-51-cars3d_FDAE_seed1_rank0',
    ],
    [
        '2023-12-18-02-00-41-shapes3d_FDAE_seed8_rank0',
        '2023-12-17-13-23-49-shapes3d_FDAE_seed7_rank0',
        '2023-12-18-01-54-33-shapes3d_FDAE_seed4_rank0',
        '2023-12-17-13-18-58-shapes3d_FDAE_seed3_rank0',
        '2023-12-17-00-46-36-shapes3d_FDAE_seed6_rank0',
        '2023-12-17-00-44-01-shapes3d_FDAE_seed2_rank0',
        '2023-12-16-12-07-40-shapes3d_FDAE_seed5_rank0',
        '2023-12-16-12-07-18-shapes3d_FDAE_seed1_rank0',
        '2023-12-15-21-51-24-shapes3d_FDAE_seed0_rank0',
    ],
    [
        '2023-12-18-02-18-59-mpi3d_real_complex_FDAE_seed8_rank0',
        '2023-12-17-13-36-46-mpi3d_real_complex_FDAE_seed7_rank0',
        '2023-12-18-02-02-54-mpi3d_real_complex_FDAE_seed4_rank0',
        '2023-12-17-13-26-20-mpi3d_real_complex_FDAE_seed3_rank0',
        '2023-12-17-00-54-06-mpi3d_real_complex_FDAE_seed6_rank0',
        '2023-12-17-00-49-01-mpi3d_real_complex_FDAE_seed2_rank0',
        '2023-12-16-12-10-48-mpi3d_real_complex_FDAE_seed5_rank0',
        '2023-12-16-12-11-10-mpi3d_real_complex_FDAE_seed9_rank0',
        '2023-12-16-12-12-08-mpi3d_real_complex_FDAE_seed1_rank0',
        '2023-12-15-21-52-50-mpi3d_real_complex_FDAE_seed0_rank0',
    ]
]

result_name = 'metric_step100000.json'


def get_metric(metrics, metric_type):
    if 'mig' in metric_type:
        metric = metrics.get('MIG')
    elif 'dci' in metric_type:
        metric = metrics.get('dci_score')
    elif 'fvae' in metric_type:
        metric = metrics.get('factor_VAE_score')
    if not metric:
        return metric

    if 'mig' in metric_type:
        metric = metrics['MIG']['discrete_mig']
    elif 'dci_d' in metric_type:
        metric = metrics['dci_score']['disentanglement']
    elif 'dci_c' in metric_type:
        metric = metrics['dci_score']['completeness']
    elif 'dci_i_train' in metric_type:
        metric = metrics['dci_score']['informativeness_train']
    elif 'dci_i_test' in metric_type:
        metric = metrics['dci_score']['informativeness_test']
    elif 'fvae_train' in metric_type:
        metric = metrics['factor_VAE_score']['train_accuracy']
    elif 'fvae_test' in metric_type:
        metric = metrics['factor_VAE_score']['eval_accuracy']
    else:
        metric = None
    return metric


for metric_type in metric_type_list:
    print(metric_type)
    root_dir = 'exp_results'
    for model_name_i in model_list:
        model_name = model_name_i[0]
        new_result_dir = os.path.join(root_dir, model_name + '_average')
        if not os.path.exists(new_result_dir):
            os.makedirs(new_result_dir)
        new_result_path = os.path.join(new_result_dir, 'metric.txt')
        if os.path.exists(new_result_path):
            with open(new_result_path, 'r') as f:
                metrics_output = json.load(f)
        else:
            metrics_output = {}

        metrics_output[metric_type] = []

        for model_name_sub in model_name_i:
            result_path = os.path.join('logs', model_name_sub, result_name)
            if not os.path.exists(result_path):
                print(result_path, 'not exist')
                continue
            with open(result_path, 'r') as f:
                metrics = json.load(f)
            metric = get_metric(metrics, metric_type)
            metrics_output[metric_type].append(metric)

        metrics_output[metric_type + '_run_time'] = len(metrics_output[metric_type])
        print(len(metrics_output[metric_type]), ': ' + new_result_path)
        metrics_output[metric_type] = [float(np.mean(metrics_output[metric_type])),
                                       float(np.std(metrics_output[metric_type]))]

        with open(new_result_path, 'w') as f:
            json.dump(metrics_output, f, indent=4)
