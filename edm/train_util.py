import os
from evaluate_diffusion import evaluate_diffusion
import functools
import glob
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import copy

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler

import math

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        dataset_name,
        batch_size,
        microbatch,
        lr,
        log_interval,
        save_interval,
        eval_interval,
        resume_checkpoint,
        eval_only,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        condition_generator=None,
        update_condition_generator_only=False,
        debug_mode=False,
        max_step=10000,
        class_cond=False,
        condition_idx=[0],
        content_decorrelation_weight=-1.0,
        mask_entropy_weight=-1.0,
        train_args=None,
    ):
        self.content_decorrelation_weight = content_decorrelation_weight
        self.mask_entropy_weight = mask_entropy_weight
        self.semantic_code_str = train_args.semantic_code_str
        self.model = model
        self.condition_idx = condition_idx
        self.condition_generator = condition_generator
        self.update_condition_generator_only = update_condition_generator_only
        self.max_step = max_step
        self.diffusion = diffusion
        self.data = data
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.resume_checkpoint = resume_checkpoint
        self.eval_only = eval_only
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.train_args = train_args

        if th.cuda.is_available() and not debug_mode:
            self.use_ddp = True
            self.global_batch = self.batch_size * dist.get_world_size()
        else:
            self.use_ddp = False
            self.global_batch = self.batch_size
        self.step = 0
        self.resume_step = 0
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            condition_generator=self.condition_generator,
            update_condition_generator_only=update_condition_generator_only,
            class_cond=class_cond,
        )

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if update_condition_generator_only:
            self.resume_step = 0
        if self.resume_step:
            self._load_optimizer_state()

        if self.use_ddp:
            if self.update_condition_generator_only:
                self.ddp_model = self.model
            else:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            if self.condition_generator is not None:
                self.ddp_conditional_generator = DDP(
                    self.condition_generator,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                self.ddp_conditional_generator = None
        else:
            self.ddp_model = self.model
            self.ddp_conditional_generator = self.condition_generator

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if self.use_ddp:
                if dist.get_rank() == 0:
                    logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                    self.model.load_state_dict(
                        dist_util.load_state_dict(
                            resume_checkpoint, map_location=f'cuda:{dist_util.dev()}'
                        ), strict=False
                    )
                    if self.condition_generator is not None:
                        self.condition_generator.load_state_dict(
                            dist_util.load_state_dict(
                                resume_checkpoint, map_location=f'cuda:{dist_util.dev()}'
                            ), strict=False
                        )
            else:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    ), strict=False
                )
                if self.condition_generator is not None:
                    self.condition_generator.load_state_dict(
                        dist_util.load_state_dict(
                            resume_checkpoint, map_location=dist_util.dev()
                        ), strict=False
                    )
        if self.use_ddp:
            dist_util.sync_params(self.model.parameters())
            dist_util.sync_params(self.model.buffers())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if self.eval_only:
            print(f'testing {self.resume_checkpoint}')
            evaluate_diffusion(self.dataset_name, self.train_args, os.path.split(self.resume_checkpoint)[0],
                               self.ddp_conditional_generator, self.resume_step)
            return
        while not self.lr_anneal_steps or self.step < self.lr_anneal_steps:
            if self.step >= self.max_step:
                break
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.step % self.eval_interval == 0:
                evaluate_diffusion(self.dataset_name, self.train_args, get_blob_logdir(), self.ddp_conditional_generator, self.step)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
        if self.step % self.eval_interval != 0:
            evaluate_diffusion(self.dataset_name, self.train_args, get_blob_logdir(), self.ddp_conditional_generator, self.step)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self.step += 1
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            model_kwargs_input = micro_cond
            if self.condition_generator is not None:
                model_kwargs_input['condition_generator'] = self.condition_generator
            model_kwargs_input['condition_idx'] = self.condition_idx
            model_kwargs_input['content_decorrelation_weight'] = self.content_decorrelation_weight
            model_kwargs_input['mask_entropy_weight'] = self.mask_entropy_weight
            model_kwargs_input['semantic_code_str'] = self.semantic_code_str
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=model_kwargs_input,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                if self.update_condition_generator_only:
                    with self.ddp_conditional_generator.no_sync():
                        losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        # linear lr decay
        # frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        # lr = self.lr * (1 - frac_done)
        # cosine lr decay
        lr = self.lr * (1 + math.cos(math.pi * self.step / self.lr_anneal_steps)) / 2
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"saving model...")
                filename = f"model{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        if not dist.is_initialized() or dist.get_rank() == 0:
            # remove old opt files
            files = glob.glob(os.path.join(get_blob_logdir(), 'opt*.pt'))
            for file in files:
                os.remove(file)
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, self.mp_trainer.master_params)
        if dist.is_initialized():
            dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
