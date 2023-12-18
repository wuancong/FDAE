"""
Based on: https://github.com/crowsonkb/k-diffusion
"""
import numpy as np
import torch as th
from piq import LPIPS
from . import dist_util
from .losses import content_decorrelation_loss, mask_entropy_loss
from .nn import mean_flat, append_dims, append_zero
from .random_util import get_generator


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        sigma_weight=1.0,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_weight = sigma_weight
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, sigmas, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        sigmas *= self.sigma_weight
        terms = {}

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised, condition_generator_output = self.denoise(model, x_t, sigmas, **model_kwargs)

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        content_decorrelation_weight = model_kwargs['content_decorrelation_weight']
        mask_entropy_weight = model_kwargs['mask_entropy_weight']
        semantic_code_str = model_kwargs['semantic_code_str']

        if content_decorrelation_weight > 0:
            group_num = len(condition_generator_output[semantic_code_str])
            feature = th.cat(condition_generator_output[semantic_code_str], dim=1)
            terms['content_decorrelation_loss'] = content_decorrelation_loss(feature, group_num)
        if mask_entropy_weight > 0:
            mask = condition_generator_output['mask']
            terms['mask_entropy_loss'] = mask_entropy_loss(mask.float())

        terms_weight = dict(content_decorrelation_loss=content_decorrelation_weight,
                            mask_entropy_loss=mask_entropy_weight)
        loss = 0.0
        for k in terms.keys():
            if k == 'xs_mse':
                continue
            if terms_weight.get(k) is not None:
                loss += terms[k] * terms_weight[k]
            else:
                loss += terms[k]
        terms['loss'] = loss

        return terms

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        condition_generator = model_kwargs.get('condition_generator')
        factors = model_kwargs.get('factors')
        y = model_kwargs.get('y')
        condition_idx = model_kwargs.get('condition_idx')
        if condition_generator is not None and factors is not None:
            condition_generator_output = condition_generator(factors, model_kwargs)
            condition_map = condition_generator_output['condition_map']
        else:
            condition_generator_output = None
            condition_map = None

        model_output = model(c_in * x_t, rescaled_t, y=y, condition_map=condition_map, condition_idx=condition_idx)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, condition_generator_output


def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    image_input=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    if image_input is not None:
        x_T = generator.randn(*shape, device=device) * sigma_max + image_input
    else:
        x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        # if i == indices[-1]:
        #     pass
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw the letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -th.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images
