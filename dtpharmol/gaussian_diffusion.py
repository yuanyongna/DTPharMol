"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import numpy as np
import torch as th
import sys

sys.path.append(".")
import torch.nn.functional as F
from .utils.nn import mean_flat
from .utils.losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "pw_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps:
        if True, pass floating point timesteps into the model so that they are always scaled like in the original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        rescale_timesteps=False,
        num_props=0,
        ppgraph_len=0,
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl
        self.num_props = num_props
        self.ppgraph_len = ppgraph_len

        # Use float64 for accuracy.
        # betas 来自于 get_named_beta_schedule 函数，
        # 此函数根据给定的噪声调度（noise_schedule）和扩散步骤数（diffusion_steps）生成对应的 beta 值数组
        # beta 值在开始时接近于 1，随着 t 的增加，beta 值会逐渐减小
        # 将 betas 转换为 NumPy 数组，并指定数据类型为 float64
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # 将 betas 数组的形状的第一个维度（即 beta 值的数量）赋值给 self.num_timesteps，表示扩散模型的总时间步数
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(
            1.0, self.alphas_cumprod[:-1]
        )  # 1,a1,a1a2,a1a2a3...
        self.alphas_cumprod_next = np.append(
            self.alphas_cumprod[1:], 0.0
        )  # a1a2,a1a2a3... 0.0
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  # 开根号
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(
            1.0 - self.alphas_cumprod
        )  # 1-根号下...
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mapping_func = None  # implement in train main()
        self.add_mask_noise = False  # TODO

    def training_losses(self, model, *args, **kwargs):
        # print("\n进入父类 GaussianDiffusion 类的 training_losses 方法")
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            # # 将时间步转换为浮点数张量，以确保后续的数值计算不会因为整数类型而丢失精度
            # 之后计算缩放因子。这通常将时间步的范围线性扩展到 [0, 1000]，其中 self.num_timesteps 是总的时间步数
            # 这样处理后，可以使得模型在更大的时间范围内进行训练或推理
            # return t.float() * (1000.0 / self.num_timesteps)
            tmp = t.float() * (1000.0 / self.num_timesteps)
            return tmp
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.

        对数据进行扩散，生成带噪声的样本
        :param x_start: 初始数据批次, 表示在扩散过程的起始状态
        :param t: 扩散步骤的数量 (减去 1), 例如, t=0 表示进行一次扩散
        :param noise: 可选参数，如果指定，将使用提供的噪声；如果未提供，则生成与 x_start 形状相同的随机噪声
        :param mask: 可选参数，用于锚定特定位置。具体来说，它指示哪些位置应该保留原始数据，哪些位置应该替换为噪声样本
        :return: 带噪声的 x_start
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        # 用 _extract_into_tensor 函数从累积的平方根 alpha 和 1 - alphas_cumprod 中提取相应的值
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

        # 如果 mask 为 None，直接返回带噪声的样本 x_t
        # 如果提供了 mask，将其扩展到与 x_start 相同的形状，如果 mask 中的值为 0，则保留原始数据；否则使用带噪声的样本
        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(
            -1
        )  # B 是批量大小（样本数），C 是最后一个维度的大小（通常是通道数）
        # print('验证，在 p_mean_variance 函数中 x 的形状:', x.shape)
        assert t.shape == (B,)  # 确保时间步 t 的形状与批量大小相同
        # 将输入张量 x 和经过缩放的时间步 t 传递给模型，获取模型输出
        # _scale_timesteps 函数将时间步 t 转换为浮点数张量并范围线性扩展到 [0, 1000]
        # model_kwargs 默认为空字典
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood.
        # 对于 fixedlarge，我们像这样设置初始（对数）方差以获得更好的解码器对数似然。
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(
            np.append(self.posterior_variance[1], self.betas[1:])
        )

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        top_p=None,
        mask=None,
        x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:  # 默认为 5
            # print('top_p sampling')
            noise = th.randn_like(x)  # 创建一个与 x 形状相同的随机噪声张量
            replace_mask = th.abs(noise) > top_p  # 标记噪声值的绝对值大于 top_p 的位置
            # 如果替换掩码中有值，则重新生成这些位置的噪声，直到所有噪声值的绝对值都小于或等于 top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (
                th.abs(noise) <= top_p
            ).all()  # 确保所有噪声值的绝对值都不超过 top_p
        else:  # 如果 top_p 为 None 或小于等于 0，则直接生成与 x 形状相同的随机噪声
            noise = th.randn_like(x)
        # 这个掩码用于控制在时间步 t 等于 0 时，不添加噪声。它的形状将与 x 适配
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # 通过使用均值和方差计算最终样本
        # out["mean"]：从 p_mean_variance 返回的均值
        # nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise: 当 t 不为 0 时，添加噪声，噪声的标准差由方差的对数值决定
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # 如果 mask 为 None，则不进行任何处理
        if mask == None:
            pass
        # 否则，使用 th.where 将对应于 mask 中值为 0 的位置替换为 x_start，即保留原始数据
        else:
            sample = th.where(mask == 0, x_start, sample)
        # 返回一个字典，包含：
        # "sample": 生成的样本
        # "pred_xstart": 对 x_0 的预测
        # "greedy_mean": 直接返回的均值
        # "out": 包含其他输出信息的完整结果
        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out,
        }

    def p_sample_loop(
        self,
        model,  # 调用 create_model_and_diffusion 方法创建 Transformer 模型, 并加载了之前训练过程最终得到的参数权重文件
        shape,  # sample_shape = (x_start.shape[0], args.seq_len-args.num_props+1, args.hidden_dim)
        noise=None,  # 将 x_start 中格式化的 SMILES 部分及之后的填充部分用噪声替换, 性质与骨架部分保留原始值, 得到 x_noised
        clip_denoised=True,  # 默认设置为 False
        # 嵌入层 model_emb 初始化自 torch.nn.Embedding, 并将训练过程得到的的嵌入层权重 model.word_embedding.weight 克隆至 model_emb
        # denoised_fn_round 函数定义在 round.py 文件中, 就是该扩散模型去噪过程所需要遵循的方法
        # 该函数通过计算模型嵌入 (model_emb) 和文本嵌入 (text_emb) 之间的 K 最近邻 (KNN), 从而选择最相似的嵌入
        # 返回更新后的嵌入, 以便在生成或去噪过程中使用
        # partial 函数: 来自 functools 模块的一个函数, 用于固定某些参数并返回一个新的函数
        # 在这里, denoised_fn 被定义为调用 denoised_fn_round 函数的一个固定版本, 其中 args 和 model_emb 被预先填充
        # 这使得在调用 sample_fn 时可以直接使用该函数，而无需每次都传递这些参数
        denoised_fn=None,
        model_kwargs=None,  # 默认为空字典 {}
        device=None,  # 未传递
        progress=False,  # 未传递
        top_p=None,  # 默认为 5
        clamp_step=None,  # 默认为 0
        clamp_first=None,  # 默认为 True
        mask=None,  # 与 x_noised 对应的掩码，0 对应的保留，1 对应的替换为高斯噪声
        x_start=None,  # 没有被使用随机噪声替换的原始数据
        gap=1,  # 默认为 1
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on. If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start,
        ):
            final.append(sample["sample"])
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:  # custom your the start point of x_0
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)

        # self.num_timesteps 是betas 数组的形状的第一个维度，也就是扩散模型的总时间步数
        # 之后生成一个从 0 到 self.num_timesteps - 1 的列表
        # [::-1] 将列表反转，因此 indices 将包含从 self.num_timesteps - 1 到 0 的索引
        indices = list(range(self.num_timesteps))[::-1]

        if progress:  # 默认为 False
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:  # from T to 0
            # 创建一个 PyTorch 张量 t，其内容为当前索引 i，长度与 shape[0] 相同
            # 这个张量用于表示当前时间步
            t = th.tensor([i] * shape[0], device=device)
            # 如果 clamp_first 为 False，且当前时间步 i 大于 clamp_step，则设置 denoised_fn_cur 为 None，这意味着在这一阶段不进行去噪处理
            # 否则，denoised_fn_cur 设置为 denoised_fn，表示在当前时间步进行去噪。
            if not clamp_first:  # 默认为 True
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            # 如果 clamp_first 为 True，且当前时间步 i 大于或等于 clamp_step，则使用 denoised_fn 进行去噪
            else:
                if i >= clamp_step:  # 默认为 0
                    # 是一个新的函数，只需要提供 text_emb 和 t 参数来调用 denoised_fn_round 函数，其余参数已被固定
                    # denoised_fn_round: 通过将输入的文本嵌入替换为与模型嵌入最接近的嵌入，从而对嵌入进行“取整”
                    denoised_fn_cur = denoised_fn
                # 否则，将 denoised_fn_cur 设置为 None，表示不进行去噪处理
                else:
                    denoised_fn_cur = None
            with th.no_grad():  # 在 with 语句块中执行的操作不会计算梯度
                out = self.p_sample(
                    model,
                    sample_x,  # 符合模型形状的数据，属性与骨架对应的值保留为原始数值，其余部分替换为随机高斯噪声
                    t,  # 是一个 PyTorch 张量，内容为当前索引 i，长度与 shape[0] 相同，i 从 T 递减至 1，这个张量用于表示当前时间步
                    clip_denoised=clip_denoised,  # 默认为 False
                    denoised_fn=denoised_fn_cur,  # 当前时间步的去噪函数，可能为 None 或 denoised_fn
                    model_kwargs=model_kwargs,  # 默认为空字典
                    top_p=top_p,  # 默认为 5
                    mask=mask,  # 与 x_noised 对应的掩码，0 对应的保留，1 对应的替换为高斯噪声
                    x_start=x_start,  # 没有被使用随机噪声替换的原始数据
                )
                yield out  # 使用 yield 语句将生成的样本返回。这使得该函数成为一个生成器，可以逐步生成样本而不占用过多内存
                sample_x = out[
                    "sample"
                ]  # 将生成的样本 out["sample"] 更新为 sample_x，为下一个时间步的迭代做准备

    def _get_x_start(self, x_start_mean, std):
        """
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        """
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return x_start_mean + std * noise

    def _token_discrete_loss(
        self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None
    ):
        """
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        """
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        loss_fct = th.nn.CrossEntropyLoss(reduction="none")
        decoder_nll = (
            loss_fct(
                logits.contiguous().view(-1, logits.size(-1)),
                input_ids.contiguous().view(-1),
            )
            .contiguous()
            .view(input_ids.shape)
        )
        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:  # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {"pred_xprev": pred_prev, "pred_xstart": pred_xstart}

    def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):
        # print("\n进入 GaussianDiffusion 类的 training_losses_seq2seq 方法")
        # print("参数: 初始化的经过多进程函数处理过的 Transformer 模型 model: ", type(model))
        # print("参数: 嵌入后的分子序列 x_start: ", x_start.shape)
        # print(f"参数: 时间步索引列表 t (len={len(t)}): {t}")
        # print("参数: 原始分子序列与掩码 model_kwargs:")
        # for key, value in model_kwargs.items():
        #     print(f"{key} (len={value.shape}):\n{value[:3]}\n......")
        # print("参数: 指定需要删除的噪声 noise: ", noise)
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.

        model -> model（直接传递）
        *args -> (x_start, t, model_kwargs, noise)（具体取决于如何传递这些参数）
        """

        # x_start_fix = x_start # save the orignal x_0
        assert "input_ids" in model_kwargs

        # print("参数 ppgraph_len: ", self.ppgraph_len)
        # print("参数 num_props: ", self.num_props)
        if self.ppgraph_len and self.num_props:
            input_ids_mask = (
                model_kwargs.pop("input_mask")
                .to(t.device)[:, (self.ppgraph_len + self.num_props - 1) :]
                .contiguous()
            )
        elif self.num_props:
            input_ids_mask = (
                model_kwargs.pop("input_mask")
                .to(t.device)[:, self.num_props - 1 :]
                .contiguous()
            )
        elif self.ppgraph_len:
            input_ids_mask = (
                model_kwargs.pop("input_mask")
                .to(t.device)[:, self.ppgraph_len - 1 :]
                .contiguous()
            )
        else:
            input_ids_mask = model_kwargs.pop("input_mask").to(t.device)
        input_ids_x = (
            model_kwargs.pop("input_ids").to(th.float).to(t.device)
        )  # seq对应的字典序号
        # print("\n提取出 input_ids_x:", input_ids_x.shape)
        # print("提取出 input_ids_mask (将属性对应的前几列除外):", input_ids_mask.shape)

        # input_ids_mask = model_kwargs.pop('input_mask').to(t.device) # seq中条件部分和生成部分
        if self.ppgraph_len and self.num_props:
            props = input_ids_x[:, : self.ppgraph_len + self.num_props].clone()
            # print("从 input_ids_x 中提取出 props: ", props.shape)
            props = model.model.module.get_props(props)
            # print("经过 model.model.module.get_props() 处理后的 props: ", props.shape)
            others = model.model.module.get_embeds(
                input_ids_x[:, self.ppgraph_len + self.num_props :]
            )
            # print("经过 model.model.module.get_embeds() 处理后的其他部分: ", others.shape)
            x_start_mean = th.cat([props, others], 1)
            # print("将处理后的 props 与其他部分连接在一起得到 x_start_mean: ", x_start_mean.shape)
        elif self.num_props:
            props = input_ids_x[:, : self.num_props].clone()
            # print("从 input_ids_x 中提取出 props: ", props.shape)
            props = model.model.module.get_props(props)
            # print("经过 model.model.module.get_props() 处理后的 props: ", props.shape)
            others = model.model.module.get_embeds(input_ids_x[:, self.num_props :])
            # print("经过 model.model.module.get_embeds() 处理后的其他部分: ", others.shape)
            x_start_mean = th.cat([props, others], 1)
            # print("将处理后的 props 与其他部分连接在一起得到 x_start_mean: ", x_start_mean.shape)
        elif self.ppgraph_len:
            props = input_ids_x[:, : self.ppgraph_len].clone()
            # print("从 input_ids_x 中提取出 pp_graph: ", props.shape)
            props = model.model.module.get_props(props)
            # print("经过 model.model.module.get_props() 处理后的 pp_graph: ", props.shape)
            others = model.model.module.get_embeds(input_ids_x[:, self.ppgraph_len :])
            # print("经过 model.model.module.get_embeds() 处理后的其他部分: ", others.shape)
            x_start_mean = th.cat([props, others], 1)
            # print("将处理后的 pp_graph 与其他部分连接在一起得到 x_start_mean: ", x_start_mean.shape)
        else:
            x_start_mean = model.model.module.get_embeds(
                input_ids_x
            )  # Transformer中的embed层，之前的完全不需要，可以删除
            # print("没有属性条件时直接将 input_ids_x 使用 model.model.module.get_embeds 进行处理得到 x_start_mean: ", x_start_mean.shape)

        # nn.Linear(self.num_props, self.input_dims)
        std = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]).to(x_start_mean.device),
            x_start_mean.shape,
        )
        # print("使用函数 _extract_into_tensor 获取 std: ", std.shape)
        # x_start_log_var = 2 * th.log(std)
        x_start = self._get_x_start(x_start_mean, std)  # 一开始的均值和方差
        # print("使用参数 x_start_mean 和 std, 使用函数 _get_x_start 计算 x_start: ", x_start.shape)

        if noise is None:  # 获得噪声
            noise = th.randn_like(x_start)
        # print("初始化噪声 noise: ", noise.shape)

        x_t = self.q_sample(
            x_start, t, noise=noise, mask=input_ids_mask
        )  # reparametrization trick. 重参数化，就是进行t时刻采样
        # print("使用参数 x_start, input_ids_mask, noise 和 t, 使用函数 q_sample 获取 x_t: ", x_t.shape)

        get_logits = model.model.module.get_logits  # nll计算
        # print("使用函数 model.model.module.get_logits 获取 get_logits: ", type(get_logits))
        terms = {}
        target = x_start
        # print(f"t: {t.dtype}\n{t}")
        # print("rescale_timesteps: ", self.rescale_timesteps)
        tmp = self._scale_timesteps(t)
        # print(f"tmp: {tmp.dtype}\n{tmp}")
        model_output = model(x_t, tmp, **model_kwargs)  # 根据采样后结果，进行开始预测
        # print("使用参数 x_t, self._scale_timesteps(t), 调用函数 model 获取 model_output: ", model_output.shape)
        assert model_output.shape == target.shape == x_start.shape

        terms["mse"] = mean_flat((target - model_output) ** 2)  # 预测和输出的均方误差
        # print("得到 terms['mse']: ", terms["mse"].shape)
        model_out_x_start = self._x0_helper(model_output, x_t, t)[
            "pred_xstart"
        ]  # predicted_xstart = model_output
        # print("得到 model_out_x_start:", model_out_x_start.shape)
        t0_mask = t == 0
        t0_loss = mean_flat(
            (x_start_mean - model_out_x_start) ** 2
        )  # 这是真实和预测的均值
        # print("得到 t0_loss: ", t0_loss.shape)
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])
        # print("更新 terms['mse']: ", terms["mse"])
        # tT_mask = (t == self.num_timesteps - 1)
        out_mean, _, _ = self.q_mean_variance(
            x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device)
        )
        tT_loss = mean_flat(out_mean**2)
        # print("得到 tT_loss: ", tT_loss)

        if self.ppgraph_len and self.num_props:
            decoder_nll = self._token_discrete_loss(
                x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.ppgraph_len + self.num_props :].to(th.long),
            )
            terms["nll"] = self._token_discrete_loss(
                model_out_x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.ppgraph_len + self.num_props :].to(th.long),
                mask=input_ids_mask[:, 1:],
                truncate=True,
                t=t,
            )
        elif self.num_props:
            decoder_nll = self._token_discrete_loss(
                x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.num_props :].to(th.long),
            )  # embedding regularization
            terms["nll"] = self._token_discrete_loss(
                model_out_x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.num_props :].to(th.long),
                mask=input_ids_mask[:, 1:],
                truncate=True,
                t=t,
            )  # x_0->model_out_x_start
        elif self.ppgraph_len:
            decoder_nll = self._token_discrete_loss(
                x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.ppgraph_len :].to(th.long),
            )  # embedding regularization
            terms["nll"] = self._token_discrete_loss(
                model_out_x_start[:, 1:, :],
                get_logits,
                input_ids_x[:, self.ppgraph_len :].to(th.long),
                mask=input_ids_mask[:, 1:],
                truncate=True,
                t=t,
            )  # x_0->model_out_x_start
        else:
            decoder_nll = self._token_discrete_loss(
                x_start, get_logits, input_ids_x.to(th.long)
            )  # embedding regularization
            terms["nll"] = self._token_discrete_loss(
                model_out_x_start,
                get_logits,
                input_ids_x.to(th.long),
                mask=input_ids_mask,
                truncate=True,
                t=t,
            )  # x_0->model_out_x_start
        # print("得到 decoder_nll: ", decoder_nll.shape)
        # print("得到 terms['nll']: ", terms["nll"].shape)

        # assert (model.lm_head.weight == model.word_embedding.weight).all()
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss
        return terms

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample = langevin_fn(
                sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x
            )

        if mask == None:
            pass
        else:
            sample = th.where(mask == 0, x_start, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap=gap,
        ):
            final.append(sample["sample"])
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        print("Gap value:", gap)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start,
                )
                yield out
                sample_x = out["sample"]


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.

    根据提供的索引从一维 NumPy 数组中提取值，然后将结果调整为指定的广播形状
    :param arr: 一维 NumPy 数组
    :param timesteps: 用于提取数组中索引的张量
    :param broadcast_shape: 更大的 K 维形状，其中批次维度等于 timesteps 的长度
    :return: 形状为 [batch_size, 1, ...] 的张量，形状有 K 维
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
        # print(kwargs.keys())
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self,
        model,  # 这是 self.ddp_model，在 compute_losses 中固定的第一个参数
        *args,  # 这是一个可变参数，接收 micro 和 t，它们在 compute_losses 中作为第二和第三个位置参数传递
        **kwargs,  # 这里接收关键字参数，model_kwargs=micro_cond 将作为关键字参数传递
    ):
        # print("\n进入 SpacedDiffusion 类的 training_losses 方法")
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            # print("model 的类型为 _WrappedModel, 直接返回 model")
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = (
            timestep_map  # 一个映射表，定义了如何从输入时间步 ts 转换到模型所需的时间步
        )
        self.rescale_timesteps = (
            rescale_timesteps  # 布尔值，指示是否对时间步进行重新缩放
        )
        self.original_num_steps = (
            original_num_steps  # 原始时间步的数量，通常用于计算缩放因子
        )

    # 特殊方法，使得 _WrappedModel 实例可以像函数一样被调用
    def __call__(self, x, ts, **kwargs):
        # print("\nmodel 的类型不是 _WrappedModel, 需要调用 _WrappedModel 类方法处理 model")
        # print("参数: model: ", type(self.model))
        # print(f"参数: 初始的时间步映射表 timestep_map (len={len(self.timestep_map)}): ", self.timestep_map[:21], "······")
        # print("参数: 是否需要缩放时间步 (rescale_timesteps): ", self.rescale_timesteps)
        # print(f"参数--原始时间步的数量即 beta 数组的大小 (original_num_steps): {self.original_num_steps}")
        # print("参数: 输入序列 x: ", x.shape)
        # print(f"参数: 时间步索引列表 ts (len={len(ts)}): ", ts)
        # print("其他参数 kwargs:")
        # for key, value in kwargs.items():
        #     print(f"{key}——>{value.shape if hasattr(value, 'shape') else len(value)}")

        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # print(f"将 timestep_map 转换为一个张量 map_tensor (shape={map_tensor.shape}):\n{map_tensor}")
        new_ts = map_tensor[ts]
        # print(f"使用 map_tensor[ts] 来获取新的时间步 new_ts (shape={new_ts.shape}):\n{new_ts}")
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
            # print(f"进行缩放后的时间步 new_ts (shape={new_ts.shape}):\n{new_ts}")

        temp = self.model(x, new_ts, **kwargs)
        # print(f"经过 self.model(x, new_ts, **kwargs) 预测后的返回值 temp (shape={temp.shape}):\n{temp}")
        # print("_WrappedModel 类方法结束\n\n")
        # 最后，调用模型 self.model，并将输入数据 x 和变换后的时间步 new_ts 传递给它，同时保留其他任何额外的关键字参数
        return temp
