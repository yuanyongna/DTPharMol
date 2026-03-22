import copy
import os
import functools
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import io
from diffumol.utils import dist_util, logger
from diffumol.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from diffumol.utils.nn import update_ema
from diffumol.step_sample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        tip_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        learning_steps=0,
        checkpoint_path="",
        gradient_clipping=-1.0,
        eval_data=None,  # 默认传入 data_valid
        eval_interval=-1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        # 处理 ema_rate, 确保为列表格式
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.tip_interval = tip_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping
        # 初始化训练步骤、全局批次、模型参数等
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self.checkpoint_path = checkpoint_path  # DEBUG **
        # 加载并同步模型参数
        self._load_and_sync_parameters()
        # 如果使用混合精度, 设置相关配置
        if self.use_fp16:
            self._setup_fp16()
        # 初始化 AdamW 优化器
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        # 如果存在恢复步骤, 调整学习率并加载 EMA 参数, 否则初始化 EMA 参数
        if self.resume_step:
            # self._load_optimizer_state()
            frac_done = (self.step + self.resume_step) / self.learning_steps
            lr = self.lr * (1 - frac_done)
            self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]
        # 检查 CUDA 是否可用, 以确定是否使用分布式数据并行（DDP）, 并初始化 DDP 模型
        if th.cuda.is_available():  # DEBUG **
            self.use_ddp = True
            # print(dist_util.dev())
            # DDP 是 PyTorch 的 torch.nn.parallel.DistributedDataParallel 类，用于在多个 GPU 上并行训练模型
            self.ddp_model = DDP(
                self.model,  # 这是要进行分布式训练的模型实例
                device_ids=[dist_util.dev()],  # 指定使用的 GPU 设备
                output_device=dist_util.dev(),  # 指定输出张量的设备，通常与 device_ids 相同
                broadcast_buffers=False,  # 指定是否在每个训练步骤中广播模型的缓冲区
                bucket_cap_mb=128,  # 设置每个 bucket 的最大容量（以 MB 为单位），用于控制 DDP 内部的梯度聚合
                find_unused_parameters=False,  # 指定是否查找未使用的参数
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        """
        从指定的恢复检查点加载模型的参数
        在分布式训练环境中，确保所有进程的模型参数同步
        通过解析检查点文件名，获取恢复步骤信息，以便在训练时继续从相应的步骤开始
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint[-3:] == ".pt":
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        actual_model_path(resume_checkpoint),
                        map_location=dist_util.dev(),
                    )
                )
        dist_util.sync_params(self.model.parameters())

    def _setup_fp16(self):
        # make_master_params 函数的目的是将模型参数复制到一个新的列表中，并将这些参数转换为全精度（float32）格式
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                print(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    actual_model_path(ema_checkpoint), map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if bf.exists(main_checkpoint):
            print(f"loading optimizer state from checkpoint: {main_checkpoint}")
            state_dict = dist_util.load_state_dict(
                actual_model_path(main_checkpoint), map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        model_path = None
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            if (self.step + 1) % self.tip_interval == 0:
                print(f"第 {self.step + 1}/{self.learning_steps} 次迭代")

            # print("self.data:", self.data)
            batch, cond = next(self.data)

            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.eval_data is not None and self.step % self.eval_interval == 0:
                # batch_eval 指该批次验证集的嵌入模型权重
                # cond_eval 指该批次验证集的内容，包括 input_ids 和 input_mark
                batch_eval, cond_eval = next(self.eval_data)
                self.forward_only(batch_eval, cond_eval)
                print("在验证集上评估:")

                logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                # 在集成测试中运行有限的时间
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            # print("本次训练迭代结束")
        if (self.step - 1) % self.save_interval != 0:
            model_path = self.save()

        return model_path

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_normal()
        self.log_step()

    def forward_only(self, batch, cond):

        # print("batch_eval 指该批次验证集 input_ids 的嵌入向量, cond_eval 指该批次验证集原始 input_ids 和 input_mark 的内容")
        # print("batch_eval: ", batch.shape)
        # print("cond_eval: ", type(cond), " batch_size=", len(cond["input_ids"]))
        # for key, value in cond.items():
        #     print(f"{key}——>{value.shape if hasattr(value, 'shape') else len(value)}")

        with th.no_grad():
            zero_grad(self.model_params)  # 将模型参数的梯度清零
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i : i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(
                    micro.shape[0], dist_util.dev()
                )
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()
                log_loss_dict(
                    self.diffusion,
                    t,
                    {f"eval_{k}": v * weights for k, v in losses.items()},
                )

    def forward_backward(self, batch, cond):
        """
        该方法负责执行模型的前向传播和反向传播过程，计算损失并更新模型参数。
        """
        # print("batch 指该批次训练集的 input_ids 的嵌入张量, cond 指该批次训练集 input_ids 和 input_mask 的原始内容")
        # print("batch: ", type(batch), batch.shape)
        # print("cond: ", type(cond), " 包含的键: ")
        # for key, value in cond.items():
        #     print(f"{key}——>{value.shape if hasattr(value, 'shape') else len(value)}")

        # 在每次训练前清零模型参数的梯度，以确保每个批次的梯度是独立计算的
        zero_grad(self.model_params)

        # 使用循环将输入批次分成多个微批次
        # batch.shape[0] 就是 batch_size 默认为 512, microbatch 默认为 64
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(
                dist_util.dev()
            )  # 从 batch 中提取当前的微批次，并将其移动到适当的设备（如 GPU）
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }  # 从 cond 中提取与当前微批次对应的原始数据信息
            last_batch = (i + self.microbatch) >= batch.shape[
                0
            ]  # 判断当前是否为最后一个微批次，以便在计算时做适当处理
            # schedule_sampler 默认为 "lossaware", 使用 schedule_sampler 确定当前微批次的采样时间步长 t 和对应的权重 weights
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # print(f"\n\n\nlossaware 采样器获取时间步长 t ({t.shape}) 和对应的权重 weights ({weights.shape})")

            # 使用 functools.partial 创建一个部分应用的函数 compute_losses，该函数用于计算损失，只需提供微批次和时间步
            # functools.partial 用于创建 self.diffusion.training_losses 一个新的可调用对象（函数），
            # 该对象将原函数的一部分参数固定为特定值, 这在需要多次调用同一函数但某些参数保持不变时非常有用。
            # print("输入微批次 micro 和采样时间步长 t, 调用 compute_losses 函数计算 losses")
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            # 如果是最后一个微批次或不使用分布式数据并行（DDP），直接计算损失
            # 否则，在 no_sync() 上下文管理器中计算损失，以避免在分布式训练中同步梯度，从而提高效率。
            if last_batch or not self.use_ddp:
                losses = compute_losses()
                # print("\n最后一个微批次的 losses: ", losses)
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
                    # print("\n该微批次的 losses: ", losses)
            # 如果使用的是 LossAwareSampler，则更新采样器，以便根据当前的损失信息调整采样策略。
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # 计算加权损失，使用之前采样的权重
            loss = (losses["loss"] * weights).mean()
            # 调用 log_loss_dict 函数记录当前的损失信息
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # 执行反向传播，计算梯度以更新模型参数
            loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            print(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2**self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm = self.gradient_clipping  # 3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            # print(cnt, p) ## DEBUG
            # print(cnt, p.grad)
            # cnt += 1
            if p.grad != None:
                sqsum += (p.grad**2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = (self.step + self.resume_step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                print(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                # 构建保存路径
                local_path = bf.join(self.checkpoint_path, filename)
                blob_path = bf.join(get_blob_logdir(), filename)
                print("writing to", blob_path)
                print("writing to", local_path)
                # 使用 BlobFile 保存模型
                with bf.BlobFile(local_path, "wb") as f:
                    th.save(state_dict, f)  # save locally
                return local_path  # 返回保存的本地路径
            return None  # 如果不是主进程，返回 None

        model_path = None
        for rate, params in zip(self.ema_rate, self.ema_params):
            model_path = save_checkpoint(rate, params)
        dist.barrier()
        return model_path  # 返回保存的本地路径

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params  # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == ".pt":
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    记录损失字典中每个损失的平均值
    计算并记录每个损失值在不同时间步的分位数，特别是四分位数
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def actual_model_path(model_path):
    return model_path
