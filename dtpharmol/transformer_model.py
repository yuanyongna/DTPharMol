from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.
    具有注意力机制和时间步嵌入的完整 Transformer 模型。
    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,  # 未传递
        config_name="bert-base-uncased",  # 默认 "./datasets/model.json"
        vocab_size=None,
        init_pretrained="no",
        logits_mode=1,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            # 从指定的预训练模型名称或路径加载模型的配置文件, 返回一个配置对象
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims  # 默认 128
        self.hidden_t_dim = hidden_t_dim  # 来自于 config.json，默认为 128
        self.output_dims = output_dims  # 默认 128
        self.dropout = dropout  # 默认 0.1
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size  # 来自于 model.json，默认为 768
        self.num_props = kwargs["num_props"]
        self.ppgraph_len = kwargs["ppgraph_len"]
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)

        # print(self.ppgraph_len, self.num_props)
        if self.ppgraph_len and self.num_props:
            self.prop_nn = nn.Linear(self.num_props + self.ppgraph_len, self.input_dims)
        elif self.num_props:
            self.prop_nn = nn.Linear(self.num_props, self.input_dims)
        elif self.ppgraph_len:
            self.prop_nn = nn.Linear(self.ppgraph_len, self.input_dims)

        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        # hidden_t_dim 参数根源来自于参数文件 config.json，默认为 128，它代表时间步嵌入的初始维度
        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # 输入的维度 self.input_dims 与配置文件中的隐藏层大小 config.hidden_size 如果不相等，需要进行维度转换
        if self.input_dims != config.hidden_size:
            # 定义一个序列模型，用于将输入的维度转换为隐藏层大小
            self.input_up_proj = nn.Sequential(
                nn.Linear(input_dims, config.hidden_size),
                nn.Tanh(),
                # 这一层通常用于增加模型的表达能力，而不改变输出的维度
                nn.Linear(config.hidden_size, config.hidden_size),
            )

        if init_pretrained == "bert":  # init_pretrained 默认为 no
            # print('使用预训练的 BERT 模型进行初始化')
            # print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            self.input_transformers = temp_bert.encoder
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
            )
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm
            del temp_bert.embeddings
            del temp_bert.pooler
        elif init_pretrained == "no":  # 默认为 no
            # BertEncoder 类实现了 BERT 模型的编码器，负责处理输入的 token，并生成其对应的隐藏状态表示
            # print("config: ", config)
            self.input_transformers = BertEncoder(config)
            # config.max_position_embeddings 是超参数，来自于 ./datasets/model.json，默认为 512, 通常用于限制序列的最大长度
            # 生成一个从 0 到 config.max_position_embeddings - 1 的一维张量, 再将生成的张量的形状扩展为 (1, 512)，即增加一个维度
            # 这里的 -1 表示保持原始大小，而 1 表示增加一个新的维度, 这使得张量可以在批量处理时使用
            # 使用 register_buffer 方法将 position_ids 注册为模型的缓冲区,
            # 缓冲区是模型的一部分，但不会在优化过程中更新（不会被视为模型的可训练参数）, 这对于存储模型的状态信息（如位置 ID）非常有用
            # 最终，self.position_ids 将是一个形状为 (1, 512) 的张量，包含从 0 到 511 的位置 ID
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
            )
            # config.max_position_embeddings 是超参数，来自于 ./datasets/model.json，默认为 512
            # config.hidden_size 是超参数，来自于 ./datasets/model.json，默认为 768
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            # config.layer_norm_eps 是超参数，来自于 ./datasets/model.json，默认为 1e-12
            # eps 控制层归一化（Layer Normalization）中的数值稳定性，用于防止在计算归一化时出现除以零的情况
            # nn.LayerNorm：PyTorch 的层归一化模块，用于对输入的每个样本进行归一化处理，用于提高模型的训练稳定性和加快收敛速度，
            # 它通过对每个样本的特征进行归一化，使得每个样本的均值为 0，标准差为 1，从而减少内部协变量偏移（internal covariate shift）
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            assert False, "init_pretrained 参数的类型不正确"

        # config.hidden_dropout_prob 是超参数，来自于 ./datasets/model.json，默认为 0.1
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # output_dims 默认为 128，hidden_size 默认为 768
        # 用于将模型的隐藏状态从 config.hidden_size（768）转换为所需的输出维度 self.output_dims（128）
        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, self.output_dims),
            )

    def get_embeds(self, input_ids):
        # print(input_ids.size())
        # print(self.word_embedding)
        return self.word_embedding(input_ids.to(th.int64))

    def get_props(self, props):
        props = props.unsqueeze(1)
        # print(f"进入函数 get_props, 接收参数为 props: {props}, {props.shape}")
        return self.prop_nn(props)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight**2).sum(-1).view(-1, 1)
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb**2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = (
                emb_norm
                + arr_norm.transpose(0, 1)
                - 2.0 * th.mm(self.lm_head.weight, text_emb_t)
            )
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(
                emb_norm.size(0), hidden_repr.size(0), hidden_repr.size(1)
            )
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # timestep_embedding 返回形状为 [N x dim] 的时间步嵌入张量，N 是 timesteps 的长度，dim 就是 hidden_t_dim
        # time_embed 依赖于参数 hidden_t_dim
        # hidden_t_dim 参数根源来自于参数文件 config.json，默认为 128
        # 将生成的时间步嵌入传递给 self.time_embed，经过前面定义的线性变换和激活函数处理后，输出一个新的嵌入张量
        # 总之，timestep_embedding 函数生成的时间步嵌入形状为 [N, hidden_t_dim]
        # self.time_embed 将此嵌入转换为形状为 [N, config.hidden_size] 的新嵌入
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        # self.hidden_size 是一个来自于 ./datasets/model.json 的超参数，默认为 768
        # input_dims=hidden_dim 根源来自于参数文件，默认为 128
        if self.input_dims != self.hidden_size:
            # input_up_proj 能够确保输入维度与模型的隐藏层大小一致，通过一个线性变换和激活函数序列来实现
            # 假设输入张量的形状为 [N, input_dims]，其中 N 是批量大小，
            # 输出形状为 [N, config.hidden_size]
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x
        # 获取输入序列的长度 seq_length，即 x 的第二维大小
        seq_length = x.size(1)
        # self.position_ids 是一个形状为 (1, 512) 的张量，包含从 0 到 511 的位置 ID
        # 截取前 seq_length 个元素作为 position_ids 位置编码，生成形状为 (1, seq_length) 的张量
        position_ids = self.position_ids[:, :seq_length]
        # 使用 self.position_embeddings 获取位置嵌入，形状为 [1, seq_length, config.hidden_size]
        # emb_t.unsqueeze(1) 将 emb_t 的形状转换为 [N, 1, config.hidden_size]，然后通过 expand 扩展为 [N, seq_length, config.hidden_size]
        # 将位置嵌入、输入嵌入 emb_x 和时间步嵌入 emb_t 相加，形成最终的嵌入
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        )
        # 对 emb_inputs 应用层归一化，确保每个样本的特征均值为 0，标准差为 1
        # 接着应用 Dropout，随机丢弃一定比例的神经元，以减少过拟合
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        # 将经过嵌入处理的输入 emb_inputs 传递给 BERT 编码器，从而生成输入的隐藏状态
        input_trans_hidden_states = self.input_transformers(
            emb_inputs
        ).last_hidden_state

        if self.output_dims != self.hidden_size:
            # 用于将模型的隐藏状态从 config.hidden_size（768）转换为所需的输出维度 self.output_dims（128）
            # h 的形状将为 [N, seq_length, self.output_dims]
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        # 将 h 的数据类型转换为与输入 x 的数据类型相同
        h = h.type(x.dtype)
        return h
