import torch
import torch.nn.functional as F
from torch import nn


# 缩放点积注意力函数
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    实现缩放点积注意力机制
    :param query: 查询张量，形状 [batch_size, seq_length, head_dim]
    :param key: 键张量，形状 [batch_size, seq_length, head_dim]
    :param value: 值张量，形状 [batch_size, seq_length, head_dim]
    :param mask: 注意力掩码，形状 [batch_size, seq_length, seq_length]，0 表示掩盖，1 表示可见
    :return: 注意力输出，形状 [batch_size, seq_length, head_dim]
    """
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / dim_k ** 0.5  # 计算注意力分数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # 掩盖无效位置
    weights = F.softmax(scores, dim=-1)  # 归一化得到注意力权重
    return torch.bmm(weights, value)  # 加权求和


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        单头注意力机制
        :param embed_dim: 输入嵌入维度
        :param head_dim: 单个注意力头的输出维度
        """
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)  # 查询线性变换
        self.k = nn.Linear(embed_dim, head_dim)  # 键线性变换
        self.v = nn.Linear(embed_dim, head_dim)  # 值线性变换

    def forward(self, hidden_states=None, encoder_output=None, mask=None):
        """
        前向传播
        :param hidden_states: 输入隐藏状态，形状 [batch_size, seq_length, embed_dim]
        :param encoder_output: 编码器输出（用于交叉注意力），形状 [batch_size, seq_length, embed_dim]
        :param mask: 注意力掩码，形状 [batch_size, seq_length, seq_length]
        :return: 注意力输出，形状 [batch_size, seq_length, head_dim]
        """
        if hidden_states is None:
            raise ValueError("hidden_states 不能为 None")
        if encoder_output is not None:
            # 交叉注意力：查询来自 hidden_states，键和值来自 encoder_output
            x = scaled_dot_product_attention(self.q(hidden_states), self.k(encoder_output), self.v(encoder_output))
        else:
            # 自注意力：查询、键、值均来自 hidden_states
            x = scaled_dot_product_attention(self.q(hidden_states), self.k(hidden_states), self.v(hidden_states), mask)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        """
        多头注意力机制
        :param config: 模型配置，包含 num_attention_heads 和 hidden_size
        """
        super().__init__()
        num_heads = config.num_attention_heads
        embed_dim = config.hidden_size
        head_dim = embed_dim // num_heads  # 每个头的维度
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_heads)])  # 多头列表
        self.output_linear = nn.Linear(embed_dim, embed_dim)  # 输出线性层

    def forward(self, hidden_states=None, encoder_output=None, mask=None):
        """
        前向传播
        :param hidden_states: 输入隐藏状态，形状 [batch_size, seq_length, embed_dim]
        :param encoder_output: 编码器输出（用于交叉注意力），形状 [batch_size, seq_length, embed_dim]
        :param mask: 注意力掩码，形状 [batch_size, seq_length, seq_length]
        :return: 多头注意力输出，形状 [batch_size, seq_length, embed_dim]
        """
        if hidden_states is None:
            raise ValueError("hidden_states 不能为 None")
        if encoder_output is None:
            # 自注意力
            x = torch.cat([head(hidden_states=hidden_states, mask=mask) for head in self.heads], dim=-1)
        else:
            # 交叉注意力
            x = torch.cat([head(hidden_states=hidden_states, encoder_output=encoder_output) for head in self.heads],
                          dim=-1)
        return self.output_linear(x)  # 合并多头输出并线性变换


class FeedForward(nn.Module):
    def __init__(self, config):
        """
        前馈网络
        :param config: 模型配置，包含 hidden_size, intermediate_size
        """
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一层线性变换
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二层线性变换
        self.gelu = nn.GELU()  # GELU 激活函数
        self.dropout = nn.Dropout()  # Dropout 层（默认概率 0.1）

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 [batch_size, seq_length, hidden_size]
        :return: 前馈网络输出，形状 [batch_size, seq_length, hidden_size]
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncAndDecLayer(nn.Module):
    def __init__(self, config):
        """
        编码器-解码器 Transformer 层，包含自注意力、交叉注意力和前馈网络
        :param config: 模型配置
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)  # 层归一化
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_3 = nn.LayerNorm(config.hidden_size)
        self.self_attention = MultiHeadAttention(config)  # 自注意力
        self.cross_attention = MultiHeadAttention(config)  # 交叉注意力
        self.feed_forward = FeedForward(config)  # 前馈网络

    def forward(self, input_ids, encoder_output, mask=None):
        """
        前向传播
        :param input_ids: 解码器输入（嵌入表示），形状 [batch_size, seq_length, hidden_size]
        :param encoder_output: 编码器输出，形状 [batch_size, seq_length, hidden_size]
        :param mask: 注意力掩码，形状 [batch_size, seq_length, seq_length]
        :return: 层输出，形状 [batch_size, seq_length, hidden_size]
        """
        # 自注意力 + 残差连接
        x = self.layer_norm_1(input_ids)
        self_attention_output = self.self_attention(hidden_states=x, mask=mask)
        x = x + self_attention_output

        # 交叉注意力 + 残差连接
        x = self.layer_norm_2(x)
        cross_attention_output = self.cross_attention(hidden_states=x, encoder_output=encoder_output)
        x = x + cross_attention_output

        # 前馈网络 + 残差连接
        x = self.layer_norm_3(x)
        feed_forward_output = self.feed_forward(x)
        x = x + feed_forward_output
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        """
        编码器 Transformer 层，包含自注意力和前馈网络
        :param config: 模型配置
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 [batch_size, seq_length, hidden_size]
        :return: 层输出，形状 [batch_size, seq_length, hidden_size]
        """
        x_norm = self.layer_norm_1(x)
        x = x + self.self_attention(x_norm)
        x_norm = self.layer_norm_2(x)
        x = x + self.feed_forward(x_norm)
        return x


class Embeddings(nn.Module):
    def __init__(self, config):
        """
        嵌入层，将 token 和位置信息转换为嵌入向量
        :param config: 模型配置，包含 vocab_size, hidden_size, max_position_embeddings, layer_norm_eps
        """
        super().__init__()
        self.dropout = nn.Dropout()  # Dropout 层（默认概率 0.1）
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids):
        """
        前向传播
        :param input_ids: token 索引，形状 [batch_size, seq_length]
        :return: 嵌入表示，形状 [batch_size, seq_length, hidden_size]
        """
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.token_embeddings(input_ids) + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        """
        Transformer 编码器，包含嵌入层和多层编码器层
        :param config: 模型配置
        """
        super().__init__()
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids):
        """
        前向传播
        :param input_ids: 输入 token 索引，形状 [batch_size, seq_length]
        :return: 编码器输出，形状 [batch_size, seq_length, hidden_size]
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, config):
        """
        Transformer 解码器，包含嵌入层和多层解码器层
        :param config: 模型配置
        """
        super().__init__()
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncAndDecLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, encoder_output, mask=None):
        """
        前向传播
        :param input_ids: 解码器输入 token 索引，形状 [batch_size, seq_length]
        :param encoder_output: 编码器输出，形状 [batch_size, seq_length, hidden_size]
        :param mask: 注意力掩码，形状 [batch_size, seq_length, seq_length]
        :return: 解码器输出，形状 [batch_size, seq_length, hidden_size]
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        return x


class TransformerEncAndDec(nn.Module):
    def __init__(self, config):
        """
        Transformer 编码器-解码器模型
        :param config: 模型配置
        """
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, encoder_input_ids, decoder_input_ids, mask=None):
        """
        前向传播
        :param encoder_input_ids: 编码器输入 token 索引，形状 [batch_size, src_seq_length]
        :param decoder_input_ids: 解码器输入 token 索引，形状 [batch_size, tgt_seq_length]
        :param mask: 注意力掩码，形状 [batch_size, tgt_seq_length, tgt_seq_length]
        :return: 输出 logits，形状 [batch_size, tgt_seq_length, vocab_size]
        """
        encoder_output = self.encoder(encoder_input_ids)
        x = self.decoder(decoder_input_ids, encoder_output, mask)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    from transformers import AutoConfig, AutoTokenizer

    checkpoint = "facebook/bart-base"
    config = AutoConfig.from_pretrained(checkpoint)
    config.layer_norm_eps = 1e-12  # 设置层归一化 epsilon
    config.intermediate_size = 4 * config.hidden_size  # 设置前馈网络中间层大小
    model = TransformerEncAndDec(config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder_input_ids = tokenizer("time flies like an arrow", return_tensors="pt", add_special_tokens=False).input_ids
    decoder_input_ids = tokenizer("时间像箭一样飞逝", return_tensors="pt", add_special_tokens=False).input_ids
    seq_length = decoder_input_ids.shape[1]
    mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0)  # 创建因果掩码
    print(model(encoder_input_ids, decoder_input_ids, mask).shape)