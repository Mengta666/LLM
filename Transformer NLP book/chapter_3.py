import torch
from torch import nn
import torch.nn.functional as F

# 注意力机制实现
def scaled_dot_product_aatention(query, key, value, mask=None):
    dim_k = key.size(-1)
    # transpose函数用来交换维度（也就是转置），（1，2）表示torch.Size([1, 5, 768])中的5，768
    scores = torch.bmm(query, key.transpose(1,2)) / dim_k**0.5
    # 对于解码器设置掩码，上三角的部分填充为无穷小（softmax下这部分为0）
    if mask is not None:
        scores.masked_fill(mask==0, float("-inf"))
    wrights = F.softmax(scores, dim=-1)
    return torch.bmm(wrights, value)

# 注意力头
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """
        :param embed_dim: 总的嵌入向量维数（列）
        :param head_dim: 注意力头的维数（单头的嵌入向量维数）
        """
        super().__init__()
        # nn.Linear(embed_dim, head_dim) 初始化了一个权重矩阵（embed_dim * head_dim）和偏置
        # self.q 是一个 nn.Linear 层，用于生成查询向量（query）。
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_states, mask=None):
        """
        :param hidden_states: 输入的 hidden_states 矩阵，形状为 [batch_size, seq_len, embed_dim]
        """
        # self.q(hidden_states) 将 hidden_states 通过线性变换生成查询向量，形状变为 [batch_size, seq_length, head_dim]。
        # 线性变换公式：output = hidden_states · W_q + b_q
        # 在 PyTorch 中，self.q（nn.Linear 层）中的权重矩阵 W_q 和偏置 b_q 会自动参与训练，并通过反向传播优化以获得最优权重。
        x = scaled_dot_product_aatention(self.q(hidden_states), self.k(hidden_states), self.v(hidden_states), mask)
        return x

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([ AttentionHead(embed_dim, head_dim) for _ in range(num_heads) ])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states, mask=None):
        x = torch.cat( [attention_head(hidden_states, mask) for attention_head in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

# 前馈神经网络（全连接）
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

# 层规范化
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    def forward(self, x , mask=None):
        x = self.layer_norm_1(x)
        x = self.attention(x , mask)
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        return x

# 位置嵌入
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids):
        token_embeddings = self.token_embeddings(input_ids)
        seq_length = input_ids.size(1)
        # 为序列中的每个 token 分配一个唯一的整数索引（例如 [0, 1, 2, 3, 4]），表示其绝对位置。
        # 也就是说用position_id来表示token的位置，这个位置和嵌入向量有关系
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = token_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# 合并前面全部操作（编码器）
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList( [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        for layer in self.layers:
            embeddings = layer(embeddings)
        return embeddings

# 解码器
class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embeddings(config)
        self.layers = nn.ModuleList( [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, mask=None):
        embeddings = self.embedding(input_ids)
        for layer in self.layers:
            embeddings = layer(embeddings, mask)
        return embeddings


if __name__ == '__main__':

    # 一个简易的注意力机制实现
    # 分词
    from transformers import AutoTokenizer
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    input_ids = tokenizer("time files like an arrow", return_tensors="pt", add_special_tokens=False).input_ids

    # 注意力机制实现
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(checkpoint)
    # 创建一个词汇表嵌入矩阵（随机初始化）
    token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
    # 通过词汇表嵌入矩阵得到一个输入序列嵌入矩阵，用于训练
    inputs_embeds = token_emb(input_ids)
    print(inputs_embeds.shape)                        # torch.Size([1, 5, 768])

    # 这里为了简单，这里将Q，K，V都设为输入序列嵌入矩阵
    query = key = value = inputs_embeds
    dim_k = key.size(-1)
    # 批量矩阵乘法函数bmm，利用矩阵乘法来模拟点积，得到一个权重矩阵，/dim_k**0.5为了使得点积过后的值过大，进行缩小
    scores = torch.bmm(query, key.transpose(1,2)) / dim_k**0.5
    # 利用softmax函数进行归一化
    import torch.nn.functional as F
    weights = F.softmax(scores, dim = -1)
    output = torch.bmm(weights, value)
    # torch.Size([1, 5, 768])
    print(output.shape)

    #  使用类的构建方式
    print("编码器构建方式:")
    encoder = TransformerEncoder(config)
    output = encoder(input_ids)
    # torch.Size([1, 5, 768])
    print(output.shape)

    # 对于解码器
    # 设置掩码矩阵
    print("解码器构建方式:")
    seq_length = input_ids.size(1)
    # unsqueeze(0) 在第 0 维增加一个维度（batch 维度）
    mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0)
    decoder = TransformerDecoder(config)
    output = decoder(input_ids, mask)
    # torch.Size([1, 5, 768])
    print(output.shape)


