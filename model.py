import torch
import torch.nn as nn
import math


# 对 q和 k应用旋转位置编码RoPE
def apply_rope(q, k, seq_len):
    # 定义旋转函数
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    '''
    生成旋转位置编码的角度，position 的形状是 [seq_len, 1]
    freqs 的形状是 [dim/2]（假设 q.shape[-1] 是特征维度 dim）
    当执行 position * freqs 时，PyTorch 会自动进行广播
    最终结果 freqs 的形状是 [seq_len, dim/2]，与外积的结果完全一致
    '''
    position = torch.arange(seq_len, dtype=torch.float32, device=q.device).unsqueeze(1)
    freqs = 1.0 / (10000 ** (torch.arange(0, q.shape[-1], 2, dtype=torch.float32, device=q.device) / q.shape[-1]))
    freqs = position * freqs
    # 生成正弦和余弦值
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(1)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(1)
    # 重复 cos 和 sin 以匹配 q 和 k 的维度
    sin = torch.cat([sin, sin], dim=-1)
    cos = torch.cat([cos, cos], dim=-1)
    # 应用旋转位置编码
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

# LayerNorm调整的是均值和标准差，RMSNorm调整的是均方根
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps  # 设置 epsilon，防止除零错误
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数

    def _norm(self, x):
        #开平方后，倒数。rsqrt(x) = 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight  # 应用 RMSNorm,乘以权重参数


class MultiAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, \
            "Embedding dimension must be divisible by number of heads"

        # 线性投影层
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # softmax层
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        """
        参数: x输入张量，形状为 [batch_size, seq_len, embed_dim]
             mask可选的注意力掩码，形状为 [batch_size, seq_len, seq_len]
        返回:输出张量，形状为 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        # 线性投影
        q, k, v = self.q(x), self.k(x), self.v(x)

        # 分割头，将x形状为[batch_size, seq_len, embed_dim]分割为多个头
        # 分割后的张量，形状为[batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 应用旋转位置编码
        q, k = apply_rope(q, k, seq_len)

        # 缩放点积注意力计算
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores = scores + torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # 应用掩码（如果提供）
        if mask is not None:
            # 扩展原始掩码维度以匹配注意力分数的形状
            # 从 (batch_size, seq_len) 扩展为 (batch_size, 1, 1, seq_len)
            attention_mask = mask.unsqueeze(1).unsqueeze(2)
            # 将需要屏蔽的位置（值为0的位置）设为负无穷
            attention_mask = (1.0 - attention_mask) * -1e9
            scores = scores + attention_mask

        # 计算注意力权重
        scores = self.softmax(scores)
        # 计算上下文向量
        context = scores @ v
        # 重新排列维度并合并多头结果
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, -1)
        # 输出投影层
        context = self.out_proj(context)
        return context



class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_size, bias=False)
        self.gate_proj = nn.Linear(embed_dim, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, embed_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.silu(self.up_proj(x)) * self.gate_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, hidden_size):
        super().__init__()
        self.attention = MultiAttention(embed_dim, num_heads)
        self.attention_layernorm = RMSNorm(embed_dim, eps=1e-6)
        self.feed_forward = FeedForward(embed_dim, hidden_size)
        self.feed_forward_layernorm = RMSNorm(embed_dim, eps=1e-6)

    def forward(self, x, mask=None):
        residue = x
        x = self.attention(x, mask)
        x = self.attention_layernorm(x + residue)
        residue = x
        x = self.feed_forward(x)
        x = self.feed_forward_layernorm(x + residue)
        return x


class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super().__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4*embed_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        # 词嵌入层
        x = self.embed_layer(x)
        # N层 Transformer模块
        for block in self.transformer_blocks:
            x = block(x, mask)
        # 输出投影层
        x = self.output_proj(x)
        return x

    def generate(self, x, max_length=512, temperature=0.7, top_k=50):
        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(x)  # [B, L, V]
                next_token_logits = logits[:, -1, :] / temperature  # [B, V]

                if top_k > 0:
                    # 取 top_k 及其下标
                    top_v, top_i = torch.topk(next_token_logits, top_k, dim=-1)  # [B, top_k]
                    # 构建 mask：仅保留 top_k，其余置 -inf
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits = mask.scatter(-1, top_i, top_v)

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)

                if next_token[0, 0] == 2:  # 结束符
                    break
        return x

if __name__ == "__main__":
   pass