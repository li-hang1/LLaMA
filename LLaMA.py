import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.weight

def precompute_rope(max_len, d_k):
    """
    return: [max_len, d_k/2], [max_len, d_k/2]
    """
    theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2) / d_k))
    pos = torch.arange(max_len)
    pos_theta_mul = torch.einsum("i,j->ij", pos, theta)
    return torch.cos(pos_theta_mul), torch.sin(pos_theta_mul)

def apply_rope(x, cos, sin):
    """
    x: (B, n_heads, seq_len, d_k)
    cos: [max_len, d_k/2]
    sin: [max_len, d_k/2]
    return: (B, n_heads, seq_len, d_k)
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    # 这个拼接的前后部分根据论文应该是穿插拼接，但这里整体分成了前后来拼接，因为内积是个求和，本质上是交换了求和中各项的顺序，加法交换律保证等价

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, mask):
        """
        x: [B, seq_len, d_model]
        cos: [max_len, d_k/2]
        sin: [max_len, d_k/2]
        mask: [batch_size, seq_len, seq_len]
        return: [B, seq_len, d_model]
        """
        B, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:seq_len], sin[:seq_len])
        k = apply_rope(k, cos[:seq_len], sin[:seq_len])

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = F.softmax(attn + mask.unsqueeze(1), dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.out(out)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w2(x)) * self.w1(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class LLaMA(nn.Module):
    def __init__(self, n_layers, vocab_size, d_model, num_heads, hidden_dim, max_seq_len=2056):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, hidden_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

        # 权重共享
        self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(max_seq_len, d_model // num_heads)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, input_ids, content_lens):
        """
        input_ids: [batch_size, seq_len]
        content_lens: (batch_size, )
        return: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.embed(input_ids)

        mask = torch.zeros((batch_size, seq_len, seq_len), device=x.device)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float("-inf"), diagonal=1)
        for i in range(batch_size):
            c = content_lens[i]
            mask[i, :c, c:seq_len] = float("-inf")
            mask[i, c:seq_len, c:seq_len] = causal_mask[c:seq_len, c:seq_len]

        for block in self.blocks:
            x = block(x, self.cos, self.sin, mask)

        x = self.norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    model = LLaMA(n_layers=2, vocab_size=100, d_model=512, num_heads=8, hidden_dim=512)
    dummy_input = torch.randint(0, 100, (4, 10))
    output = model(dummy_input, content_lens=(7, 5, 5, 5))
    print(output.shape)
