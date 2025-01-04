import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 应用mask
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 计算输出
        context = torch.matmul(attn, v)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attention_mask):
        residual, batch_size = q, q.size(0)
        
        # 线性变换并分头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # 计算注意力
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        
        # 拼接多头输出
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_pos, d_model)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos = pos.unsqueeze(0).expand_as(inputs)
        return self.pos_embedding(pos)

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

    def forward(self, inputs, attention_mask):
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        
        # 生成上三角掩码
        subsequence_mask = get_attn_subsequence_mask(inputs, self.device)
        
        if attention_mask is not None:
            attention_mask = get_attn_pad_mask(attention_mask)
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            attention_mask = subsequence_mask.bool()

        self_attns = []
        for layer in self.layers:
            outputs, self_attn = layer(outputs, attention_mask)
            self_attns.append(self_attn)
            
        return outputs, self_attns

class GPTModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(GPTModel, self).__init__()
        self.decoder = Decoder(d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device)
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, attention_mask=None):
        outputs, self_attns = self.decoder(inputs, attention_mask)
        logits = self.projection(outputs)
        return logits.view(-1, logits.size(-1)), self_attns

def get_attn_subsequence_mask(seq, device):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)

def get_attn_pad_mask(attention_mask):
    batch_size, len_seq = attention_mask.size()
    attention_mask = attention_mask.data.eq(0).unsqueeze(1)
    return attention_mask.expand(batch_size, len_seq, len_seq) 