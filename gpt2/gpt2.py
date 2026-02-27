# -*- coding: utf-8 -*-
import time
import math
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset
from torch.optim import AdamW
import numpy as np


# 自动选择最优设备
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
print(f'current device is `{device}`')

# config
d_model = 768
d_vocab = 50257
d_context = 2048
head_cnt = 12
d_head = d_model//head_cnt

masked = torch.triu(torch.full((d_context,d_context), float('-inf')), diagonal=1).to(device) #(T,T)

class Attention(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_attn = nn.Linear(d_model, 3*d_model) # Q,K,V
    self.c_proj = nn.Linear(d_model, d_model) # W_o
    self.Drop = nn.Dropout(0.05)

  def forward(self, x, padding_mask=None):
    B,T,C = x.shape # T maybe != d_context
    qkv = self.c_attn(x) # (B,T,3*C)
    q,k,v = torch.split(qkv, qkv.size(-1)//3, dim=-1) #(B,T,C) * 3
    q = q.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()    #(B,H,T,c) 这里 c=C/H
    k = k.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()
    v = v.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()
    #scores = torch.matmul(q,k.transpose(-1,-2))/(d_head ** 0.5) + masked[:T, :T]
    #scores = F.softmax(scores, dim=-1) #(B,H,T,T)
    #scores = self.Drop(scores)
    #attn = torch.matmul(scores, v).transpose(1,2) #(B,H,T,c) -> (B,T,H,c)
    attn_masks = None
    if padding_mask is not None:
      # ① 因果掩码：(1, 1, T, T)，上三角为 -inf
      causal = masked[:T, :T]
      causal = torch.triu(
        torch.full((T, T), float('-inf'), device=x.device, dtype=torch.float32),
        diagonal=1
      ).unsqueeze(0).unsqueeze(0)

      # ② padding 掩码：(B, 1, 1, T)，padding 位置为 -inf，作用在 key 维度
      #    padding_mask: 1=真实 token, 0=padding
      key_pad = torch.where(
        padding_mask.bool().unsqueeze(1).unsqueeze(2),          # (B, 1, 1, T)
        torch.zeros(1, device=x.device, dtype=torch.float32),   # 真实 token → 0
        torch.full((1,), float('-inf'), device=x.device, dtype=torch.float32)  # padding → -inf
      )
      # ③ 合并两个掩码，并对齐 dtype（autocast 下 q 可能是 bfloat16）
      attn_masks = (causal + key_pad).to(dtype=q.dtype)  # (B, 1, T, T)

    attn = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_masks,
        dropout_p=0.05 if self.training else 0.0,
        is_causal=(attn_masks is None)
    )
    attn = attn.contiguous().reshape(B,T,C) # (B,T,C)
    return self.c_proj(attn) # (B,T,C)

class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.ln_1 = nn.LayerNorm(d_model)
    self.attn = Attention()
    self.ln_2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(OrderedDict({
        "c_fc": nn.Linear(d_model, 4*d_model),
        "gelu": nn.GELU(approximate='tanh'),
        "c_proj": nn.Linear(4*d_model, d_model),
        "dropout": nn.Dropout(0.1)
    }))

  def forward(self, x, padding_mask=None):
    residual = x
    x = self.ln_1(x) # layer norm
    x = self.attn(x, padding_mask=padding_mask) # attention
    x = x+residual
    residual = x
    x = self.ln_2(x) # layer norm
    x = self.mlp(x)  # mlp
    x = x+residual
    return x

class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.wte = nn.Embedding(d_vocab, d_model)
    self.wpe = nn.Embedding(d_context, d_model)
    self.Drop = nn.Dropout(0.1)
    self.h = nn.ModuleList([
        Block() for _ in range (12)
    ])
    self.ln_f = nn.LayerNorm(d_model)

  def forward(self, x, padding_mask=None):
    B,T = x.shape
    if T > d_context:
        raise ValueError(f"输入序列长度 T={T} 超出了最大上下文长度 d_context={d_context}")
    x = self.wte(x)+self.wpe(torch.arange(0, T, device=device))
    x = self.Drop(x)
    for block in self.h:
      x = block(x, padding_mask=padding_mask)
    x = self.ln_f(x)
    return x

class GPT2(nn.Module):
  def __init__(self):
    super().__init__()
    self.transformer = Transformer()
    self.lm_head = nn.Linear(d_model, d_vocab, bias=False)
    self.lm_head.weight = self.transformer.wte.weight
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, x, padding_mask=None):
    x = self.transformer(x, padding_mask=padding_mask) # (B,T,C)
    x = self.lm_head(x) # (B,T,V)
    return x

class FineWebDataset(IterableDataset):
  def __init__(self, hf_dataset, tokenizer, max_length=d_context):
    self.dataset = hf_dataset
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __iter__(self):
    buffer = []
    for sample in self.dataset:
      ids = self.tokenizer.encode(sample["text"])
      ids.append(self.tokenizer.eos_token_id)
      buffer.extend(ids)
      while len(buffer) >= self.max_length+1:
        chunk = buffer[:self.max_length+1]
        buffer = buffer[self.max_length+1:]
        yield {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long)
        }

batch_size = 32 # maximum batch size accordding to HBM
accumulation_steps = (2**19)//d_context//batch_size
print(f'accumulation_steps={accumulation_steps}')
epoch_nums = 10
warmup_steps = 10000
max_steps =    1000000

def get_lr(step, max_lr=6e-4):
  min_lr = max_lr*0.1
  if step <= warmup_steps:
    return (step+1)/warmup_steps*max_lr
  elif step >= max_steps:
    return min_lr

  ratio = (step - warmup_steps) / (max_steps - warmup_steps)
  ratio = 0.5*(math.cos(ratio*math.pi)+1)
  return min_lr + ratio*(max_lr-min_lr)

# Data
# use name="sample-10BT" to use the 10BT sample
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
finewebRawData = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
dataset = FineWebDataset(finewebRawData, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size)


gpt2 = GPT2()
gpt2 = gpt2.to(device)
gpt2.compile()
#dataloader = DataLoaderInLocalFile('input.txt', batch_size=batch_size)
optimizer = AdamW(gpt2.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)
optimizer.zero_grad(set_to_none=True)

gpt2.train()
trained = 0
step = 0
global_step = 0
t1 = time.time()

accu_loss = 0
monitor_steps = 200
for epoch in range(epoch_nums):
  for batch in dataloader:
    x, y = batch["input_ids"], batch["labels"]
    x = x.to(device, dtype=torch.long) # 确保是整数
    y = y.to(device, dtype=torch.long) # 确保是整数

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      logits = gpt2(x)
      #loss = F.cross_entropy(logits.permute(0,2,1), y)
      # 将 logits 从 [B, T, V] 展平成 [B*T, V]
      # 将 y 从 [B, T] 展平成 [B*T]
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
      accu_loss += loss.item()
      loss = loss/accumulation_steps
    loss.backward()

    if (step+1) % accumulation_steps == 0:
      global_step += 1
      lr = get_lr(global_step)
      for group in optimizer.param_groups:
        group['lr'] = lr   # 手动注入
      torch.nn.utils.clip_grad_norm_(gpt2.parameters(), max_norm=1.0)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)

    step += 1
    trained += batch_size*d_context
    if step % monitor_steps == 0:
      torch.cuda.synchronize()
      t2 = time.time()
      print(f'step:{global_step: 6d} | tokens:{trained: 12.3e} | loss:{accu_loss/(monitor_steps):06.3f} | lr: {lr: .6f} | speed: {batch_size*monitor_steps*d_context / (t2-t1):6.0f} tokens/sec')
      accu_loss = 0
      t1 = time.time()

def generate(text, my_model=None, max_tokens=d_context, temperature=0.7, top_k=40):
  assert my_model is not None, "my_model is None"
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
  my_model.eval()
  with torch.no_grad():
    for _ in range(max_tokens):
      if input_ids.shape[1] >= d_context:
        print(f'BREAK: require too much tokens')
        break
      logits = my_model(input_ids)
      logits = logits[:,-1,:]
      logits = logits/temperature
      if top_k > 0:
        # 找到第 k 大的那个值
        v, _ = torch.topk(logits, top_k)
        min_val = v[:, -1].unsqueeze(1)
        # 将小于第 k 大值的 logits 设为负无穷，这样 softmax 后概率为 0
        logits = torch.where(
            logits < min_val,
            torch.tensor(float('-inf')).to(device),
            logits
        )
      probs = F.softmax(logits, dim=-1)
      newidx = torch.multinomial(probs, num_samples=1)
      input_ids = torch.cat((input_ids, newidx), dim=1)
    print(f'{tokenizer.decode(input_ids[0].tolist())}')

generate("I am your father. Where are you?")