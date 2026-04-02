# -*- coding: utf-8 -*-
import os
import time
import math
import contextlib
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset_builder, load_dataset
from torch.optim import AdamW
import torch.distributed as dist
import numpy as np
from datetime import datetime

assert torch.cuda.is_available()

current_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
logfile = open(f'./{current_time}.gpt2_train.logs', 'a', encoding='utf-8')

def logs(s: str):
  logfile.write(s+"\n")

# device selection and distributed process group
worldsize = int(os.environ.get("WORLD_SIZE", "1"))
is_distributed = worldsize > 1
if is_distributed:
  dist.init_process_group(backend='nccl')
  localrank = int(os.environ["LOCAL_RANK"])
  globalrank = int(os.environ["RANK"])
  torch.cuda.set_device(localrank)
  device = torch.device(f"cuda:{localrank}")
else:
  localrank = 0
  globalrank = 0
  device = torch.device("cuda")
print(f'is_distributed: {is_distributed} current device is {device}, world size is {worldsize}')

# model config
d_model = 768
d_vocab = 50257
d_context = 1024 # this is different from original GPT2 model
head_cnt = 12
d_head = d_model//head_cnt

masked = torch.triu(torch.full((d_context,d_context), float('-inf')), diagonal=1).to(device) #(T,T)

# model definition
class Attention(nn.Module):
  def __init__(self):
    super().__init__()
    self.c_attn = nn.Linear(d_model, 3*d_model) # Q,K,V
    self.c_proj = nn.Linear(d_model, d_model) # W_o
    self.Drop = nn.Dropout(0.05)

  def forward(self, x):
    B,T,C = x.shape # T maybe != d_context
    qkv = self.c_attn(x) # (B,T,3*C)
    q,k,v = torch.split(qkv, qkv.size(-1)//3, dim=-1) #(B,T,C) * 3
    q = q.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()    #(B,H,T,c) 这里 c=C/H
    k = k.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()
    v = v.view(B,T,head_cnt,d_head).transpose(1,2).contiguous()
    attn = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=0.05 if self.training else 0.0,
        is_causal=True
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

  def forward(self, x):
    residual = x
    x = self.ln_1(x) # layer norm
    x = self.attn(x) # attention
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

  def forward(self, x):
    B,T = x.shape
    if T > d_context:
        raise ValueError(f"输入序列长度 T={T} 超出了最大上下文长度 d_context={d_context}")
    x = self.wte(x)+self.wpe(torch.arange(0, T, device=x.device))
    x = self.Drop(x)
    for block in self.h:
      x = block(x)
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

  def forward(self, x):
    x = self.transformer(x) # (B,T,C)
    x = self.lm_head(x) # (B,T,V)
    return x

# data loader
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

def estimate_batch_size(device, seq_len=d_context, safety_factor=0.75):
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory  # bytes
    # model parameters 
    param_mem = sum(p.numel() * p.element_size() for p in gpt2.parameters())
    # AdamW 优化器状态：fp32 master weights + 2个momentum，约 3× param_mem（fp32）
    optimizer_mem = param_mem * (4 / 2) * 3  # 换算成 fp32 再乘 3
    # 每个样本激活内存（GPT-2 经验值，bf16 下约 1MB per token 再乘 layers）
    # 粗略估算：2 bytes × seq_len × d_model × 12层 × 前向+反向(×2)
    bytes_per_sample = 2 * seq_len * d_model * 12 * 2
    available = total_mem * safety_factor - param_mem - optimizer_mem
    batch = max(1, int(available // bytes_per_sample))
    # 取 2 的幂次（对 CUDA 友好）
    batch = 2 ** int(math.log2(batch))
    print(f"esitmated batch size of {device} is {batch}")
    return batch

# train config
batch_size = estimate_batch_size(device) # maximum batch size accordding to HBM
tokens_per_optim_step = 2**19
accumulation_steps = max(1, tokens_per_optim_step // (d_context * batch_size * worldsize))
print(f'accumulation_steps={accumulation_steps}')
epoch_nums = 10
max_steps = 40*(10**9) // tokens_per_optim_step
warmup_steps = max_steps*0.01

def get_lr(step, max_lr=6e-4):
  min_lr = max_lr*0.1
  if step <= warmup_steps:
    return (step+1)/warmup_steps*max_lr
  elif step >= max_steps:
    return min_lr

  ratio = (step - warmup_steps) / (max_steps - warmup_steps)
  ratio = 0.5*(math.cos(ratio*math.pi)+1)
  return min_lr + ratio*(max_lr-min_lr)

def save_checkpoint(global_step, model, optimizer):
  torch.save({
    "global_step": global_step,
    "model": (model.module if is_distributed else model).state_dict(),
    "optimizer": optimizer.state_dict(),
  }, f"checkpoint_step{global_step}.pt")

def save_finished_checkpoint(model, optimizer):
  torch.save({
    "global_step": -1,
    "model": (model.module if is_distributed else model).state_dict(),
    "optimizer": optimizer.state_dict(),
  }, f"checkpoint_finished.pt")

def load_checkpoint(filename):
  ckpt = torch.load(filename, map_location=device)
  gpt2.load_state_dict(ckpt["model"])
  optimizer.load_state_dict(ckpt["optimizer"])
  return ckpt["global_step"]

# Data
# use name="sample-10BT" to use the 10BT sample
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"tokenzier is loaded")
tokenizer.model_max_length = d_context
tokenizer.pad_token = tokenizer.eos_token
finewebRawData = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
print(f"dataset is loaded")
sharedRawData = finewebRawData.shard(num_shards=worldsize, index=globalrank)
dataset = FineWebDataset(sharedRawData, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size)

gpt2 = GPT2()
gpt2 = gpt2.to(device)
if is_distributed:
  gpt2 = DistributedDataParallel(gpt2, device_ids=[localrank])
gpt2 = torch.compile(gpt2)
print(f"gpt2 model is compiled")

optimizer = AdamW(gpt2.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)
optimizer.zero_grad(set_to_none=True)
lr = optimizer.param_groups[0]["lr"]

gpt2.train()
trained = 0
step = 0
global_step = 0
t1 = time.time()

accu_loss = 0
interval_tokens = 0
global_tokens = 0
monitor_steps = 200
save_step_interval, last_save_step = 1000, 0
data_iter = iter(dataloader)

# global_step = load_checkpoint("checkpoint_step100000.pt")

print(f"ready? train loop is starting")
while global_step < max_steps:
  try:
    batch = next(data_iter)
  except StopIteration:
    data_iter = iter(dataloader)
    batch = next(data_iter)

  x, y = batch["input_ids"], batch["labels"]
  x = x.to(device, dtype=torch.long)
  y = y.to(device, dtype=torch.long)

  is_accumulation_step = (step + 1) % accumulation_steps == 0

  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits = gpt2(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    accu_loss += loss.item()
    loss = loss / accumulation_steps

  sync_ctx = gpt2.no_sync if (is_distributed and not is_accumulation_step) else contextlib.nullcontext
  with sync_ctx():
    loss.backward()

  if is_accumulation_step:
    global_step += 1
    lr = get_lr(global_step)
    for group in optimizer.param_groups:
      group['lr'] = lr
    torch.nn.utils.clip_grad_norm_(gpt2.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

  if global_step - last_save_step > save_step_interval:
    save_checkpoint(global_step, gpt2, optimizer) 
    last_save_step = global_step

  step += 1
  batch_tokens = x.numel()
  trained += batch_tokens
  interval_tokens += batch_tokens
  if step % monitor_steps == 0:
    torch.cuda.synchronize()
    loss_t = torch.tensor(accu_loss / monitor_steps, device=device, dtype=torch.float64)
    tok_t = torch.tensor(interval_tokens, device=device, dtype=torch.float64)
    if is_distributed:
      dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
      dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
      loss_t /= worldsize
    if localrank == 0:
      t2 = time.time()
      global_tokens += int(tok_t.item())
      print(f'step:{global_step: 6d} | tokens:{global_tokens: 12.3e} | loss:{loss_t.item():06.3f} | lr: {lr: .6f} | speed: {tok_t.item() / (t2-t1):6.0f} tokens/sec')
      t1 = time.time()
    accu_loss = 0
    interval_tokens = 0

print(f"training is over, start to save model parameters")
save_finished_checkpoint(gpt2, optimizer)
print(f"saving is over, bye!")

# gracefully byebye
if is_distributed:
  dist.destroy_process_group()
