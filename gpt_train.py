import inspect
from dataclasses import dataclass
from datetime import datetime
from itertools import repeat
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

from sympy import false
from torch.backends.mkl import verbose
from torch.nn import functional as F

#auto detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using device: {device}")


@dataclass
class GPTConfig:
    """Hyper Parameters for GPT"""
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #number of tockens 50,000 BPE merges + 256 byte tockens+(end of text)
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12


class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_layer == 0
        # key query value projection for all heads,but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        ## bias/mask following OpenAI/HF naming
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))
        self.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()  # batch size,sequence length,embedding  dimensionality(n_embd)

        # calculate query ,key,value for all heads in batch and move head forward to be the batch
        # nh in "number of heads",hs is "head size" an C(number of channels) = nh*hs
        # e.g. in GPT-2 (124),n_head = 12,hs = 64,so nh*hs=C=768 channels in Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,nh,T,hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B,
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,
                                                                  2)  # (B,nh,T,hs)            #attention (materialize the large(T,T)matrix for all the queries and keys)
        #uncomment the below four lines if you want do not want to use flash attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B,nh,T,T)x(B,nh,T,hs)
        # the line below implements flash attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # can use approximate = none also
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # residual network
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # weights of tocken embedding
            wpe=nn.Embedding(config.block_size, config.n_embd),  # weights of position embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing  scheme
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        loss = None
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizer(self,weight_decay,learning_rate,device):
        #start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        print(f"num decayed parameter tensors: {num_decay_params},with {num_decay_params:,}parameters")
        print(f"num non decayed parameter tensors: {num_nodecay_params},with {num_nodecay_params:,}parameters")
        #create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused adamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_group, lr=learning_rate, weight_decay=weight_decay,betas=(0.9, 0.98), eps=1e-8)
        return optimizer


    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer,n_headand n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-meadium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        # discard this mask

        # init a huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameter are aligned and match in name and shape
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model





from transformers import GPT2Tokenizer
enc = GPT2Tokenizer.from_pretrained('gpt2')

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = GPT2Tokenizer.from_pretrained('gpt2')
        tockens = enc.encode(text)
        self.tockens = torch.tensor(tockens)
        print(f"loaded{len(self.tockens)} tokens")
        print(f'1 epoch = {len(self.tockens) // (B * T)} batches')

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tockens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # imputs
        y = buf[1:].view(B, T)  # targets
        x = x.to('cuda')
        y = y.to('cuda')
        self.current_position += B * T
        if self.current_position + (B * T + 1) >= len(self.tockens):
            self.current_position = 0
        return x, y

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288#2**19,~0.5M in number of tockens
B= 16
T = 1024
assert total_batch_size % (B * T) == 0,"make sure total_batch_sixze if divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size:{total_batch_size}")
print(f'=> calculated gradient accumulation steps:{grad_accum_steps}')



torch.set_float32_matmul_precision('high')
train_loader = DataLoaderLite(B = 4,T = 1024)
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
#model = torch.compile(model)
#only used for Linux
max_steps = 50
max_lr = 6e-4
min_lr = max_lr *0.1
warmup_steps = 10

def get_lr(it):
    if it < max_steps:
        return max_lr * (it+1)/warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps)/(max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(),lr = 6e-4,betas=(0.9,0.95),eps=1e-8)
optimizer = model.configure_optimizer(weight_decay = 0.1,learning_rate=6e-4,device=device)

for step in range(max_steps):
    t0 = datetime.now()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device,dtype=torch.float16):
            logits,loss = model(x,y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm =  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = datetime.now()
    dt = (t1 - t0).total_seconds() * 1000
    print(f"step {step},loss {loss_accum.item():.6f},norm:{norm:.4f},dt:{dt:.2f}ms,learning_rate:{lr:.6f}")
