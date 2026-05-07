import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# -------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGOT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

# class TanhGELU(nn.Module):
#     def forward(self, input):
#         return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGOT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.n_drop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_drop: float = 0.1
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGOT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# ----------------------------------------------------------
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
enc = tiktoken.get_encoding("gpt2")
total_batch_size = 524288 # 2**19
B = 4 # micro batch size
T = 32 # sequence length
assert total_batch_size % (B * T) == 0, "make sure that the total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T) * ddp_world_size
if master_process:
    print(f"total batch size: {total_batch_size}")
    print(f"=> calculated gradient accumlation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 3e-4
min_lr = max_lr * 0.1
wramup_steps = 715
max_steps = 19073
def get_lr(it):
    # 1) linear warmup for the first warmup_steps
    if it < wramup_steps:
        return max_lr * (it + 1) / wramup_steps
    # 2) if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - wramup_steps) / (max_steps - wramup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# optimize:
# ptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

                # once in a while evaluate hellaswag
                if (step % 250 == 0 or last_step) and (not use_compile):
                    num_correct_norm = 0
                    num_total = 0
                    for i, example in enumerate(iterate_examples("val")):
                        # only process examples where i % ddp_world_size == ddp_rank
                        if i % ddp_world_size != ddp_rank:
                            continue
                        # render the example into tokens and labels
                        _, tokens, mask, label = render_example(example)
                        tokens = tokens.to(device)
                        mask = mask.to(device)
                        # get the logits
                        with torch.no_grad():
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                logits, loss = model(tokens)
                            pred_norm = get_most_likely_row(tokens, mask, logits)
                        num_total += 1
                        num_correct_norm += int(pred_norm == label)
                    # reduce the stats across all processes
                    if ddp:
                        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                        num_total = num_total.item()
                        num_correct_norm = num_correct_norm.item()
                    acc_norm = num_correct_norm / num_total
                    if master_process:
                        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                        with open(log_file, "a") as f:
                            f.write(f"{step} hella {acc_norm:.4f}\n")

                # once in a while generate from the model (except step 0, which is noise)
                if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
                    model.eval()
                    num_return_sequences = 4
                    max_length = 32
                    tokens = enc.encode("Hello, I'm a language model,")
                    tokens = torch.tensor(tokens, dtype=torch.long)
                    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                    xgen = tokens.to(device)
                    sample_rng = torch.Generator(device=device)
                    sample_rng.manual_seed(42 + ddp_rank)
                    while xgen.size(1) < max_length:
                        # forward the model to get the logits
                        with torch.no_grad():
                            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                                logits, loss = model(xgen)  # (B, T, vocab_size)
                            # take the logits at the last position
                            logits = logits[:, -1, :]  # (B, vocab_size)
                            # get the probabilities
                            probs = F.softmax(logits, dim=-1)
                            # do top-k sampling of 50 (huggingface pipeline default)
                            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                            # select a token from the top-k probabilities
                            # note: multinomial does not demand the input to sum to 1
                            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                            # gather the corresponding indices
                            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                            # append to the sequence
                            xgen = torch.cat((xgen, xcol), dim=1)
                    # print the generated text
                    for i in range(num_return_sequences):
                        tokens = xgen[i, :max_length].tolist()
                        decoded = enc.decode(tokens)
                        print(f"rank {ddp_rank} sample {i}: {decoded}")

    model.train()
    optimizer.zero_grad()
    loss_accm = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accm += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accm, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss {loss.item():.6f} | lr: {lr:.5f}| norm: {norm:.4f} | dt {dt:.2f}ms | token/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
import sys; sys.exit(0)


"""
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")

Output:

loading weights from pretrained gpt: gpt2
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 148/148 [00:00<00:00, 10784.90it/s]
didn't crash yay!

-----------------------------------------------
# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('mps')

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length: # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

loading weights from pretrained gpt: gpt2
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 148/148 [00:00<00:00, 9021.58it/s]
> Hello, I'm a language model, which means I'm familiar with it, but I'm not fluent in that. Well, with that said,
> Hello, I'm a language model, and the syntax, to make use of it, is pretty good. So why do you have that and not
> Hello, I'm a language model, I'm doing this work in Python, and then I'm writing code for Haskell.

So we can
> Hello, I'm a language model, and you're making assumptions about my use of them. I'm not a software developer, I'm not a
> Hello, I'm a language model, isn't it? Now what is this model, a set of words? A set, as a whole,

-----------------------------------------------
num_return_sequences = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to('mps')

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('mps')

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length: # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

Output:

> Hello, I'm a language model, pursuits Sounds atoms taken Achievementedited hesitate Kashmir fluentihad flattened approach stamps OEMレContribut solvent z fps tutorialsVitualensitivity
> Hello, I'm a language model,jpg brutalityAngel TE Vaticancgi Cov endors 13 showertimeRange systematic lifestyles024leck Kendall thankfully Speedirming Ampl conclusions
> Hello, I'm a language model,oled electronic millennial terr enjoyRIPT Make inductdeath instincts formats crushed south maple RAD batouting UpdatedqiPolitical unsolved referendum
> Hello, I'm a language model, Autumn SoundsTORMovie sorted arg germ stageVEL swamp maybe       idelitylessnessisSpecial 95 posesbidden TER richness ceilingsère
> Hello, I'm a language model, attending frozenari Sorcerer biscuitsesame listeningFallpass photograph dst� Sing Gilbertssongay liberation 2030 ecaccompanied Pse feral

-------------·------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)

B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B, T).to(device)
y = buf[1:].view(B, T).to(device)

 # get logits
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)
print(loss)
import sys; sys.exit(0)

Output: 

using device: mps
tensor(10.8506, device='mps:0', grad_fn=<NllLossBackward0>)

-------------·------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)

B, T = 4, 32
buf = torch.tensor(tokens[:B*T+1]).to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss {loss.item()}")
import sys; sys.exit(0)

Output:

using device: mps
step 0: loss 11.029706954956055
step 1: loss 10.545978546142578
step 2: loss 10.2586669921875
step 3: loss 9.95492935180664
step 4: loss 9.747784614562988
step 5: loss 9.622285842895508
step 6: loss 9.579493522644043
step 7: loss 9.632568359375
step 8: loss 9.739675521850586
step 9: loss 9.966226577758789
step 10: loss 10.060922622680664
step 11: loss 9.831652641296387
step 12: loss 9.222914695739746
step 13: loss 8.862298011779785
step 14: loss 8.859153747558594
step 15: loss 8.635176658630371
step 16: loss 8.414689064025879
step 17: loss 8.592874526977539
step 18: loss 8.304832458496094
step 19: loss 7.500905513763428
step 20: loss 7.587424278259277
step 21: loss 7.996984958648682
step 22: loss 7.347835063934326
step 23: loss 6.851075172424316
step 24: loss 7.698177814483643
step 25: loss 8.079285621643066
step 26: loss 7.69926643371582
step 27: loss 7.616123199462891
step 28: loss 6.749787330627441
step 29: loss 6.54010009765625
step 30: loss 6.20652961730957
step 31: loss 6.040524482727051
step 32: loss 6.509182929992676
step 33: loss 6.562403678894043
step 34: loss 6.7208404541015625
step 35: loss 6.923167705535889
step 36: loss 7.163439750671387
step 37: loss 6.706636428833008
step 38: loss 6.10413932800293
step 39: loss 5.18312931060791
step 40: loss 6.526401519775391
step 41: loss 7.101323127746582
step 42: loss 6.774909019470215
step 43: loss 5.7354536056518555
step 44: loss 5.054420471191406
step 45: loss 6.353511810302734
step 46: loss 6.423314094543457
step 47: loss 5.509213447570801
step 48: loss 6.094334602355957
step 49: loss 5.915782451629639

-------------·------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

train_loader = DataLoaderLite(B=4, T=32)

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss {loss.item()}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.864021301269531
step 1: loss 10.803430557250977
step 2: loss 10.630294799804688
step 3: loss 10.717875480651855
step 4: loss 10.683014869689941
step 5: loss 10.586713790893555
step 6: loss 10.763566970825195
step 7: loss 10.685805320739746
step 8: loss 10.616341590881348
step 9: loss 10.786633491516113
step 10: loss 10.942501068115234
step 11: loss 11.384876251220703
step 12: loss 11.456494331359863
step 13: loss 10.915283203125
step 14: loss 10.046724319458008
step 15: loss 10.026792526245117
step 16: loss 10.738702774047852
step 17: loss 10.902728080749512
step 18: loss 10.724113464355469
step 19: loss 10.49101734161377
step 20: loss 10.306425094604492
step 21: loss 10.11232852935791
step 22: loss 9.63621997833252
step 23: loss 9.576298713684082
step 24: loss 9.701951026916504
step 25: loss 9.427221298217773
step 26: loss 9.324045181274414
step 27: loss 9.534696578979492
step 28: loss 9.237090110778809
step 29: loss 9.26202392578125
step 30: loss 9.052268981933594
step 31: loss 9.098008155822754
step 32: loss 9.120502471923828
step 33: loss 9.182266235351562
step 34: loss 9.62336540222168
step 35: loss 9.702461242675781
step 36: loss 9.955219268798828
step 37: loss 11.520862579345703
step 38: loss 12.656980514526367
step 39: loss 13.009371757507324
step 40: loss 13.127694129943848
step 41: loss 13.251201629638672
step 42: loss 12.92184829711914
step 43: loss 12.607610702514648
step 44: loss 9.894437789916992
step 45: loss 8.75416088104248
step 46: loss 10.551626205444336
step 47: loss 14.082304000854492
step 48: loss 13.911428451538086
step 49: loss 14.107409477233887

 -------------·------------------------------------------------------------
# weight
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss {loss.item()}")
import sys; sys.exit(0)

Output: using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.958171844482422
step 1: loss 9.881328582763672
step 2: loss 10.094736099243164
step 3: loss 9.937455177307129
step 4: loss 9.89342212677002
step 5: loss 9.924232482910156
step 6: loss 10.593067169189453
step 7: loss 10.743429183959961
step 8: loss 10.259727478027344
step 9: loss 9.899251937866211
step 10: loss 9.92605209350586
step 11: loss 9.431317329406738
step 12: loss 9.641067504882812
step 13: loss 9.558398246765137
step 14: loss 9.58387565612793
step 15: loss 9.670307159423828
step 16: loss 9.95462417602539
step 17: loss 10.57908821105957
step 18: loss 9.358396530151367
step 19: loss 9.18147087097168
step 20: loss 9.006112098693848
step 21: loss 9.068843841552734
step 22: loss 8.474428176879883
step 23: loss 8.566959381103516
step 24: loss 8.511438369750977
step 25: loss 8.470565795898438
step 26: loss 8.51710319519043
step 27: loss 8.652408599853516
step 28: loss 8.49587631225586
step 29: loss 8.569062232971191
step 30: loss 8.449703216552734
step 31: loss 8.607945442199707
step 32: loss 8.780381202697754
step 33: loss 8.738457679748535
step 34: loss 9.068102836608887
step 35: loss 9.021249771118164
step 36: loss 9.043392181396484
step 37: loss 9.000631332397461
step 38: loss 9.086548805236816
step 39: loss 9.039274215698242
step 40: loss 9.257291793823242
step 41: loss 8.698537826538086
step 42: loss 8.585567474365234
step 43: loss 8.55419921875
step 44: loss 8.297470092773438
step 45: loss 8.523971557617188
step 46: loss 8.224993705749512
step 47: loss 8.29050064086914
step 48: loss 8.226791381835938
step 49: loss 8.209710121154785

# -------------·------------------------------------------------------------
# interaction
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    print(f"step {i}: loss {loss.item()}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> logits.dtype
torch.float32
>>> exit()

# -------------·------------------------------------------------------------
# record time
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.958171844482422, dt 5552.49ms, token/sec:  23.05
step 1: loss 9.881328582763672, dt 477.04ms, token/sec:  268.32
step 2: loss 9.107891082763672, dt 220.90ms, token/sec:  579.45
step 3: loss 9.284090042114258, dt 212.02ms, token/sec:  603.70
step 4: loss 8.887201309204102, dt 214.90ms, token/sec:  595.63
step 5: loss 8.602737426757812, dt 213.76ms, token/sec:  598.81
step 6: loss 9.011346817016602, dt 221.57ms, token/sec:  577.70
step 7: loss 8.977106094360352, dt 211.41ms, token/sec:  605.47
step 8: loss 8.275749206542969, dt 217.47ms, token/sec:  588.59
step 9: loss 8.183601379394531, dt 212.21ms, token/sec:  603.18
step 10: loss 8.497385025024414, dt 211.49ms, token/sec:  605.22
step 11: loss 7.623585224151611, dt 214.56ms, token/sec:  596.56
step 12: loss 7.968697547912598, dt 222.48ms, token/sec:  575.33
step 13: loss 7.551008224487305, dt 215.98ms, token/sec:  592.64
step 14: loss 7.676279067993164, dt 218.01ms, token/sec:  587.13
step 15: loss 7.470975399017334, dt 218.52ms, token/sec:  585.76
step 16: loss 7.50407075881958, dt 215.88ms, token/sec:  592.92
step 17: loss 8.327247619628906, dt 215.86ms, token/sec:  592.98
step 18: loss 7.2459001541137695, dt 224.71ms, token/sec:  569.63
step 19: loss 7.915576934814453, dt 227.43ms, token/sec:  562.82
step 20: loss 7.536590576171875, dt 214.00ms, token/sec:  598.12
step 21: loss 7.8413214683532715, dt 214.57ms, token/sec:  596.54
step 22: loss 6.43955659866333, dt 210.69ms, token/sec:  607.53
step 23: loss 6.901613235473633, dt 212.87ms, token/sec:  601.31
step 24: loss 6.835077285766602, dt 212.00ms, token/sec:  603.76
step 25: loss 6.64145040512085, dt 210.81ms, token/sec:  607.18
step 26: loss 6.763551712036133, dt 217.88ms, token/sec:  587.48
step 27: loss 7.606548309326172, dt 213.43ms, token/sec:  599.74
step 28: loss 7.159242153167725, dt 213.11ms, token/sec:  600.61
step 29: loss 6.92185115814209, dt 208.06ms, token/sec:  615.22
step 30: loss 7.007440567016602, dt 209.70ms, token/sec:  610.38
step 31: loss 7.206015586853027, dt 213.50ms, token/sec:  599.53
step 32: loss 7.086920261383057, dt 214.64ms, token/sec:  596.36
step 33: loss 7.006921768188477, dt 211.80ms, token/sec:  604.34
step 34: loss 7.870562553405762, dt 212.04ms, token/sec:  603.67
step 35: loss 7.745656967163086, dt 211.89ms, token/sec:  604.10
step 36: loss 7.675146579742432, dt 217.80ms, token/sec:  587.70
step 37: loss 7.685906410217285, dt 213.05ms, token/sec:  600.81
step 38: loss 8.004254341125488, dt 222.16ms, token/sec:  576.17
step 39: loss 7.543290138244629, dt 209.74ms, token/sec:  610.27
step 40: loss 7.414009094238281, dt 218.20ms, token/sec:  586.62
step 41: loss 6.982158660888672, dt 212.64ms, token/sec:  601.96
step 42: loss 7.138924598693848, dt 211.70ms, token/sec:  604.64
step 43: loss 7.136780261993408, dt 211.25ms, token/sec:  605.93
step 44: loss 7.006180763244629, dt 213.36ms, token/sec:  599.91
step 45: loss 7.17011833190918, dt 216.07ms, token/sec:  592.40
step 46: loss 6.24928617477417, dt 220.42ms, token/sec:  580.71
step 47: loss 6.376556873321533, dt 212.49ms, token/sec:  602.38
step 48: loss 6.959323883056641, dt 210.39ms, token/sec:  608.40
step 49: loss 6.842306137084961, dt 216.76ms, token/sec:  590.51

# -------------·------------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.958171844482422, dt 6449.32ms, token/sec:  19.85
step 1: loss 9.881328582763672, dt 711.77ms, token/sec:  179.83
step 2: loss 10.094736099243164, dt 223.26ms, token/sec:  573.34
step 3: loss 9.937455177307129, dt 219.69ms, token/sec:  582.65
step 4: loss 9.893423080444336, dt 223.11ms, token/sec:  573.71
step 5: loss 9.924232482910156, dt 220.11ms, token/sec:  581.54
step 6: loss 10.593067169189453, dt 218.39ms, token/sec:  586.12
step 7: loss 10.743429183959961, dt 216.80ms, token/sec:  590.40
step 8: loss 10.25972843170166, dt 232.97ms, token/sec:  549.43
step 9: loss 9.899252891540527, dt 219.93ms, token/sec:  582.00
step 10: loss 9.926053047180176, dt 224.20ms, token/sec:  570.93
step 11: loss 9.431318283081055, dt 218.66ms, token/sec:  585.39
step 12: loss 9.641067504882812, dt 216.97ms, token/sec:  589.95
step 13: loss 9.558398246765137, dt 233.88ms, token/sec:  547.29
step 14: loss 9.58387565612793, dt 242.02ms, token/sec:  528.88
step 15: loss 9.670307159423828, dt 233.70ms, token/sec:  547.72
step 16: loss 9.954623222351074, dt 235.47ms, token/sec:  543.60
step 17: loss 10.579090118408203, dt 241.74ms, token/sec:  529.48
step 18: loss 9.358396530151367, dt 244.07ms, token/sec:  524.45
step 19: loss 9.18147087097168, dt 239.60ms, token/sec:  534.23
step 20: loss 9.006113052368164, dt 250.99ms, token/sec:  509.98
step 21: loss 9.06884479522705, dt 241.79ms, token/sec:  529.38
step 22: loss 8.474428176879883, dt 234.76ms, token/sec:  545.24
step 23: loss 8.566959381103516, dt 224.22ms, token/sec:  570.86
step 24: loss 8.511438369750977, dt 228.81ms, token/sec:  559.42
step 25: loss 8.470565795898438, dt 257.52ms, token/sec:  497.05
step 26: loss 8.51710319519043, dt 237.12ms, token/sec:  539.80
step 27: loss 8.652408599853516, dt 239.35ms, token/sec:  534.78
step 28: loss 8.49587631225586, dt 240.35ms, token/sec:  532.57
step 29: loss 8.569063186645508, dt 219.57ms, token/sec:  582.96
step 30: loss 8.44970417022705, dt 230.21ms, token/sec:  556.01
step 31: loss 8.607946395874023, dt 223.56ms, token/sec:  572.54
step 32: loss 8.78038215637207, dt 222.03ms, token/sec:  576.51
step 33: loss 8.738458633422852, dt 221.23ms, token/sec:  578.59
step 34: loss 9.068103790283203, dt 220.74ms, token/sec:  579.87
step 35: loss 9.021248817443848, dt 228.46ms, token/sec:  560.28
step 36: loss 9.043392181396484, dt 228.23ms, token/sec:  560.83
step 37: loss 9.000630378723145, dt 221.09ms, token/sec:  578.96
step 38: loss 9.086548805236816, dt 223.37ms, token/sec:  573.05
step 39: loss 9.039271354675293, dt 224.83ms, token/sec:  569.32
step 40: loss 9.257287979125977, dt 230.99ms, token/sec:  554.13
step 41: loss 8.698533058166504, dt 231.86ms, token/sec:  552.07
step 42: loss 8.585566520690918, dt 219.05ms, token/sec:  584.35
step 43: loss 8.554197311401367, dt 220.08ms, token/sec:  581.61
step 44: loss 8.297456741333008, dt 231.68ms, token/sec:  552.48
step 45: loss 8.523958206176758, dt 234.47ms, token/sec:  545.91
step 46: loss 8.225024223327637, dt 219.03ms, token/sec:  584.40
step 47: loss 8.290549278259277, dt 220.80ms, token/sec:  579.72
step 48: loss 8.226852416992188, dt 228.49ms, token/sec:  560.19
step 49: loss 8.209890365600586, dt 222.45ms, token/sec:  575.40

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
        import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
Python 3.11.5 (main, Sep 11 2023, 08:31:25) [Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> logits.dtype
torch.float16
>>> model.transformer.wte
Embedding(50257, 768)
>>> model.transformer.wte.weight
Parameter containing:
tensor([[ 9.9743e-05,  1.6065e-03,  1.6118e-02,  ..., -2.3506e-02,
         -9.5092e-03,  8.6977e-04],
        [ 5.6103e-03, -8.9315e-04,  3.3858e-02,  ...,  1.8497e-02,
         -1.2024e-02,  4.2558e-03],
        [ 1.2853e-02,  8.0832e-03,  1.8367e-02,  ..., -2.2407e-02,
         -1.2174e-02, -1.2083e-02],
        ...,
        [ 6.8869e-03,  1.8946e-02,  2.7229e-02,  ..., -9.2498e-03,
         -1.6403e-02,  1.1806e-02],
        [-6.4153e-03,  6.4614e-03, -1.8471e-02,  ...,  3.3779e-04,
          8.5628e-03, -4.6225e-03],
        [ 4.5271e-03, -2.1883e-02,  2.6784e-02,  ..., -4.7267e-03,
         -1.2253e-02,  2.1918e-02]], device='mps:0', requires_grad=True)
>>> model.transformer.wte.weight.dtype
torch.float32
>>> exit()

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig())
model.to(device)
# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.9581298828125, dt 4637.60ms, token/sec:  27.60
step 1: loss 9.794281005859375, dt 541.48ms, token/sec:  236.39
step 2: loss 9.070457458496094, dt 233.79ms, token/sec:  547.49
step 3: loss 9.176834106445312, dt 237.00ms, token/sec:  540.09
step 4: loss 8.82110595703125, dt 238.03ms, token/sec:  537.75
step 5: loss 8.422988891601562, dt 239.73ms, token/sec:  533.94
step 6: loss 8.929458618164062, dt 237.72ms, token/sec:  538.45
step 7: loss 8.890380859375, dt 237.89ms, token/sec:  538.07
step 8: loss 8.128128051757812, dt 238.10ms, token/sec:  537.60
step 9: loss 8.044921875, dt 234.62ms, token/sec:  545.55
step 10: loss 8.405509948730469, dt 230.83ms, token/sec:  554.53
step 11: loss 7.484733581542969, dt 227.62ms, token/sec:  562.35
step 12: loss 7.822052001953125, dt 228.18ms, token/sec:  560.95
step 13: loss 7.4376220703125, dt 225.18ms, token/sec:  568.44
step 14: loss 7.539520263671875, dt 237.91ms, token/sec:  538.01
step 15: loss 7.4039154052734375, dt 228.43ms, token/sec:  560.35
step 16: loss 7.482025146484375, dt 231.53ms, token/sec:  552.86
step 17: loss 8.287704467773438, dt 244.08ms, token/sec:  524.43
step 18: loss 7.1959686279296875, dt 244.40ms, token/sec:  523.73
step 19: loss 7.879180908203125, dt 238.15ms, token/sec:  537.48
step 20: loss 7.483558654785156, dt 237.42ms, token/sec:  539.12
step 21: loss 7.82196044921875, dt 234.04ms, token/sec:  546.92
step 22: loss 6.4590301513671875, dt 236.50ms, token/sec:  541.24
step 23: loss 6.885475158691406, dt 235.28ms, token/sec:  544.04
step 24: loss 6.8293304443359375, dt 236.87ms, token/sec:  540.38
step 25: loss 6.7048492431640625, dt 240.23ms, token/sec:  532.82
step 26: loss 6.8207244873046875, dt 238.42ms, token/sec:  536.87
step 27: loss 7.6143798828125, dt 235.29ms, token/sec:  544.00
step 28: loss 7.193733215332031, dt 223.76ms, token/sec:  572.03
step 29: loss 6.95855712890625, dt 229.71ms, token/sec:  557.23
step 30: loss 6.9754638671875, dt 238.58ms, token/sec:  536.51
step 31: loss 7.216766357421875, dt 241.62ms, token/sec:  529.75
step 32: loss 7.13336181640625, dt 252.43ms, token/sec:  507.07
step 33: loss 7.0078277587890625, dt 253.09ms, token/sec:  505.74
step 34: loss 7.8594207763671875, dt 233.61ms, token/sec:  547.93
step 35: loss 7.73553466796875, dt 238.59ms, token/sec:  536.48
step 36: loss 7.672882080078125, dt 238.86ms, token/sec:  535.87
step 37: loss 7.6718292236328125, dt 232.67ms, token/sec:  550.14
step 38: loss 7.967987060546875, dt 238.43ms, token/sec:  536.86
step 39: loss 7.4837799072265625, dt 246.13ms, token/sec:  520.05
step 40: loss 7.4139556884765625, dt 262.43ms, token/sec:  487.75
step 41: loss 6.9335479736328125, dt 277.13ms, token/sec:  461.88
step 42: loss 7.089820861816406, dt 262.40ms, token/sec:  487.81
step 43: loss 7.084251403808594, dt 264.92ms, token/sec:  483.16
step 44: loss 6.971649169921875, dt 265.91ms, token/sec:  481.36
step 45: loss 7.075584411621094, dt 252.13ms, token/sec:  507.67
step 46: loss 6.21185302734375, dt 273.84ms, token/sec:  467.42
step 47: loss 6.388458251953125, dt 276.74ms, token/sec:  462.53
step 48: loss 6.9532318115234375, dt 252.43ms, token/sec:  507.07
step 49: loss 6.8386077880859375, dt 234.40ms, token/sec:  546.07

# ----------------------------------------------------------
# y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
# Flashattention
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig())
model.to(device)
# model = torch.compile(model) cannot in mps


# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.95831298828125, dt 4907.62ms, token/sec:  26.08
step 1: loss 9.794143676757812, dt 341.74ms, token/sec:  374.55
step 2: loss 9.070182800292969, dt 232.30ms, token/sec:  551.01
step 3: loss 9.176651000976562, dt 235.54ms, token/sec:  543.42
step 4: loss 8.820648193359375, dt 232.41ms, token/sec:  550.75
step 5: loss 8.422637939453125, dt 231.17ms, token/sec:  553.70
step 6: loss 8.929595947265625, dt 226.53ms, token/sec:  565.04
step 7: loss 8.890289306640625, dt 228.45ms, token/sec:  560.31
step 8: loss 8.128471374511719, dt 228.02ms, token/sec:  561.35
step 9: loss 8.044677734375, dt 229.77ms, token/sec:  557.07
step 10: loss 8.405464172363281, dt 234.10ms, token/sec:  546.78
step 11: loss 7.484619140625, dt 226.77ms, token/sec:  564.46
step 12: loss 7.8219451904296875, dt 229.92ms, token/sec:  556.71
step 13: loss 7.437469482421875, dt 229.34ms, token/sec:  558.12
step 14: loss 7.5394134521484375, dt 230.76ms, token/sec:  554.69
step 15: loss 7.4041595458984375, dt 230.86ms, token/sec:  554.45
step 16: loss 7.4820556640625, dt 228.78ms, token/sec:  559.49
step 17: loss 8.287811279296875, dt 228.06ms, token/sec:  561.27
step 18: loss 7.19586181640625, dt 234.19ms, token/sec:  546.57
step 19: loss 7.8791656494140625, dt 227.20ms, token/sec:  563.37
step 20: loss 7.483421325683594, dt 229.19ms, token/sec:  558.48
step 21: loss 7.822052001953125, dt 231.86ms, token/sec:  552.05
step 22: loss 6.458953857421875, dt 226.37ms, token/sec:  565.45
step 23: loss 6.885429382324219, dt 228.90ms, token/sec:  559.20
step 24: loss 6.829231262207031, dt 234.88ms, token/sec:  544.96
step 25: loss 6.704948425292969, dt 232.25ms, token/sec:  551.13
step 26: loss 6.820915222167969, dt 232.52ms, token/sec:  550.50
step 27: loss 7.614356994628906, dt 231.99ms, token/sec:  551.75
step 28: loss 7.193473815917969, dt 229.24ms, token/sec:  558.36
step 29: loss 6.958343505859375, dt 226.83ms, token/sec:  564.31
step 30: loss 6.975410461425781, dt 227.95ms, token/sec:  561.52
step 31: loss 7.2167205810546875, dt 232.04ms, token/sec:  551.63
step 32: loss 7.13330078125, dt 227.00ms, token/sec:  563.87
step 33: loss 7.007965087890625, dt 225.92ms, token/sec:  566.58
step 34: loss 7.8594207763671875, dt 229.73ms, token/sec:  557.17
step 35: loss 7.73565673828125, dt 230.96ms, token/sec:  554.21
step 36: loss 7.6728363037109375, dt 241.03ms, token/sec:  531.05
step 37: loss 7.672088623046875, dt 232.82ms, token/sec:  549.78
step 38: loss 7.9680328369140625, dt 228.39ms, token/sec:  560.45
step 39: loss 7.48394775390625, dt 228.85ms, token/sec:  559.31
step 40: loss 7.413848876953125, dt 230.81ms, token/sec:  554.56
step 41: loss 6.933441162109375, dt 228.65ms, token/sec:  559.81
step 42: loss 7.0897979736328125, dt 232.26ms, token/sec:  551.10
step 43: loss 7.084403991699219, dt 229.35ms, token/sec:  558.11
step 44: loss 6.9716644287109375, dt 232.15ms, token/sec:  551.38
step 45: loss 7.075569152832031, dt 226.97ms, token/sec:  563.96
step 46: loss 6.212028503417969, dt 226.49ms, token/sec:  565.15
step 47: loss 6.3884429931640625, dt 226.70ms, token/sec:  564.62
step 48: loss 6.9532012939453125, dt 228.24ms, token/sec:  560.81
step 49: loss 6.8386077880859375, dt 228.69ms, token/sec:  559.72

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps


# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}: loss {loss.item()}, dt {dt:.2f}ms, token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step 0: loss 10.9808349609375, dt 2152.43ms, token/sec:  59.47
step 1: loss 9.881103515625, dt 425.01ms, token/sec:  301.17
step 2: loss 8.947944641113281, dt 240.19ms, token/sec:  532.91
step 3: loss 9.2283935546875, dt 233.14ms, token/sec:  549.03
step 4: loss 8.748870849609375, dt 232.01ms, token/sec:  551.70
step 5: loss 8.445571899414062, dt 237.01ms, token/sec:  540.07
step 6: loss 9.144989013671875, dt 236.26ms, token/sec:  541.77
step 7: loss 8.841957092285156, dt 229.21ms, token/sec:  558.44
step 8: loss 8.141281127929688, dt 229.41ms, token/sec:  557.94
step 9: loss 8.099212646484375, dt 226.86ms, token/sec:  564.23
step 10: loss 8.482528686523438, dt 227.42ms, token/sec:  562.83
step 11: loss 7.504859924316406, dt 224.17ms, token/sec:  571.01
step 12: loss 7.8735504150390625, dt 228.23ms, token/sec:  560.85
step 13: loss 7.557929992675781, dt 226.72ms, token/sec:  564.59
step 14: loss 7.582283020019531, dt 228.17ms, token/sec:  560.99
step 15: loss 7.4419708251953125, dt 228.07ms, token/sec:  561.23
step 16: loss 7.3779296875, dt 227.84ms, token/sec:  561.79
step 17: loss 8.227188110351562, dt 232.38ms, token/sec:  550.82
step 18: loss 7.2001800537109375, dt 228.85ms, token/sec:  559.32
step 19: loss 7.8297882080078125, dt 227.28ms, token/sec:  563.19
step 20: loss 7.46337890625, dt 227.95ms, token/sec:  561.52
step 21: loss 7.7310333251953125, dt 226.91ms, token/sec:  564.11
step 22: loss 6.4691619873046875, dt 228.55ms, token/sec:  560.05
step 23: loss 6.830535888671875, dt 226.71ms, token/sec:  564.59
step 24: loss 6.891021728515625, dt 227.02ms, token/sec:  563.83
step 25: loss 6.699363708496094, dt 225.05ms, token/sec:  568.75
step 26: loss 6.7591705322265625, dt 227.75ms, token/sec:  562.02
step 27: loss 7.621551513671875, dt 226.41ms, token/sec:  565.33
step 28: loss 7.119781494140625, dt 226.22ms, token/sec:  565.82
step 29: loss 7.013359069824219, dt 228.09ms, token/sec:  561.19
step 30: loss 6.96484375, dt 230.38ms, token/sec:  555.61
step 31: loss 7.28594970703125, dt 228.34ms, token/sec:  560.58
step 32: loss 7.1584320068359375, dt 230.85ms, token/sec:  554.47
step 33: loss 7.059906005859375, dt 226.87ms, token/sec:  564.20
step 34: loss 7.9029083251953125, dt 225.58ms, token/sec:  567.44
step 35: loss 7.830780029296875, dt 227.78ms, token/sec:  561.93
step 36: loss 7.6209716796875, dt 227.96ms, token/sec:  561.50
step 37: loss 7.647674560546875, dt 227.00ms, token/sec:  563.87
step 38: loss 7.9117584228515625, dt 228.93ms, token/sec:  559.13
step 39: loss 7.4202880859375, dt 226.90ms, token/sec:  564.13
step 40: loss 7.3924560546875, dt 226.36ms, token/sec:  565.46
step 41: loss 6.865379333496094, dt 226.24ms, token/sec:  565.77
step 42: loss 7.0479888916015625, dt 226.45ms, token/sec:  565.25
step 43: loss 7.034431457519531, dt 226.55ms, token/sec:  564.99
step 44: loss 7.0184478759765625, dt 229.68ms, token/sec:  557.29
step 45: loss 6.9474334716796875, dt 229.17ms, token/sec:  558.55
step 46: loss 6.129608154296875, dt 228.46ms, token/sec:  560.27
step 47: loss 6.3230438232421875, dt 227.60ms, token/sec:  562.38
step 48: loss 6.9045257568359375, dt 227.28ms, token/sec:  563.17
step 49: loss 6.790611267089844, dt 229.83ms, token/sec:  556.93

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps


# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i:4d} | loss {loss.item():.6f} | norm: {norm:.4f} | dt {dt:.2f}ms | token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step    0 | loss 10.978027 | norm: 15.5521 | dt 3612.99ms | token/sec:  35.43
step    1 | loss 9.886108 | norm: 5.4822 | dt 377.10ms | token/sec:  339.43
step    2 | loss 8.937561 | norm: 3.6732 | dt 310.39ms | token/sec:  412.38
step    3 | loss 9.194824 | norm: 4.2398 | dt 313.36ms | token/sec:  408.48
step    4 | loss 8.730469 | norm: 4.7621 | dt 303.85ms | token/sec:  421.26
step    5 | loss 8.437988 | norm: 3.7637 | dt 320.04ms | token/sec:  399.95
step    6 | loss 9.178162 | norm: 4.7727 | dt 325.82ms | token/sec:  392.85
step    7 | loss 8.867981 | norm: 3.6167 | dt 322.26ms | token/sec:  397.20
step    8 | loss 8.167725 | norm: 3.6700 | dt 321.63ms | token/sec:  397.97
step    9 | loss 8.111084 | norm: 3.4665 | dt 316.49ms | token/sec:  404.44
step   10 | loss 8.464478 | norm: 3.2132 | dt 310.60ms | token/sec:  412.11
step   11 | loss 7.447632 | norm: 3.8311 | dt 325.26ms | token/sec:  393.53
step   12 | loss 7.831299 | norm: 2.7233 | dt 356.50ms | token/sec:  359.05
step   13 | loss 7.544861 | norm: 3.6002 | dt 312.30ms | token/sec:  409.86
step   14 | loss 7.551147 | norm: 2.7389 | dt 321.77ms | token/sec:  397.80
step   15 | loss 7.443115 | norm: 2.5372 | dt 338.56ms | token/sec:  378.07
step   16 | loss 7.369141 | norm: 3.1235 | dt 311.41ms | token/sec:  411.03
step   17 | loss 8.226685 | norm: 3.0291 | dt 303.02ms | token/sec:  422.42
step   18 | loss 7.158691 | norm: 2.6071 | dt 300.17ms | token/sec:  426.43
step   19 | loss 7.795410 | norm: 3.2330 | dt 311.88ms | token/sec:  410.41
step   20 | loss 7.427368 | norm: 2.6131 | dt 309.75ms | token/sec:  413.23
step   21 | loss 7.713074 | norm: 3.0221 | dt 309.79ms | token/sec:  413.18
step   22 | loss 6.438477 | norm: 3.1037 | dt 300.17ms | token/sec:  426.42
step   23 | loss 6.788391 | norm: 2.4885 | dt 303.62ms | token/sec:  421.59
step   24 | loss 6.859436 | norm: 2.3492 | dt 308.93ms | token/sec:  414.33
step   25 | loss 6.697998 | norm: 2.7277 | dt 338.80ms | token/sec:  377.80
step   26 | loss 6.709351 | norm: 2.8913 | dt 322.09ms | token/sec:  397.41
step   27 | loss 7.603882 | norm: 2.9657 | dt 318.53ms | token/sec:  401.85
step   28 | loss 7.056335 | norm: 3.2891 | dt 310.53ms | token/sec:  412.20
step   29 | loss 6.991882 | norm: 2.9656 | dt 304.40ms | token/sec:  420.50
step   30 | loss 6.942261 | norm: 3.4492 | dt 303.88ms | token/sec:  421.22
step   31 | loss 7.264282 | norm: 3.0961 | dt 306.82ms | token/sec:  417.18
step   32 | loss 7.125122 | norm: 2.5841 | dt 302.84ms | token/sec:  422.67
step   33 | loss 7.013428 | norm: 3.5710 | dt 304.38ms | token/sec:  420.52
step   34 | loss 7.901367 | norm: 2.8137 | dt 302.06ms | token/sec:  423.75
step   35 | loss 7.835510 | norm: 2.8414 | dt 302.42ms | token/sec:  423.26
step   36 | loss 7.521606 | norm: 2.6813 | dt 304.25ms | token/sec:  420.70
step   37 | loss 7.577393 | norm: 2.7950 | dt 303.32ms | token/sec:  421.99
step   38 | loss 7.728455 | norm: 3.3209 | dt 307.24ms | token/sec:  416.61
step   39 | loss 7.356354 | norm: 2.9148 | dt 292.81ms | token/sec:  437.14
step   40 | loss 7.403778 | norm: 3.3171 | dt 319.74ms | token/sec:  400.33
step   41 | loss 6.647278 | norm: 3.3172 | dt 299.40ms | token/sec:  427.52
step   42 | loss 6.808685 | norm: 2.8572 | dt 297.20ms | token/sec:  430.68
step   43 | loss 6.924728 | norm: 6.6540 | dt 296.81ms | token/sec:  431.25
step   44 | loss 6.889587 | norm: 2.7048 | dt 316.10ms | token/sec:  404.94
step   45 | loss 6.853516 | norm: 2.8369 | dt 314.80ms | token/sec:  406.61
step   46 | loss 5.936737 | norm: 2.7971 | dt 383.79ms | token/sec:  333.51
step   47 | loss 6.166870 | norm: 3.5953 | dt 399.34ms | token/sec:  320.53
step   48 | loss 6.833191 | norm: 3.4541 | dt 334.32ms | token/sec:  382.87
step   49 | loss 6.729462 | norm: 2.9496 | dt 324.92ms | token/sec:  393.94

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps

max_lr = 3e-4
min_lr = max_lr * 0.1
wramup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for the first warmup_steps
    if it < wramup_steps:
        return max_lr * (it + 1) / wramup_steps
    # 2) if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - wramup_steps) / (max_steps - wramup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# optimize:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss {loss.item():.6f} | lr: {lr:.5f}| norm: {norm:.4f} | dt {dt:.2f}ms | token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
step    0 | loss 10.978027 | lr: 0.00003| norm: 15.5521 | dt 1119.66ms | token/sec:  114.32
step    1 | loss 10.433594 | lr: 0.00006| norm: 9.6453 | dt 382.50ms | token/sec:  334.64
step    2 | loss 9.472168 | lr: 0.00009| norm: 7.6970 | dt 290.59ms | token/sec:  440.48
step    3 | loss 9.621826 | lr: 0.00012| norm: 5.7578 | dt 287.92ms | token/sec:  444.57
step    4 | loss 9.066895 | lr: 0.00015| norm: 5.4889 | dt 287.94ms | token/sec:  444.54
step    5 | loss 8.865967 | lr: 0.00018| norm: 4.5996 | dt 286.33ms | token/sec:  447.04
step    6 | loss 9.487488 | lr: 0.00021| norm: 4.5104 | dt 295.82ms | token/sec:  432.69
step    7 | loss 9.274414 | lr: 0.00024| norm: 3.7818 | dt 296.65ms | token/sec:  431.48
step    8 | loss 8.660645 | lr: 0.00027| norm: 4.6259 | dt 297.71ms | token/sec:  429.95
step    9 | loss 8.451660 | lr: 0.00030| norm: 4.0346 | dt 291.53ms | token/sec:  439.06
step   10 | loss 8.752319 | lr: 0.00030| norm: 3.1268 | dt 300.32ms | token/sec:  426.21
step   11 | loss 7.856934 | lr: 0.00030| norm: 3.4747 | dt 296.22ms | token/sec:  432.12
step   12 | loss 8.175659 | lr: 0.00030| norm: 3.1097 | dt 312.10ms | token/sec:  410.12
step   13 | loss 7.850830 | lr: 0.00030| norm: 3.3040 | dt 288.22ms | token/sec:  444.10
step   14 | loss 7.886841 | lr: 0.00029| norm: 2.8674 | dt 288.61ms | token/sec:  443.51
step   15 | loss 7.782104 | lr: 0.00029| norm: 2.7492 | dt 287.88ms | token/sec:  444.63
step   16 | loss 7.626099 | lr: 0.00029| norm: 3.2068 | dt 288.24ms | token/sec:  444.07
step   17 | loss 8.395508 | lr: 0.00028| norm: 2.8627 | dt 289.02ms | token/sec:  442.88
step   18 | loss 7.390869 | lr: 0.00027| norm: 2.5230 | dt 290.08ms | token/sec:  441.26
step   19 | loss 7.983337 | lr: 0.00027| norm: 3.0352 | dt 291.54ms | token/sec:  439.05
step   20 | loss 7.576782 | lr: 0.00026| norm: 3.0521 | dt 290.97ms | token/sec:  439.91
step   21 | loss 7.824951 | lr: 0.00025| norm: 2.9743 | dt 297.90ms | token/sec:  429.67
step   22 | loss 6.553589 | lr: 0.00024| norm: 3.2653 | dt 302.75ms | token/sec:  422.79
step   23 | loss 6.926270 | lr: 0.00024| norm: 2.5441 | dt 296.56ms | token/sec:  431.62
step   24 | loss 6.969482 | lr: 0.00023| norm: 2.5422 | dt 307.42ms | token/sec:  416.37
step   25 | loss 6.763062 | lr: 0.00022| norm: 2.8249 | dt 309.32ms | token/sec:  413.81
step   26 | loss 6.748291 | lr: 0.00021| norm: 2.8813 | dt 332.60ms | token/sec:  384.85
step   27 | loss 7.601135 | lr: 0.00020| norm: 3.0752 | dt 303.32ms | token/sec:  422.00
step   28 | loss 7.159058 | lr: 0.00019| norm: 3.5002 | dt 313.08ms | token/sec:  408.85
step   29 | loss 7.009949 | lr: 0.00018| norm: 2.8170 | dt 313.75ms | token/sec:  407.97
step   30 | loss 7.018433 | lr: 0.00016| norm: 3.7048 | dt 303.61ms | token/sec:  421.60
step   31 | loss 7.326660 | lr: 0.00015| norm: 2.9146 | dt 310.09ms | token/sec:  412.78
step   32 | loss 7.210327 | lr: 0.00014| norm: 2.7601 | dt 305.78ms | token/sec:  418.61
step   33 | loss 7.052490 | lr: 0.00013| norm: 3.4114 | dt 303.42ms | token/sec:  421.86
step   34 | loss 7.885132 | lr: 0.00012| norm: 3.1001 | dt 319.54ms | token/sec:  400.58
step   35 | loss 7.875610 | lr: 0.00011| norm: 2.8943 | dt 311.83ms | token/sec:  410.49
step   36 | loss 7.643555 | lr: 0.00010| norm: 2.6479 | dt 303.75ms | token/sec:  421.40
step   37 | loss 7.719116 | lr: 0.00009| norm: 2.8450 | dt 315.10ms | token/sec:  406.22
step   38 | loss 8.095825 | lr: 0.00009| norm: 3.3608 | dt 310.96ms | token/sec:  411.63
step   39 | loss 7.528687 | lr: 0.00008| norm: 2.7705 | dt 305.26ms | token/sec:  419.32
step   40 | loss 7.508301 | lr: 0.00007| norm: 3.1289 | dt 309.12ms | token/sec:  414.08
step   41 | loss 7.199829 | lr: 0.00006| norm: 3.3694 | dt 303.31ms | token/sec:  422.01
step   42 | loss 7.356689 | lr: 0.00006| norm: 2.9233 | dt 306.33ms | token/sec:  417.84
step   43 | loss 7.323669 | lr: 0.00005| norm: 2.5982 | dt 305.92ms | token/sec:  418.40
step   44 | loss 7.477661 | lr: 0.00004| norm: 2.4761 | dt 302.31ms | token/sec:  423.40
step   45 | loss 7.406067 | lr: 0.00004| norm: 2.7719 | dt 317.11ms | token/sec:  403.64
step   46 | loss 6.424133 | lr: 0.00004| norm: 3.1812 | dt 299.45ms | token/sec:  427.45
step   47 | loss 6.303467 | lr: 0.00003| norm: 3.1112 | dt 355.31ms | token/sec:  360.25
step   48 | loss 7.082397 | lr: 0.00003| norm: 3.1034 | dt 335.18ms | token/sec:  381.88
step   49 | loss 6.990479 | lr: 0.00003| norm: 2.4505 | dt 307.08ms | token/sec:  416.83

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps

max_lr = 3e-4
min_lr = max_lr * 0.1
wramup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for the first warmup_steps
    if it < wramup_steps:
        return max_lr * (it + 1) / wramup_steps
    # 2) if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - wramup_steps) / (max_steps - wramup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# optimize:
# ptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_pre_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss {loss.item():.6f} | lr: {lr:.5f}| norm: {norm:.4f} | dt {dt:.2f}ms | token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

using device: mps
loaded 338025 tokens
1 epoch = 2640 steps
num decayed parameter tensors: 50, with 124,354,560 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
using fused AdamW: False
step    0 | loss 10.978027 | lr: 0.00003| norm: 15.5521 | dt 2596.28ms | token/sec:  49.30
step    1 | loss 10.433105 | lr: 0.00006| norm: 9.6456 | dt 448.23ms | token/sec:  285.57
step    2 | loss 9.471436 | lr: 0.00009| norm: 7.6947 | dt 307.98ms | token/sec:  415.61
step    3 | loss 9.623657 | lr: 0.00012| norm: 5.7578 | dt 295.58ms | token/sec:  433.05
step    4 | loss 9.065918 | lr: 0.00015| norm: 5.4889 | dt 289.39ms | token/sec:  442.31
step    5 | loss 8.866455 | lr: 0.00018| norm: 4.5997 | dt 291.72ms | token/sec:  438.77
step    6 | loss 9.487732 | lr: 0.00021| norm: 4.5102 | dt 289.72ms | token/sec:  441.81
step    7 | loss 9.273438 | lr: 0.00024| norm: 3.7821 | dt 287.48ms | token/sec:  445.25
step    8 | loss 8.660889 | lr: 0.00027| norm: 4.6234 | dt 289.67ms | token/sec:  441.88
step    9 | loss 8.451416 | lr: 0.00030| norm: 4.0353 | dt 289.23ms | token/sec:  442.55
step   10 | loss 8.752319 | lr: 0.00030| norm: 3.1268 | dt 289.67ms | token/sec:  441.88
step   11 | loss 7.859253 | lr: 0.00030| norm: 3.4749 | dt 287.18ms | token/sec:  445.72
step   12 | loss 8.176392 | lr: 0.00030| norm: 3.1138 | dt 287.70ms | token/sec:  444.91
step   13 | loss 7.851074 | lr: 0.00030| norm: 3.3047 | dt 286.86ms | token/sec:  446.21
step   14 | loss 7.887329 | lr: 0.00029| norm: 2.8689 | dt 294.18ms | token/sec:  435.11
step   15 | loss 7.783325 | lr: 0.00029| norm: 2.7519 | dt 290.84ms | token/sec:  440.11
step   16 | loss 7.625122 | lr: 0.00029| norm: 3.2080 | dt 287.18ms | token/sec:  445.72
step   17 | loss 8.395508 | lr: 0.00028| norm: 2.8614 | dt 289.90ms | token/sec:  441.53
step   18 | loss 7.390991 | lr: 0.00027| norm: 2.5240 | dt 292.13ms | token/sec:  438.16
step   19 | loss 7.983337 | lr: 0.00027| norm: 3.0354 | dt 293.53ms | token/sec:  436.08
step   20 | loss 7.576294 | lr: 0.00026| norm: 3.0502 | dt 287.88ms | token/sec:  444.64
step   21 | loss 7.826416 | lr: 0.00025| norm: 2.9742 | dt 288.02ms | token/sec:  444.42
step   22 | loss 6.552124 | lr: 0.00024| norm: 3.2635 | dt 285.25ms | token/sec:  448.73
step   23 | loss 6.926880 | lr: 0.00024| norm: 2.5436 | dt 299.21ms | token/sec:  427.79
step   24 | loss 6.969116 | lr: 0.00023| norm: 2.5428 | dt 290.01ms | token/sec:  441.36
step   25 | loss 6.762756 | lr: 0.00022| norm: 2.8246 | dt 290.77ms | token/sec:  440.22
step   26 | loss 6.749329 | lr: 0.00021| norm: 2.8821 | dt 296.34ms | token/sec:  431.93
step   27 | loss 7.601440 | lr: 0.00020| norm: 3.0693 | dt 291.58ms | token/sec:  438.99
step   28 | loss 7.159729 | lr: 0.00019| norm: 3.4954 | dt 292.20ms | token/sec:  438.06
step   29 | loss 7.010986 | lr: 0.00018| norm: 2.8136 | dt 305.18ms | token/sec:  419.43
step   30 | loss 7.018433 | lr: 0.00016| norm: 3.6980 | dt 417.10ms | token/sec:  306.88
step   31 | loss 7.325500 | lr: 0.00015| norm: 2.9104 | dt 329.57ms | token/sec:  388.38
step   32 | loss 7.211914 | lr: 0.00014| norm: 2.7585 | dt 309.03ms | token/sec:  414.20
step   33 | loss 7.052979 | lr: 0.00013| norm: 3.4087 | dt 295.61ms | token/sec:  433.01
step   34 | loss 7.885132 | lr: 0.00012| norm: 3.0995 | dt 297.90ms | token/sec:  429.68
step   35 | loss 7.875122 | lr: 0.00011| norm: 2.8935 | dt 294.02ms | token/sec:  435.35
step   36 | loss 7.643677 | lr: 0.00010| norm: 2.6480 | dt 294.32ms | token/sec:  434.91
step   37 | loss 7.718384 | lr: 0.00009| norm: 2.8444 | dt 288.42ms | token/sec:  443.80
step   38 | loss 8.095093 | lr: 0.00009| norm: 3.3620 | dt 289.39ms | token/sec:  442.31
step   39 | loss 7.527710 | lr: 0.00008| norm: 2.7725 | dt 301.93ms | token/sec:  423.94
step   40 | loss 7.508545 | lr: 0.00007| norm: 3.1298 | dt 289.16ms | token/sec:  442.67
step   41 | loss 7.200806 | lr: 0.00006| norm: 3.3752 | dt 288.56ms | token/sec:  443.59
step   42 | loss 7.355896 | lr: 0.00006| norm: 2.9257 | dt 286.22ms | token/sec:  447.21
step   43 | loss 7.324951 | lr: 0.00005| norm: 2.5981 | dt 287.69ms | token/sec:  444.92
step   44 | loss 7.477478 | lr: 0.00004| norm: 2.4762 | dt 286.75ms | token/sec:  446.39
step   45 | loss 7.407166 | lr: 0.00004| norm: 2.7713 | dt 285.59ms | token/sec:  448.20
step   46 | loss 6.423157 | lr: 0.00004| norm: 3.1820 | dt 298.74ms | token/sec:  428.47
step   47 | loss 6.303833 | lr: 0.00003| norm: 3.1159 | dt 286.84ms | token/sec:  446.24
step   48 | loss 7.082886 | lr: 0.00003| norm: 3.1025 | dt 286.32ms | token/sec:  447.05
step   49 | loss 6.990112 | lr: 0.00003| norm: 2.4500 | dt 288.50ms | token/sec:  443.67

# ----------------------------------------------------------
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

total_batch_size = 524288 # 2**19
B = 4 # micro batch size
T = 32 # sequence length
assert total_batch_size % (B * T) == 0, "make sure that the total batch size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total batch size: {total_batch_size}")
print(f"=> calculated gradient accumlation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)
torch.set_float32_matmul_precision('high')

 # get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model) cannot in mps

max_lr = 3e-4
min_lr = max_lr * 0.1
wramup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for the first warmup_steps
    if it < wramup_steps:
        return max_lr * (it + 1) / wramup_steps
    # 2) if it > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - wramup_steps) / (max_steps - wramup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# optimize:
# ptimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accm = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accm += loss.item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss {loss.item():.6f} | lr: {lr:.5f}| norm: {norm:.4f} | dt {dt:.2f}ms | token/sec: {tokens_pre_sec: .2f}")
import sys; sys.exit(0)

Output:
using device: mps
total batch size: 524288
=> calculated gradient accumlation steps: 4096
loaded 338025 tokens
1 epoch = 2640 steps
num decayed parameter tensors: 50, with 124,354,560 parameters
num non-decayed parameter tensors: 98, with 121,344 parameters
using fused AdamW: False

# ----------------------------------------------------------
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# run the training loop
from torch.distributed import init_process_group, destroy_process_group

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

total_batch_size = 524288 # 2**19
B = 4 # micro batch size
T = 32 # sequence length
assert total_batch_size % (B * T) == 0, "make sure that the total batch size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T) * ddp_world_size
if master_process:
    print(f"total batch size: {total_batch_size}")
    print(f"=> calculated gradient accumlation steps: {grad_accum_steps}")

print("I am GPU", ddp_rank)
print("Bye")
import sys; sys.exit(0)

# fail because of no GPUs, but the code is correct and ready to run in a multi-GPU environment with torchrun.
"""