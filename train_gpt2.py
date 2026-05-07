import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

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

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        import tiktoken
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f'1 epoch = {len(self.tokens) // (B * T)} steps')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


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

"""