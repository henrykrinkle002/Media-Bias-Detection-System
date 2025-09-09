import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 8
block_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # Neural net is bigger so| self attention cannot contain high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
dropout = 0.2
n_head = 4 # 64 dimenions for 6 heads 128 // 4
n_layer = 4

torch.manual_seed(1337)

with open('input.txt', encoding = 'utf-8') as f:
	text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias = False)
		self.query = nn.Linear(n_embd, head_size, bias = False)
		self.value = nn.Linear(n_embd, head_size, bias = False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		B,T,C = x.shape
		k = self.key(x)
		q = self.query(x)
		v = self.value(x)

		wei = q @ k.transpose(-2, -1) * C**-0.5 # when transposing k only the 1 & 2 from the end changes
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) replace the values in tril mask to T with -inf
		wei = F.softmax(wei, dim = -1) # apply softmax on the last dimension of each row
		wei = self.dropout(wei) # to drop some nodes/neurons in neural network

		v = self.value(x)
		out = wei @ v
		return out

# super simple bigram model
#sa_head: 
#lm_head: It takes the output (shape [B, T, n_embd]) and maps each 
#token’s embedding to a logit vector of size vocab_size
class MultiheadAttention(nn.Module):
    # multiple heads of self attention
	def __init__(self, num_heads, head_size): #32
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate all of the outputs
		out = self.proj(out)
		return out

	
class FeedForward(nn.Module):

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd), # input
			nn.ReLU(), # non linearity
			nn.Linear(4 * n_embd, n_embd), # output projection layer
			nn.Dropout(dropout)
			)

	def forward(self, x):
		return self.net(x)


class Block(nn.Module):

	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd//n_head # each attention head gets equally divided number of dimensions
		self.sa = MultiheadAttention(n_head, head_size) # __init__ function of MultiAttention
		self.ffwd = FeedForward(n_embd) # __init__ function of FeedForward
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)  # Normalises the features before attention mechanism


	def forward(self, x):
		x = x + self.sa(self.ln1(x)) # forward function Fork off add and norm
		x = x + self.ffwd(self.ln2(x)) # forward function add and norm
		return x
		

class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		# each token reads off the logits for the next token
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
		self.lm_head = nn.Linear(n_embd, vocab_size) # decoder
		
	def forward(self, idx, targets = None):
	    B, T = idx.shape
	    tok_emb = self.token_embedding_table(idx)
	    pos_emb = self.position_embedding_table(torch.arange(T, device = device))
	    x = tok_emb + pos_emb
	    x = self.blocks(x)
	    logits = self.lm_head(x)


	    if targets is None:
	    	loss = None
	    else: 
	        B, T, C = logits.shape
	        logits = logits.view(B*T, C)
	        targets = targets.view(B*T)
	        loss = F.cross_entropy(logits, targets)

	    return logits, loss

	def generate(self, idx, max_new_tokens):
	    for _ in range(max_new_tokens):
	    	idx_cond = idx[:, -block_size:] # we are using positional embedding upto block_size if we use all of it, pos emb might run out
	    	logits, loss = self(idx_cond)
	    	logits = logits[:, -1, :]
	    	probs = F.softmax(logits, dim=-1)
	    	idx_next = torch.multinomial(probs, num_samples = 1)
	    	idx = torch.cat((idx, idx_next), dim=1)
	    return idx
 

model = BigramLanguageModel() # embedding table size
m = model.to(device)

# Create a Pytorch Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
	# iterate through max_iters to evaluate the loss on train and val sets
	if iter % eval_interval == 0:
		losses = estimate_loss()
		print(f"step {iter}: Train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

	# sample batch of data
	xb, yb = get_batch('train')

	#evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none = True)
	loss.backward()
	optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device = device) # an empty probably 0 as token
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))	# generate random tokens upto 500, and decode() to create a string from those encoded tokens	


