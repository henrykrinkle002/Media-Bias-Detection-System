import torch
import torch.nn as nn
from torch.nn import functional as F 


batch_size = 32 # independant sequences will we process in parallel
block_size = 8
max_iters  =3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

#characters in the text
chars = sorted(list(set(text))) #set converted string to each characters
vocab_size = len(chars) # size of the texts

# a dictionary to map characters to integers
stoi = {ch:i for i, ch in enumerate(chars)} # chars is a tuple
itos = {i:ch for i, ch in enumerate(chars)}

# encoder function name - encode the string passed to 's'.
# from s each letter will be read and identified from the 
# dictionary stoi[] and their indices will be stored in a list
encode = lambda s: [stoi[c] for c in s] 

# read each index in the list and identify their corresponding 
# character from the dictionary itos[] and join
decode = lambda l: ''.join([itos[c] for c in l]) 

# Train and Test data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # threshold for training and test data
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Eg: len(data) = 300, block_size = 8, the function 
    # selects a number between 0 -> (len(data) - block_size), 
    # with each having a batch_Size 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x =  torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y #input and target

#data = torch.tensor([ord(c) for c in "The quick brown fox jumps over the lazy dog."])
    #block_size = 5
    #batch_size = 3
    #ix = torch.tensor([3, 27, 19]) random numbers, batch_size of 3 (given) extracted
    #data = tensor([84, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, ...])
    # For each i in ix: 
    # values for X and Y stack
    # for i in ix:
    # data[i:i+block_size]
    #3 → data[3:8] = ' qui'                tensor([[113, 117, 105,  99,  32],        the three rows indicate batch size/selected number of words from the data in tensor format           
    #27 → data[27:32] = 'lazy '  ====>>>            [ 97, 122, 121,  32, 100],
    #19 → data[19:24] = 'umps '                      [109, 112, 115,  32, 111]])


@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters) # iterations while evaluations
		for k in range(eval_iters):
			X, Y = get_batch(split) # load training data
			logits, loss = model(X, Y) # X and Y = Input and Targets
			losses[k] = loss.item() # extract the normal loss value from loss to an array losses[]
		out[split] = losses.mean()
	model.train()
	return out	

class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		  
	def forward(self, idx, targets = None):

		logits = self.token_embedding_table(idx)
		# shape (B, T, vocab_size) C is just random values that representing each token, token being a number This will be updated through backpropagation
		if targets is None:
			loss = None
		else: 
			B, T, C = logits.shape # shape (B, T, C) values
			logits = logits.view(B*T, C) # reducing dimensions for inputs for cross_entropy
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets) # reducing dimensions for target for cross_entropy
		return logits, loss

#[
# [[a1, a2, a3], [b1, b2, b3], [c1, c2, c3], [d1, d2, d3]],  # First sequence
# [[e1, e2, e3], [f1, f2, f3], [g1, g2, g3], [h1, h2, h3]]   # Second sequence
#]

#[
#  [d1, d2, d3],  # Last token of first sequence
#  [h1, h2, h3],  # Last token of second sequence
#]


	def generate(self, idx, max_new_tokens):
		#idx is (B, T) array of indices in the curent context
		for _ in range(max_new_tokens):
			#get the predictions
			logits, loss = self(idx)
			#focus only on the last Time
			logits = logits[:, -1, :]
			# apply softmax for probabilities
			probs = F.softmax(logits, dim = -1) # shape (B, C)  C is the possible next word to the sentence. T is the number of words received to until now and is now tokenized
			# sample from the distribution
			idx_next = torch.multinomial(probs, num_samples = 1) # shape(B, 1)
			idx = torch.cat((idx, idx_next), dim = 1) # shape (B, T+1)
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






