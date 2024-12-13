import data_rnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
#x_train, (i2w, w2i) = data_rnn.load_ndfa(n=150_000)
x_train, (i2w, w2i) = data_rnn.load_brackets(n=150_000)

seq = [w2i['.start'], w2i['('], w2i['('], w2i[')']]

lengths = [len(x) for x in x_train]

# Padding for the autoregressive task
def pad_ar(seq, pad_length):
    assert len(seq) <= pad_length-2, f"pad length {pad_length} too short for sequence of length {len(seq)}"

    padded = np.zeros(pad_length) # 0 is for '.pad'
    padded[0] = 1 # 1 is for '.start'
    padded[1:len(seq)+1] = seq # insert sequence
    padded[len(seq)+1] = 2 # 2 is for '.end'
    
    return torch.tensor(padded, dtype=torch.long)

padding_size = np.max(lengths)+2 # accounts for start and end tokens
padded_train = torch.stack([pad_ar(x, padding_size) for x in x_train])

# shifts tensor values by 1 to the left
def create_target(tensor):
    shifted = tensor[:, 1:tensor.shape[1]]
    shifted = torch.cat((shifted, torch.zeros((shifted.shape[0], 1))), dim=1)
    return shifted

padded_test = create_target(padded_train)

batch_size = 32

# set to device
padded_x = padded_train.to(device)
padded_y = padded_test.to(device)

# create train dataset with instance and label pairs
train_dataset = TensorDataset(padded_x, padded_y)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training function with TensorBoard logging
def train_model_sampling(model, trainloader, optimizer, seq, temp=0.5, nr_epochs=3, log_dir="runs/loss_curves"):
    writer = SummaryWriter(log_dir)
    for epoch in range(nr_epochs):
        running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            instances, targets = batch
            targets = targets.long()
            optimizer.zero_grad()

            outputs = model(instances)
            outputs = torch.transpose(outputs, 1, 2)
            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}, loss: {avg_loss}")
        writer.add_scalar("Training Loss", avg_loss, epoch + 1)

        for j in range(10):
            print(f"Completion of sequence (E{epoch}S{j}):")
            complete_sequence(seq, model)

    writer.close()
    print('Finished Training')
    return model

# Sampling function
def sample(lnprobs, temperature=1.0):
    if temperature == 0.0:
        return lnprobs.argmax()
    p = torch.nn.functional.softmax(lnprobs / temperature, dim=1)
    cd = dist.Categorical(p)
    return cd.sample()

# Sequence completion function
def complete_sequence(seq, model, temp=0.5, max_len=25):
    generated = seq.copy()
    seq_tensor = torch.tensor(generated).reshape(1, -1).to(device)
    for i in range(max_len):
        next_index = sample(model(seq_tensor).select(dim=1, index=-1), temp)
        next_char = i2w[next_index]
        if next_char == '.end':
            print(next_char)
            break
        else:
            print(next_char)
            generated.append(next_index)
            seq_tensor = torch.tensor(generated).reshape(1, -1).to(device)

    generated_chars = [i2w[idx] for idx in generated]
    left = generated_chars.count("(")
    right = generated_chars.count(")")
    print(f"left {left}, right {right}")

    return True if left == right else False

# Model definition
class ARModel(torch.nn.Module):
    def __init__(self, batch_size=1, e=32, h=16):
        super().__init__()
        num_chars = len(i2w)
        embedding_size = 32
        hidden = 16
        n_embeddings = len(i2w)
        self.emb = torch.nn.Embedding(num_chars, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden, num_chars)

    def forward(self, x):
        e = self.emb(x)
        h = self.lstm(e)[0]
        y = self.fc1(h)
        return y

# Initialize model and optimizer
q6model = ARModel(batch_size=1).to(device)
optimizer = optim.Adam(q6model.parameters(), lr=0.003)

# Train the model with TensorBoard logging
final_model = train_model_sampling(q6model, trainloader, optimizer, seq, temp=0.25, nr_epochs=20)
