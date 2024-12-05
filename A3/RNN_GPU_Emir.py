import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data_rnn
from torch.utils.data import TensorDataset, DataLoader

print(torch.cuda.is_available())

# Load data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data_rnn.load_imdb(final=False)


# Prepare data
lengths = [len(l) for l in x_train]
# Pad sequences
def pad(seq, pad_length):
    padded = np.zeros(pad_length)  # 0 is for padding
    padded[0:len(seq)] = seq
    return torch.tensor(padded, dtype=torch.long)
    
padding_size = max(lengths)

padded_train = torch.stack([pad(x, padding_size) for x in x_train]).cuda()  # Move to GPU
padded_val = torch.stack([pad(x, padding_size) for x in x_val]).cuda()  # Move to GPU

y_train_tensor = torch.tensor(y_train, dtype=torch.long).cuda()  # Move to GPU
y_val_tensor = torch.tensor(y_val, dtype=torch.long).cuda()  # Move to GPU

# DataLoader for batching
batch_size = 64

train_dataset = TensorDataset(padded_train, y_train_tensor)
validation_dataset = TensorDataset(padded_val, y_val_tensor)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define model
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        hidden = 300
        embedding_size = 300
        num_classes = 2
        vocab_size = len(i2w)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.fc1 = nn.Linear(embedding_size, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = torch.amax(x, dim=1)  # Max pooling across the time dimension
        x = self.fc2(x)
        return x


def acc_val(model, testloader):
  # Validation loop
  model.eval()  # Set model to evaluation mode
  correct = 0
  total = 0
  
  with torch.no_grad():  # Disable gradient computation for validation
      for i, batch in enumerate(testloader):
          # Get labels
          #labels = y_val_tensor[i * batch_size: (i + 1) * batch_size]
          instances, labels = batch
          # Forward pass
          outputs = model(instances)
          predicted = torch.argmax(outputs, dim=1)
          
          # Compute accuracy
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  
          #if i % 100 == 99:
          #    print("val_at:", i)
  
  
  accuracy = 100 * correct / total
  print(f'Validation Accuracy: {accuracy:.2f}%')
  
# Instantiate model and move to GPU
model_1 = BaselineModel().cuda()

# Define optimizer
optimizer = optim.Adam(model_1.parameters(), lr=3e-4)

# Training loop
for epoch in range(10):  # Train for 2 epochs
    model_1.train()
    running_loss = 0.0

    for i, batch in enumerate(trainloader, 0):
        # Get labels
        #labels = y_train_tensor[i * batch_size: (i + 1) * batch_size]
        instances, labels = batch
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model_1(instances)
        loss = nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            #acc_val()
    acc_val(model=model_1, testloader=testloader)
print('Finished Training')
