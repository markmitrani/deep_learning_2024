from deep_learning_2024.A3 import data_rnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from utils import pad, validate


class LSTMModel(nn.Module):
    def __init__(self, batch_size=1, embedding_size=300, hidden_size=300):
        super().__init__()
        timesteps = 2514  # from np.max(lengths)
        hidden = hidden_size
        embedding_size = embedding_size
        num_classes = 2
        vocab_size = 99430  # from len(i2w)

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.fc1 = nn.LSTM(input_size=(batch_size, timesteps, embedding_size), hidden_size=hidden, batch_first=True)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = nn.functional.relu(x[0])
        x = torch.amax(x, dim=1)  # Max pooling across the time dimension
        x = self.fc2(x)
        return x


def objective_LSTM(trial):
    embedding_size = trial.suggest_int("embedding_size", 50, 500, step=50)
    hidden_size = trial.suggest_int("hidden_size", 50, 500, step=50)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    nr_epochs = trial.suggest_categorical("nr_epochs", [10, 20, 30])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = data_rnn.load_imdb(final=False)
    padded_train = torch.stack([pad(x) for x in x_train])
    padded_val = torch.stack([pad(x) for x in x_val])

    padded_train = padded_train.to(device)
    padded_val = padded_val.to(device)

    train_dataset = TensorDataset(padded_train, torch.tensor(y_train))
    validation_dataset = TensorDataset(padded_val, torch.tensor(y_val))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(batch_size, embedding_size, hidden_size).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(nr_epochs):

        for i, batch in enumerate(trainloader, 0):
            instances, labels = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(instances)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
    # run on validation set
    val_acc = validate(model, valloader)

    return val_acc
