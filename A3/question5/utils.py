import numpy as np
import torch


def pad(seq, pad_length=2514):
    # 2514 is from np.max(lengths)
    padded = np.zeros(pad_length)  # 0 is for padding
    padded[0:len(seq)] = seq
    return torch.tensor(padded, dtype=torch.long)


def MaxPoolTime(x):
    return torch.amax(x, dim=1)


def validate(model, validation_loader):
    val_acc = 0
    val_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            instances, labels = data
            fwd = model(instances)
            predictions = torch.argmax(fwd, dim=1)
            val_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Compute and return accuracy
    val_acc = val_correct / total_samples
    return val_acc
