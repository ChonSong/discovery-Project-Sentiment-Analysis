import torch

def train_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
