import torch

def evaluate_model(model, dataloader, metric_fn, device):
    model.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())
            truths.append(labels.cpu())
    predictions = torch.cat(predictions)
    truths = torch.cat(truths)
    acc = (predictions == truths).float().mean().item()
    return acc
