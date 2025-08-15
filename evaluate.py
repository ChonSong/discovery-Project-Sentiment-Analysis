import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np

def evaluate_model(model, dataloader, metric_fn, device):
    """
    Evaluate model with accuracy only (for backward compatibility).
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        metric_fn: Unused (kept for compatibility)
        device: Device to run evaluation on
    
    Returns:
        float: Accuracy score
    """
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

def evaluate_model_comprehensive(model, dataloader, device, label_names=None):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data  
        device: Device to run evaluation on
        label_names: List of label names for classification report
    
    Returns:
        dict: Dictionary containing accuracy, f1_score, precision, recall, and report
    """
    model.eval()
    predictions, truths = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu().numpy())
            truths.append(labels.cpu().numpy())
    
    # Flatten the arrays
    predictions = np.concatenate(predictions)
    truths = np.concatenate(truths)
    
    # Calculate metrics
    accuracy = (predictions == truths).mean()
    f1 = f1_score(truths, predictions, average='weighted')
    precision = precision_score(truths, predictions, average='weighted')
    recall = recall_score(truths, predictions, average='weighted')
    
    # Generate classification report
    if label_names is None:
        label_names = ['Negative', 'Neutral', 'Positive']
    
    report = classification_report(
        truths, predictions, 
        target_names=label_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report,
        'predictions': predictions,
        'truths': truths
    }
