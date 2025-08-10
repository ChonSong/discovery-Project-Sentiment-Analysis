import torch

def train_model(model, dataloader, optimizer, loss_fn, device):
    """
    Train a model for one epoch.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run training on (cpu/cuda)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss

def train_model_epochs(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10):
    """
    Train a model for multiple epochs with validation.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run training on
        num_epochs: Number of epochs to train
    
    Returns:
        Dictionary with training history
    """
    from evaluate import evaluate_model
    
    history = {
        'train_loss': [],
        'val_accuracy': []
    }
    
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
        
        # Validation
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, None, device)
            history['val_accuracy'].append(val_acc)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        history['train_loss'].append(train_loss)
    
    return history
