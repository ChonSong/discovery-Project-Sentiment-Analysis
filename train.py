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

def train_model_epochs(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10, scheduler=None):
    """
    Train a model for multiple epochs with validation and learning rate scheduling.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run training on
        num_epochs: Number of epochs to train
        scheduler: Learning rate scheduler (optional)
    
    Returns:
        Dictionary with training history
    """
    from evaluate import evaluate_model
    
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    print(f"Training for {num_epochs} epochs...")
    if scheduler is not None:
        print(f"Using learning rate scheduler: {type(scheduler).__name__}")
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Validation
        if val_loader is not None:
            val_acc = evaluate_model(model, val_loader, None, device)
            history['val_accuracy'].append(val_acc)
            
            # Learning rate scheduling
            if scheduler is not None:
                # Handle different scheduler types
                if hasattr(scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(scheduler)):
                        scheduler.step(val_acc)  # ReduceLROnPlateau uses validation metric
                    else:
                        scheduler.step()  # Other schedulers just step
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, LR: {current_lr:.6f}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (patience: {early_stop_patience})")
                break
                
        else:
            # No validation loader, just step scheduler if it doesn't need validation metric
            if scheduler is not None and 'ReduceLROnPlateau' not in str(type(scheduler)):
                scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return history
