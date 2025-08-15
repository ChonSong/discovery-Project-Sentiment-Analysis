import torch

def train_model(model, dataloader, optimizer, loss_fn, device, gradient_clip_value=1.0):
    """
    Train a model for one epoch with gradient clipping.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run training on (cpu/cuda)
        gradient_clip_value: Maximum gradient norm for clipping (None to disable)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_grad_norm = 0.0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (common in RNNs)
        if gradient_clip_value is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            total_grad_norm += grad_norm.item()
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    average_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
    
    return average_loss, average_grad_norm

def train_model_epochs(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=10, scheduler=None, gradient_clip_value=1.0):
    """
    Train a model for multiple epochs with validation, learning rate scheduling, and gradient clipping.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function
        device: Device to run training on
        num_epochs: Number of epochs to train
        scheduler: Learning rate scheduler (optional)
        gradient_clip_value: Maximum gradient norm for clipping (None to disable)
    
    Returns:
        Dictionary with training history
    """
    from evaluate import evaluate_model
    
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
        'gradient_norms': []
    }
    
    print(f"Training for {num_epochs} epochs...")
    if scheduler is not None:
        print(f"Using learning rate scheduler: {type(scheduler).__name__}")
    if gradient_clip_value is not None:
        print(f"Using gradient clipping with max norm: {gradient_clip_value}")
    
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10
    
    for epoch in range(num_epochs):
        # Training with gradient clipping
        train_loss, avg_grad_norm = train_model(model, train_loader, optimizer, loss_fn, device, gradient_clip_value)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        history['gradient_norms'].append(avg_grad_norm)
        
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
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, LR: {current_lr:.6f}, Grad Norm: {avg_grad_norm:.4f}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (patience: {early_stop_patience})")
                break
                
        else:
            # No validation loader, just step scheduler if it doesn't need validation metric
            if scheduler is not None and 'ReduceLROnPlateau' not in str(type(scheduler)):
                scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}, Grad Norm: {avg_grad_norm:.4f}")
        
        history['train_loss'].append(train_loss)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return history
