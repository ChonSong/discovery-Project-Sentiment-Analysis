#!/usr/bin/env python3
"""
Model visualization utilities using torchviz for computational graph visualization.

This module provides functions to visualize the computational graphs of PyTorch models
using torchviz.make_dot to generate graphical representations of the model architectures.
"""

import torch
import os
from torchviz import make_dot
import matplotlib.pyplot as plt
from models import (RNNModel, LSTMModel, GRUModel, TransformerModel,
                   RNNModelEmotion, LSTMModelEmotion, GRUModelEmotion, TransformerModelEmotion)

def visualize_model_architecture(model, input_tensor, model_name, save_dir="model_visualizations"):
    """
    Create and save a visualization of the model's computational graph.
    
    Args:
        model: PyTorch model to visualize
        input_tensor: Sample input tensor for the model
        model_name: Name of the model for file naming
        save_dir: Directory to save visualization files
    
    Returns:
        str: Path to the saved visualization file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Forward pass to create computational graph
    with torch.no_grad():
        output = model(input_tensor)
    
    # Create the computational graph visualization
    dot = make_dot(output, params=dict(model.named_parameters()), 
                   show_attrs=True, show_saved=True)
    
    # Set graph attributes for better visualization
    dot.graph_attr.update(size="12,8", dpi="300")
    dot.node_attr.update(fontsize="10")
    dot.edge_attr.update(fontsize="8")
    
    # Save the visualization
    file_path = os.path.join(save_dir, f"{model_name}_architecture")
    dot.render(file_path, format='png', cleanup=True)
    
    print(f"Model architecture visualization saved to: {file_path}.png")
    return f"{file_path}.png"

def create_model_summary_plot(model, model_name, save_dir="model_visualizations"):
    """
    Create a summary plot showing model parameters and architecture info.
    
    Args:
        model: PyTorch model to summarize
        model_name: Name of the model
        save_dir: Directory to save the plot
    
    Returns:
        str: Path to the saved plot file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get layer information
    layers = []
    param_counts = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                layers.append(f"{name}\n({module.__class__.__name__})")
                param_counts.append(params)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of parameters per layer
    if layers and param_counts:
        ax1.bar(range(len(layers)), param_counts)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title(f'{model_name} - Parameters per Layer')
        ax1.grid(True, alpha=0.3)
    
    # Model summary text
    summary_text = f"""
Model: {model_name}

Architecture Summary:
• Total Parameters: {total_params:,}
• Trainable Parameters: {trainable_params:,}
• Model Size: ~{total_params * 4 / (1024**2):.2f} MB

Layer Summary:
{chr(10).join([f"• {layer}: {count:,} params" for layer, count in zip(layers[:5], param_counts[:5])])}
{f"... and {len(layers)-5} more layers" if len(layers) > 5 else ""}
    """
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title(f'{model_name} - Model Summary')
    
    plt.tight_layout()
    
    # Save the plot
    file_path = os.path.join(save_dir, f"{model_name}_summary.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model summary plot saved to: {file_path}")
    return file_path

def visualize_all_models(vocab_size=1000, embed_dim=64, hidden_dim=64, num_classes=3, 
                        max_seq_len=50, save_dir="model_visualizations"):
    """
    Create visualizations for all available model architectures.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        max_seq_len: Maximum sequence length for input
        save_dir: Directory to save visualizations
    
    Returns:
        dict: Dictionary mapping model names to their visualization file paths
    """
    # Create sample input tensor
    batch_size = 2
    sample_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))
    
    # Model configurations
    models_config = {
        'RNN': RNNModel(vocab_size, embed_dim, hidden_dim, num_classes),
        'LSTM': LSTMModel(vocab_size, embed_dim, hidden_dim, num_classes),
        'GRU': GRUModel(vocab_size, embed_dim, hidden_dim, num_classes),
        'Transformer': TransformerModel(vocab_size, embed_dim, num_heads=4, 
                                      hidden_dim=hidden_dim, num_classes=num_classes, 
                                      num_layers=2),
        'RNN-Emotion': RNNModelEmotion(vocab_size, embed_dim, hidden_dim, num_classes),
        'LSTM-Emotion': LSTMModelEmotion(vocab_size, embed_dim, hidden_dim, num_classes),
        'GRU-Emotion': GRUModelEmotion(vocab_size, embed_dim, hidden_dim, num_classes),
        'Transformer-Emotion': TransformerModelEmotion(vocab_size, embed_dim, num_heads=4, 
                                                      hidden_dim=hidden_dim, num_classes=num_classes, 
                                                      num_layers=4)
    }
    
    visualization_paths = {}
    
    print("Creating model visualizations...")
    print("=" * 50)
    
    for model_name, model in models_config.items():
        try:
            print(f"Visualizing {model_name} model...")
            
            # Create architecture visualization
            arch_path = visualize_model_architecture(model, sample_input, model_name, save_dir)
            
            # Create summary plot
            summary_path = create_model_summary_plot(model, model_name, save_dir)
            
            visualization_paths[model_name] = {
                'architecture': arch_path,
                'summary': summary_path
            }
            
        except Exception as e:
            print(f"Error visualizing {model_name}: {e}")
            continue
    
    print("=" * 50)
    print(f"All visualizations saved to: {save_dir}/")
    
    return visualization_paths

if __name__ == "__main__":
    # Generate visualizations for all models
    paths = visualize_all_models()
    
    print("\nGenerated visualizations:")
    for model_name, files in paths.items():
        print(f"{model_name}:")
        for viz_type, path in files.items():
            print(f"  {viz_type}: {path}")