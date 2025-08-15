#!/usr/bin/env python3
"""
Display script to show generated visualizations in a simple web interface.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_visualization_sample():
    """Display a sample of the generated visualizations."""
    
    # Create a figure to show multiple visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sentiment Analysis Model Visualizations - Sample Output', fontsize=16, fontweight='bold')
    
    # Define the visualizations to show
    viz_files = [
        ('model_visualizations/LSTM_summary.png', 'LSTM Model Summary'),
        ('model_visualizations/Transformer_summary.png', 'Transformer Model Summary'),
        ('model_visualizations/RNN_architecture.png', 'RNN Architecture Graph'),
        ('example_predictions_lstm.png', 'Example Predictions Results')
    ]
    
    for i, (filepath, title) in enumerate(viz_files):
        row = i // 2
        col = i % 2
        
        if os.path.exists(filepath):
            try:
                img = mpimg.imread(filepath)
                axes[row, col].imshow(img)
                axes[row, col].set_title(title, fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading\n{title}\n{str(e)}', 
                                  ha='center', va='center', fontsize=10)
                axes[row, col].set_xlim(0, 1)
                axes[row, col].set_ylim(0, 1)
        else:
            axes[row, col].text(0.5, 0.5, f'File not found:\n{filepath}', 
                              ha='center', va='center', fontsize=10)
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('visualization_showcase.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization showcase saved as: visualization_showcase.png")

if __name__ == "__main__":
    display_visualization_sample()