import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # Setup paths
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_path = os.path.join(base_dir, 'results', 'grokking_thermo_data.csv')
    output_dir = os.path.join(base_dir, 'results', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print("Data file not found. Have you successfully run run_experiments.py?")
        return
        
    df = pd.read_csv(data_path)
    
    # We want to plot:
    # X-axis: Epoch
    # Y-axis (3 subplots or dual-y): Train/Test Loss, LLC, Order Parameter
    # Group by Task. Averaged over Seed with std fill.
    
    sns.set_theme(style="whitegrid")
    
    tasks = df['Task'].unique()
    
    for task in tasks:
        task_df = df[df['Task'] == task]
        
        # Calculate mean and std over seeds for each epoch
        # Select only numeric columns to avoid TypeError on string columns
        numeric_cols = ['Train_Loss', 'Test_Loss', 'LLC', 'Order_Parameter']
        agg_df = task_df.groupby('Epoch')[numeric_cols].agg(['mean', 'std'])
        epochs = agg_df.index
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle(f'Thermodynamic Grokking Analysis: {task.capitalize()} Task', fontsize=16)
        
        # 1. Train and Test Loss
        ax = axes[0]
        ax.plot(epochs, agg_df['Train_Loss']['mean'], label='Train Loss', color='blue', lw=2)
        ax.fill_between(epochs, 
                        agg_df['Train_Loss']['mean'] - agg_df['Train_Loss']['std'],
                        agg_df['Train_Loss']['mean'] + agg_df['Train_Loss']['std'], alpha=0.2, color='blue')
                        
        ax.plot(epochs, agg_df['Test_Loss']['mean'], label='Test Loss', color='red', lw=2)
        ax.fill_between(epochs, 
                        agg_df['Test_Loss']['mean'] - agg_df['Test_Loss']['std'],
                        agg_df['Test_Loss']['mean'] + agg_df['Test_Loss']['std'], alpha=0.2, color='red')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title('Learning Curves')
        
        # 2. Local Learning Coefficient (LLC)
        ax = axes[1]
        ax.plot(epochs, agg_df['LLC']['mean'], label='LLC (Entropy Proxy)', color='purple', lw=2)
        ax.fill_between(epochs, 
                        agg_df['LLC']['mean'] - agg_df['LLC']['std'],
                        agg_df['LLC']['mean'] + agg_df['LLC']['std'], alpha=0.2, color='purple')
        ax.set_ylabel('LLC')
        ax.legend()
        ax.set_title('Structural Complexity (Entropy Proxy)')
        
        # 3. Order Parameter
        ax = axes[2]
        ax.plot(epochs, agg_df['Order_Parameter']['mean'], label='Order Parameter', color='green', lw=2)
        ax.fill_between(epochs, 
                        agg_df['Order_Parameter']['mean'] - agg_df['Order_Parameter']['std'],
                        agg_df['Order_Parameter']['mean'] + agg_df['Order_Parameter']['std'], alpha=0.2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Magnitude (k=1)')
        ax.legend()
        ax.set_title('Fourier Order Parameter (Symmetry Breaking)')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        plot_path = os.path.join(output_dir, f'{task}_grokking_thermo.png')
        plt.savefig(plot_path, dpi=300)
        print(f"Saved {plot_path}")
        plt.close()
        
if __name__ == "__main__":
    main()
