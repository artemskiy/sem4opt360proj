import os
import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    def __init__(self, results_df: pd.DataFrame, save_dir: str):
        self.df = results_df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_loss_curves(self):
        plt.figure(figsize=(10, 6))
        for opt_name, group in self.df.groupby('optimizer'):
            # Используем ромбовидные кристаллические маркеры (diamond)
            # markevery помогает не перегружать график, если датасет очень плотный
            plt.plot(group['iteration'], group['loss'], 
                     label=opt_name, marker='D', markersize=5, 
                     markevery=max(1, len(group)//20), linewidth=2)
            
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence (Log Scale)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(self.save_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualizer] Saved loss curves to {save_path}")
        plt.close()
