import os
import torch
import numpy as np
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
            plt.plot(group['iteration'], group['loss'], 
                     label=opt_name, marker='D', markersize=4, 
                     markevery=max(1, len(group)//20), linewidth=2)
            
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Convergence (Log Scale)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trajectories_2d(self, problem):
        if 'theta_0' not in self.df.columns or 'theta_1' not in self.df.columns:
            print("2D trajectory plotting requires theta_0 and theta_1 columns.")
            return

        plt.figure(figsize=(10, 8))
        
        # Строим сетку для контуров
        x_min, x_max = self.df['theta_0'].min() - 1, self.df['theta_0'].max() + 1
        y_min, y_max = self.df['theta_1'].min() - 1, self.df['theta_1'].max() + 1
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(100):
            for j in range(100):
                t = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                Z[i, j] = problem.compute_loss(t).item()
                
        # Рисуем тепловую карту (логарифмируем Z для наглядности)
        plt.contourf(X, Y, np.log10(Z + 1e-8), levels=30, cmap='viridis', alpha=0.8)
        plt.colorbar(label='Log10(Loss)')

        # Отрисовываем траектории
        colors = plt.cm.tab10.colors
        for idx, (opt_name, group) in enumerate(self.df.groupby('optimizer')):
            plt.plot(group['theta_0'], group['theta_1'], 
                     label=opt_name, color=colors[idx % len(colors)], 
                     marker='D', markersize=3, markevery=max(1, len(group)//20), 
                     linewidth=1.5, alpha=0.8)
            # Отмечаем стартовую точку
            plt.plot(group['theta_0'].iloc[0], group['theta_1'].iloc[0], 
                     marker='*', color=colors[idx % len(colors)], markersize=10)
            
        plt.xlabel('Theta 0')
        plt.ylabel('Theta 1')
        plt.title(f'Optimization Trajectories 2D ({problem.__class__.__name__})')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'trajectories_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
