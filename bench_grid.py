import itertools
from optibench.runner import BenchmarkRunner
from optibench.visualizer import Visualizer
import pandas as pd
import os

def main():
    print("Starting Grid Search on MNIST...")
    
    # Сетка параметров: перебираем learning rate
    learning_rates = [0.001, 0.01, 0.1]
    
    config = {
        'experiment_name': 'mnist_grid',
        'seed': 42,
        'device': 'cpu', # На CPU MNIST тоже быстро учится
        'problem': {
            'name': 'mnist',
            'batch_size': 128
        },
        'optimizers': [],
        'training': {
            'iterations': 500, # 500 батчей для быстрого теста
            'log_interval': 10
        },
        'output': {
            'save_dir': './results/mnist_grid/'
        }
    }
    
    # Динамически генерируем список оптимизаторов для теста
    for lr in learning_rates:
        config['optimizers'].append({'name': 'SGD', 'params': {'lr': lr, 'momentum': 0.9}})
        config['optimizers'].append({'name': 'Adam', 'params': {'lr': lr}})

    runner = BenchmarkRunner(config)
    results_df = runner.run()
    
    print("Optimization finished. Generating loss curves for Grid Search...")
    vis = Visualizer(results_df, save_dir=config['output']['save_dir'])
    vis.plot_loss_curves()
    
    # Находим лучший алгоритм и его параметры
    best_loss = results_df.groupby('optimizer').last().sort_values('loss').iloc[0]
    best_opt = results_df.groupby('optimizer').last().sort_values('loss').index[0]
    
    print("\n--- Grid Search Results ---")
    print(f"Best Optimizer & LR combination: {best_opt}")
    print(f"Final Loss: {best_loss['loss']:.4f}")
    print(f"Check the plot in {config['output']['save_dir']}loss_curves.png")

if __name__ == "__main__":
    main()
