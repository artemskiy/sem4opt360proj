from optibench.runner import BenchmarkRunner
from optibench.visualizer import Visualizer
from optibench.report import ReportGenerator
import webbrowser
import os

def main():
    print("Starting Final Presentation Benchmark (Rosenbrock with 5 Optimizers)...")
    
    # Конфигурация, которая даст красивые, запутанные траектории на презентации
    config = {
        'experiment_name': 'Final_Presentation_Rosenbrock',
        'seed': 42,
        'device': 'cpu',
        'problem': {
            'name': 'rosenbrock',
            'init_theta': [-2.0, 2.0]
        },
        'optimizers': [
            {'name': 'SGD', 'params': {'lr': 0.002, 'momentum': 0.9}},
            {'name': 'Adam', 'params': {'lr': 0.05}},
            {'name': 'AdamW', 'params': {'lr': 0.05, 'weight_decay': 0.01}},
            {'name': 'RMSprop', 'params': {'lr': 0.01, 'alpha': 0.99}},
            {'name': 'Adagrad', 'params': {'lr': 0.5}}
        ],
        'training': {
            'iterations': 1500,
            'log_interval': 5
        },
        'output': {
            'save_dir': './results/final_presentation/'
        }
    }

    runner = BenchmarkRunner(config)
    results_df = runner.run()
    
    print("Generating Visualizations...")
    vis = Visualizer(results_df, save_dir=config['output']['save_dir'])
    vis.plot_loss_curves()
    vis.plot_trajectories_2d(runner.problem)
    
    print("Generating HTML Report...")
    report = ReportGenerator(results_df, config['output']['save_dir'], config['experiment_name'])
    report.generate_html()
    
    print("Done! Opening report in browser...")
    report_path = os.path.abspath(os.path.join(config['output']['save_dir'], 'report.html'))
    try:
        webbrowser.open(f'file://{report_path}')
    except Exception:
        pass

if __name__ == "__main__":
    main()
