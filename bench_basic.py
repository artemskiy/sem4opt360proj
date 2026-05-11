from optibench.runner import BenchmarkRunner
from optibench.visualizer import Visualizer

def main():
    print("Starting optimization benchmark...")
    runner = BenchmarkRunner("config.yaml")
    results_df = runner.run()
    
    print("Optimization finished. Generating reports...")
    vis = Visualizer(results_df, save_dir=runner.config['output']['save_dir'])
    vis.plot_loss_curves()
    
    print("Done! Check the results directory.")

if __name__ == "__main__":
    main()
