import time
import pandas as pd

class Tracker:
    def __init__(self):
        self.records = []
        self.start_time = None

    def start_timer(self):
        self.start_time = time.perf_counter()

    def log_step(self, optimizer_name, iteration, loss, grad_norm, theta):
        step_time = (time.perf_counter() - self.start_time) * 1000 # ms
        record = {
            'optimizer': optimizer_name,
            'iteration': iteration,
            'loss': loss,
            'grad_norm': grad_norm,
            'step_time_ms': step_time
        }
        # Сохраняем координаты для 2D, если размерность позволяет
        if len(theta) >= 2:
            record['theta_0'] = theta[0]
            record['theta_1'] = theta[1]
            
        self.records.append(record)
        self.start_timer() # Сброс таймера для следующего шага

    def get_dataframe(self):
        return pd.DataFrame(self.records)
