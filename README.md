# OptiBench

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimizers-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

OptiBench — Python-библиотека для автоматизированного бенчмаркинга и визуализации стохастических методов оптимизации. Предназначена для анализа скорости сходимости, стабильности и чувствительности к гиперпараметрам алгоритмов SGD, Adam, AdamW, RMSprop и Adagrad.

## Основные возможности
* **Декларативное управление:** настройка экспериментов через YAML-файлы.
* **Тестовые задачи:** классические функции (Квадратичная, Розенброка, Растригина) и задачи глубокого обучения (MNIST via MLP).
* **Визуализация:** построение кривых сходимости (loss curves) и 2D-траекторий на фазовой плоскости.
* **Аналитика:** автоматический расчет метрик и генерация HTML-отчетов.

## Установка

Рекомендуется использовать изолированное окружение conda:

```bash
git clone [https://github.com/artemskiy/sem4opt360proj.git](https://github.com/artemskiy/sem4opt360proj.git)
cd OptiBench

conda env create -f environment.yml
conda activate optibench

```

*(Альтернативно, зависимости можно установить через `pip install -r requirements.txt`)*

## Использование

Доступны три базовых сценария:

1. **Базовый тест (2D функции):**
```bash
python bench_basic.py
```
2. **Перебор гиперпараметров (Grid Search для MNIST):**
```bash
python bench_grid.py


```



3. **Генерация финального HTML-отчета (Сравнение 5 оптимизаторов):**
   ```bash
   python bench_final.py
## Пример конфигурации (config.yaml)

```yaml
experiment_name: "rosenbrock_test"
seed: 42
device: "cpu"

problem:
  name: "rosenbrock"
  init_theta: [-2.0, 2.0]

optimizers:
  - name: "Adam"
    params: {lr: 0.1}
  - name: "RMSprop"
    params: {lr: 0.05, alpha: 0.99}

training:
  iterations: 500
  log_interval: 10

```

## Структура проекта

```text
.
├── optibench/             # Ядро библиотеки
│   ├── runner.py          # Парсинг конфигурации и запуск тестов
│   ├── problems.py        # Реализация целевых функций
│   ├── tracker.py         # Логирование метрик (loss, time, grad_norm)
│   ├── visualizer.py      # Построение графиков (Matplotlib)
│   └── report.py          # Генератор HTML-отчетов
├── results/               # Логи, графики и отчеты (git-ignored)
├── bench_*.py             # Скрипты запуска примеров
├── config.yaml            # Файл конфигурации
└── environment.yml        # Конфигурация conda-окружения

```

## Авторы

Проект разработан в рамках курса ФПМИ МФТИ:

* Вероника Войтещук (`voiteshchuk.vv@phystech.edu`)
* Артём Лещинский (`leshchinskii.av@phystech.edu`)
