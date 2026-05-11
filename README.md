# 🎯 OptiBench

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Optimizers-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**OptiBench** — это легковесная Python-библиотека для автоматизированного бенчмаркинга, визуализации и сравнения стохастических методов оптимизации. Проект разработан для систематического анализа скорости сходимости, стабильности и чувствительности гиперпараметров (learning rate, momentum и др.) различных алгоритмов (SGD, Adam, AdamW, RMSprop, Adagrad).

---

## 🚀 Основные возможности
* **Единый API для тестов:** Запуск экспериментов через декларативные YAML-конфигурации.
* **Зоопарк задач:** 
  * Классические тестовые функции (Квадратичная, Розенброка, Растригина).
  * Задачи глубокого обучения (классификация MNIST с помощью MLP).
* **Богатая визуализация:** Автоматическое построение кривых обучения (loss curves) в логарифмическом масштабе и 2D-тепловых карт для отслеживания траекторий на фазовой плоскости.
* **Автогенерация отчетов:** Сбор статистики (loss, градиенты, время выполнения) и формирование интерактивных HTML-дашбордов.

---

## 🛠 Установка

Рекомендуется использовать `conda` для обеспечения строгой изоляции зависимостей.

```bash
# Клонирование репозитория
git clone [https://github.com/YOUR_USERNAME/OptiBench.git](https://github.com/YOUR_USERNAME/OptiBench.git)
cd OptiBench

# Создание и активация окружения
conda env create -f environment.yml
conda activate optibench

```

*(Альтернативно, можно установить зависимости через `pip install -r requirements.txt`)*

---

## 📊 Быстрый старт (Использование)

Проект содержит три готовых сценария:

1. **Базовый тест (2D функции):**
```bash
python bench_basic.py


```



```
   *Сравнивает базовые алгоритмы на простых поверхностях с визуализацией траекторий.*

2. **Поиск гиперпараметров (Grid Search):**
   ```bash
   python bench_grid.py
   

```

*Автоматически скачивает MNIST и ищет оптимальный learning rate для полносвязной нейросети.*

3. **Генерация финального отчета:**
```bash
python bench_final.py


```



```
   *Запускает соревнование 5 различных оптимизаторов и открывает в браузере сгенерированный HTML-отчет с результатами.*

---

## ⚙️ Формат конфигурации (config.yaml)
Все эксперименты управляются без изменения кода:
```yaml
experiment_name: "rosenbrock_test"
seed: 42
device: "cpu" # или "cuda"

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

---

## 📁 Структура проекта

```text
.
├── optibench/             # Исходный код библиотеки
│   ├── runner.py          # Ядро: парсинг конфига и запуск цикла обучения
│   ├── problems.py        # Целевые функции (Rosenbrock, Rastrigin, MNIST)
│   ├── tracker.py         # Логирование метрик
│   ├── visualizer.py      # Построение графиков (Matplotlib)
│   └── report.py          # Генератор HTML-дашбордов
├── results/               # Автогенерируемые логи, графики и отчеты (git-ignored)
├── bench_*.py             # Исполняемые скрипты примеров
├── config.yaml            # Текущая конфигурация бенчмарка
└── environment.yml        # Слепок conda-окружения

```

---

## 👥 Авторы

Проект разработан в рамках учебного курса **MIT**:

* Вероника Войтещук (`voiteshchuk.vv@phystech.edu`)
* Артём Лещинский (`leshchinskii.av@phystech.edu`)
