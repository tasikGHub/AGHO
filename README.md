# Airport Ground Handling Optimizer

Оптимизация маршрутов наземного обслуживания воздушных судов на перроне с использованием ML-прогноза времени обслуживания и rule-based планировщика.

## Требования

- Python 3.10+
- Ubuntu 20.04 / 22.04 / 24.04 (или WSL / macOS)

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python src/pipeline.py --config configs/scenario_1.yaml --seed 42
```

Одна команда — без ручных правок путей и данных.

## Параметры CLI

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--config` | Путь к YAML-конфигу сценария | обязателен |
| `--seed` | Seed для воспроизводимости | `42` |

## Структура проекта

```
├── src/
│   ├── pipeline.py          # Точка входа
│   ├── data_generator.py    # Генерация синтетических данных
│   ├── ml_model.py          # ML-прогноз времени обслуживания
│   ├── optimizer.py         # Назначение ТС и маршруты
│   ├── simulator.py         # Симуляция выполнения
│   ├── metrics.py           # KPI и графики
│   └── model_report.py      # Отчёт по ML-модели
├── configs/
│   └── scenario_1.yaml      # Параметры сценария
├── tests/                   # Тесты (pytest)
├── charts/                  # Графики (.png)
├── reports/                 # KPI-метрики (.csv)
└── model_params/            # Параметры ML-модели
```

## Что сохраняется после запуска

| Папка | Файл | Содержимое |
|-------|------|------------|
| `charts/` | `routes_gantt.png` | Gantt-диаграмма маршрутов ТС |
| `charts/` | `load_chart.png` | Утилизация парка ТС (%) |
| `reports/` | `results.csv` | Агрегированные KPI метрики |
| `model_params/` | `metrics.json` | MAE RF, MAE baseline, improvement |
| `model_params/` | `feature_importance.png` | Важность признаков модели |
| `model_params/` | `correlation_matrix.png` | Корреляции признаков и цели |
| `model_params/` | `model_summary.txt` | Читаемый отчёт по модели |

## Логирование

Каждый этап выводится в stdout с таймстампом:

```
[2026-04-13 06:00:01] [Pipeline]       START — config: configs/scenario_1.yaml, seed: 42
[2026-04-13 06:00:02] [DataGenerator]  OK — 50 flights, 8 vehicles, 10 stands generated
[2026-04-13 06:00:03] [MLForecast]     OK — trained on 500 historical tasks | MAE (RF): 2.7 min | MAE (baseline): 6.2 min | Improvement: +56%
[2026-04-13 06:00:04] [Optimizer]      OK — 150/150 tasks assigned, 0 violations
[2026-04-13 06:00:05] [Simulator]      OK — 148/150 on_time, 0 cascades
[2026-04-13 06:00:06] [Metrics]        OK — on_time: 98.7%, violations: 0, utilization: avg 71.3%
[2026-04-13 06:00:06] [Pipeline]       DONE — total time: 5.2s
```

## Тесты

```bash
pytest tests/
```

## Воспроизводимость

Два запуска с одинаковым `--seed` дают идентичный результат. Все источники случайности фиксированы через `seed`.

## Future Work

- **REST API / AMQP интеграция** — текущая реализация является proof of concept
  с CLI-интерфейсом. В production-среде аэропортовые системы (AODB, FIDS, DCS)
  взаимодействуют через REST или AMQP. Планируется добавить FastAPI-эндпоинт
  для приёма расписания и выдачи маршрутных планов в формате JSON.

- **Полиязычие (казахский / русский)** — логи и вывод системы планируется
  локализовать. Параметр `language: ru` в конфиге будет переключать язык
  всех сообщений. Приоритетные языки: казахский и русский.
