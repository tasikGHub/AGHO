# Module Interfaces

Входные и выходные данные каждого модуля пайплайна.
`simulator.py` исключён из проекта (см. CONTEXT.md).

---

## DataGenerator (`src/data_generator.py`)

### Input
| Параметр | Тип | Откуда |
|----------|-----|--------|
| `config` | `dict` | `scenario_1.yaml` |
| `seed` | `int` | CLI arg `--seed` |

### Output
| Переменная | Тип | Куда |
|------------|-----|------|
| `flights_df` | `pd.DataFrame (flight_id, aircraft_type, STA, STD, stand_id, turnaround_min)` | → `ml_model.py`, `optimizer.py` |
| `tasks_df` | `pd.DataFrame (task_id, flight_id, task_type, priority_group, STA, STD, earliest_start, vehicle_type_req)` | → `ml_model.py`, `optimizer.py` |
| `vehicles_df` | `pd.DataFrame (vehicle_id, vehicle_type, speed_kmh, capacity, start_stand, free_at)` | → `optimizer.py` |
| `apron_graph` | `nx.Graph` | → `optimizer.py` |

### Raises
- `ValueError` — если `config` пустой или отсутствуют обязательные секции
- `ValueError` — если кол-во стоянок < 1 или кол-во ТС < 1

---

## MLForecast (`src/ml_model.py`)

### Input
| Параметр | Тип | Откуда |
|----------|-----|--------|
| `tasks_df` | `pd.DataFrame (task_id, task_type, aircraft_type, turnaround_min, hour_of_day, stand_id)` | ← `data_generator.py` |
| `flights_df` | `pd.DataFrame (flight_id, aircraft_type, turnaround_min, stand_id)` | ← `data_generator.py` |
| `seed` | `int` | CLI arg `--seed` |

### Output
| Переменная | Тип | Куда |
|------------|-----|------|
| `tasks_df` | `pd.DataFrame` + колонка `service_time_pred: float` | → `optimizer.py` |
| `mae` | `float` | → логгер pipeline |

### Raises
- `RuntimeError` — если обучение упало → fallback: `service_time_pred = mean(service_time_actual)`
- `ValueError` — если `tasks_df` пустой

---

## Optimizer (`src/optimizer.py`)

### Input
| Параметр | Тип | Откуда |
|----------|-----|--------|
| `tasks_df` | `pd.DataFrame` + `service_time_pred` | ← `ml_model.py` |
| `vehicles_df` | `pd.DataFrame (vehicle_id, vehicle_type, speed_kmh, capacity, start_stand, free_at)` | ← `data_generator.py` |
| `apron_graph` | `nx.Graph` | ← `data_generator.py` |
| `config` | `dict` | `scenario_1.yaml` |

### Output
| Переменная | Тип | Куда |
|------------|-----|------|
| `assigned_routes` | `List[dict (task_id, vehicle_id, start_time, end_time, route)]` | → `metrics.py` |
| `violations` | `List[dict (task_id, reason)]` | → `metrics.py` |

### Raises
- `RuntimeError` — если `apron_graph` пустой или несвязный
- `ValueError` — если `tasks_df` пустой

---

## Metrics (`src/metrics.py`)

### Input
| Параметр | Тип | Откуда |
|----------|-----|--------|
| `assigned_routes` | `List[dict (task_id, vehicle_id, start_time, end_time, route)]` | ← `optimizer.py` |
| `violations` | `List[dict (task_id, reason)]` | ← `optimizer.py` |
| `tasks_df` | `pd.DataFrame` | ← `ml_model.py` |
| `config` | `dict` | `scenario_1.yaml` |

### Output
| Переменная | Тип | Куда |
|------------|-----|------|
| `kpi_dict` | `dict (assigned_rate, avg_urgency, violation_count, avg_service_time)` | → stdout + `reports/results.csv` |
| `routes_gantt.png` | `file` | → `reports/` |
| `load_chart.png` | `file` | → `reports/` |

### Raises
- `ValueError` — если `assigned_routes` пустой

---

## Pipeline (`src/pipeline.py`)

### Input
| Параметр | Тип | Откуда |
|----------|-----|--------|
| `--config` | `str` (path) | CLI arg |
| `--seed` | `int` | CLI arg |

### Output
| Переменная | Тип | Куда |
|------------|-----|------|
| Логи каждого этапа | `stdout` | терминал |
| Артефакты | файлы | `reports/` |

### Raises
- `FileNotFoundError` — если config-файл не найден
- Пробрасывает исключения из дочерних модулей с логированием стадии
