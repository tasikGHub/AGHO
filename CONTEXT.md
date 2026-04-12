# CONTEXT — Airport Ground Handling Optimizer
> Last updated: 2026-04-12 | Status: In Progress

## Goal
Оптимизация маршрутов наземного обслуживания ВС на перроне с прогнозом спроса через ML.

## Launch
```
python src/pipeline.py --config configs/scenario_1.yaml --seed 42
```

## Stack
| Component | Choice |
|-----------|--------|
| Language  | Python 3.11 |
| ML model  | RandomForestRegressor (sklearn) → предсказывает время обслуживания (мин) |
| Optimizer | NetworkX (граф перрона) + Rule-based greedy (назначение ТС) |
| Interface | Console (no web UI, no GUI) — вывод в терминал + файлы |
| Viz       | matplotlib → .png в reports/ |
| Data out  | CSV/JSON датасеты в reports/ |
| Config    | YAML |

## Modules
| Module | File | Role |
|--------|------|------|
| Data Generator | `src/data_generator.py` | Синтетические данные (seed=42) |
| ML Forecast    | `src/ml_model.py`       | Прогноз времени обслуживания (мин) |
| Optimizer      | `src/optimizer.py`      | Greedy по urgency = STD - earliest_start - service_time_pred → Details: optimizer_work.md |
| Simulator      | `src/simulator.py`      | Проверка реалистичности плана: время в пути, простои, каскадные задержки → Details: simulator_design.md |
| Metrics        | `src/metrics.py`        | KPI в терминал + .png графики + .csv/json в reports/ |
| Pipeline       | `src/pipeline.py`       | Точка входа, связывает все модули |

## Files map
| File | Contains |
|------|----------|
| `data_schema.md`       | Схема flights_df, tasks_df, vehicles_df, apron_graph, ML features |
| `module_interfaces.md` | Inputs/outputs каждого модуля (типы, имена переменных, Raises) |
| `tech_decisions.md`    | Обоснование ключевых решений |
| `configs/scenario_1.yaml` | Все параметры сценария: рейсы, парк ТС, перрон, optimizer, output |
| `optimizer_work.md`    | Логика работы оптимайзера (urgency, сортировка, greedy) |
| `simulator_design.md`  | Дизайн симулятора (время в пути, окна, каскады, статусы) |
| `repo_structure.md`    | Структура репозитория, папки, .gitignore, разделы README |

## Status
- [x] Done: `src/data_generator.py` — code review + исправления (high/medium)
- [x] Done: `src/ml_model.py` — code review + исправления (critical/high/medium)
- [x] Done: `src/optimizer.py` — code review + исправления (critical/high/medium)
- [x] Done: `src/simulator.py` — реализован с нуля; 9 тестов в `tests/test_simulator.py`
- [ ] In progress: `src/metrics.py`
- [ ] Pending: `src/pipeline.py`
- [ ] Deadline intermediate: 13.04.2026 — рабочий пайплайн + baseline + метрики
- [ ] Deadline final: 20.04.2026 — финальная защита

## Constraints
- Запрещено обучать ML с нуля — только sklearn/lgbm/xgboost/prophet
- Ubuntu 20/22/24 + Python 3.10+
- Деайсинг > заправка > кейтеринг (жёсткий приоритет)
- seed=42 везде — полная воспроизводимость
- Simulator проверяет план optimizer: время в пути, простои, каскадные задержки
- Нет веб-интерфейса и GUI — только консоль, файлы, графики
- 1 команда запуска без ручных правок путей
