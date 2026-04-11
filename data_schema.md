# Data Schema — Синтетические данные

Все данные генерируются `src/data_generator.py` с `seed=42`.

---

## 1. Рейсы — `flights_df`

| Поле | Тип | Описание |
|------|-----|----------|
| `flight_id` | str | Уникальный ID рейса (напр. `FL001`) |
| `aircraft_type` | str | Тип ВС: `narrow`, `wide` |
| `STA` | datetime | Фактическое время прилёта |
| `STD` | datetime | Запланированное время вылета |
| `stand_id` | str | Номер стоянки на перроне (напр. `S01`) |
| `turnaround_min` | int | Минимальное время оборота (мин): STD - STA |

---

## 2. Задачи обслуживания — `tasks_df`

| Поле | Тип | Описание |
|------|-----|----------|
| `task_id` | str | Уникальный ID задачи |
| `flight_id` | str | FK → `flights_df.flight_id` |
| `task_type` | str | Тип: `deicing`, `fueling`, `catering` |
| `priority_group` | int | 1=деайсинг, 2=заправка, 3=кейтеринг |
| `STA` | datetime | Из рейса (для расчёта earliest_start) |
| `STD` | datetime | Из рейса (дедлайн задачи) |
| `earliest_start` | datetime | STA + буфер постановки ВС (из конфига) |
| `vehicle_type_req` | str | Требуемый тип ТС: `deicing_truck`, `fuel_truck`, `catering_truck` |
| `service_time_pred` | float | Прогноз ML (мин) — заполняется после `ml_model.py` |
| `urgency` | float | STD - earliest_start - service_time_pred — заполняется в `optimizer.py` |

---

## 3. Парк ТС — `vehicles_df`

| Поле | Тип | Описание |
|------|-----|----------|
| `vehicle_id` | str | Уникальный ID (напр. `V01`) |
| `vehicle_type` | str | `deicing_truck`, `fuel_truck`, `catering_truck` |
| `speed_kmh` | float | Скорость на перроне (км/ч, из конфига) |
| `capacity` | float | Вместимость (литры для заправки, порции для кейтеринга) |
| `start_stand` | str | Начальная позиция (ID стоянки или депо) |
| `free_at` | datetime | Время освобождения ТС (обновляется оптимайзером) |

---

## 4. Геометрия перрона — `apron_graph`

NetworkX граф. Узлы — стоянки и депо. Рёбра — пути с весом `distance_m`.

| Атрибут узла | Тип | Описание |
|--------------|-----|----------|
| `node_id` | str | ID стоянки (`S01`…`S{n}`) или депо (`DEPOT`) |
| `node_type` | str | `stand` или `depot` |

| Атрибут ребра | Тип | Описание |
|---------------|-----|----------|
| `distance_m` | float | Расстояние между узлами (м) |
| `travel_time_min` | float | distance_m / (speed_kmh / 60) — расчётное, не хранится |

---

## 5. Признаки для ML — `features`

Поля, которые `ml_model.py` использует для обучения и прогноза `service_time_pred`:

| Признак | Источник |
|---------|----------|
| `aircraft_type` | `flights_df` |
| `task_type` | `tasks_df` |
| `turnaround_min` | `flights_df` |
| `hour_of_day` | из `STA` |
| `stand_id` (encoded) | `flights_df` |

**Целевая переменная:** `service_time_actual` (мин) — генерируется с шумом в `data_generator.py`, используется для обучения.
