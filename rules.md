# Reglas del modelo — Bacteria Exploratoria

## Mundo

| Parámetro | Valor | Descripción |
|---|---|---|
| Grid | 32 × 32 = 1024 celdas | Toroidal (los bordes se conectan) |
| Paso de tiempo | 1 tick | Discreto, síncrono |
| Duración max | 1 000 000 ticks | Luego se guarda la población y termina |
| Población máx | 64 bacterias | |
| Agentes de comida máx | 3 | |

Cada tick, todas las bacterias actúan en paralelo sobre un **snapshot** del mundo del tick anterior. Los cambios se aplican después.

---

## Campos del mundo (WorldState)

| Campo | Tipo | Rango | Descripción |
|---|---|---|---|
| `memory` | `Vec<u8>` [1024] | 0–255 | Memoria persistente del tablero — cada bacteria escribe `floor(quórum_local × 20)` al moverse |
| `quorum` | `Vec<f32>` [1024] | 0–∞ | Señal química de presencia bacteriana; decae × 0.985 por tick |
| `food` | `Vec<f32>` [1024] | 0–∞ | Campo de señal de comida emitida por agentes de comida; decae × 0.990 por tick |
| `crowding` | `Vec<u8>` [1024] | 0–255 | Conteo de bacterias en cada celda en el tick actual |
| `stigma` | `Vec<f32>` [1024] | 0–60 | Memoria colectiva acumulada por visitas y muertes; decae × 0.9999 por tick (vida media ~7000 ticks) |

---

## Agente bacteria — Sensores de entrada (N_SENS = 14)

El CTRNN recibe 14 valores en cada tick:

| Índice | Señal | Rango | Fórmula |
|---|---|---|---|
| 0–7 | Gradiente de quórum (8 vecinos) | −1 a +1 | `tanh(quorum[vecino] − quorum[pos])` — sentido N, NE, E, SE, S, SO, O, NO |
| 8 | Comida local | 0–1 | `food[pos] / (3 + food[pos] + 1)` |
| 9 | Gradiente de comida cardinal | 0–1 | `(food_max_vecino − food_local).max(0) / (3 + food_max_vecino + 1)` |
| 10 | Hambre | 0–1 | `1 − energía / 100` |
| 11 | Densidad local (crowding) | 0–1 | `crowding[pos] / 8` clamped a 1 |
| 12 | Memoria colectiva local (stigma) | 0–1 | `stigma[pos] / (20 + stigma[pos] + 1)` |
| 13 | Gradiente de memoria colectiva | 0–∞ | `(stigma_max_vecino − stigma_local).max(0) / 21` |

---

## Agente bacteria — Acciones de salida (N_ACT = 5)

El CTRNN produce 5 logits. Se toma el argmax con ruido gaussiano ±0.25:

| Índice | Acción |
|---|---|
| 0 | Quedarse (penalización de recompensa = −0.8) |
| 1 | Moverse Norte |
| 2 | Moverse Sur |
| 3 | Moverse Este |
| 4 | Moverse Oeste |

---

## Función de recompensa (bacteria)

Si la bacteria **se mueve**:

```
reward = curiosity × novelty
       + 0.1 × memory_diff
       − recency_penalty
       + sociability × colony_bonus
       + food_bonus
       − crowding_penalty
```

Si la bacteria **se queda**: `reward = −0.8`

### Componentes

| Componente | Fórmula |
|---|---|
| `novelty` | `1 / (1 + visits[new_pos])` — disminuye con revisitas |
| `memory_diff` | `|memory[new_pos] − memory[pos]| / 255` — busca diversidad en el tablero |
| `recency_penalty` | `recency_weight × conteo_apariciones_en_últimas_500` — antiherding temporal |
| `colony_bonus` | `(quorum.min(20) / (5 + quorum.min(20))) × 1.2` — recompensa por colonia |
| `food_bonus` | `(food / (3 + food)) × (1.5 + hambre × 3)` — se amplifica con hambre |
| `crowding_penalty` | `((crowd−1).max(0) / 4) × 2.5 × (1 − hambre×0.7)` — tolera más cuando tiene hambre |

**Fitness** = media de las últimas 200 recompensas.

---

## Rasgos hereditarios (5 genes continuos)

Heredados por promedio entre padres + ruido gaussiano. Evolucionan por selección natural.

| Rasgo | Rango | Efecto |
|---|---|---|
| `metabolism` | [0.005, 0.038] | Coste energético por tick sin comida |
| `curiosity` | [0.2, 2.0] | Multiplica la recompensa por novedad |
| `sociability` | [0.2, 2.0] | Multiplica la recompensa por colonia |
| `selectivity` | [0.0, 1.0] | Umbral de diferencia de fitness para aparearse — `gate = exp(−|Δfit| × sel × 3)` |
| `altruism` | [0.2, 2.0] | Multiplica quórum depositado + coste = altruism × 0.005 por tick |

---

## Energía y ciclo de vida (bacteria)

| Evento | Cambio de energía |
|---|---|
| Tick sin comida | `−metabolism − altruism × 0.005` |
| Tick en celda con comida | `+2.0` (hasta máximo 100) |
| Reproducción | Hijo hereda `(energía_padre_A + energía_padre_B) × 0.4` |
| Muerte por edad | `age > 80 000` ticks |
| Muerte por inanición | `energy ≤ 0` **o** (`age > 5000` y `diversidad_reciente < 5%`) |

**Cooldown de reproducción**: 2000 ticks después de parir.
**Condición para reproducirse**: hambre < 15% (energía > 85).

---

## Memoria colectiva (stigma)

| Mecanismo | Fórmula por tick |
|---|---|
| Depósito vivo | Bacterias depositan `0.002 × visits[celda]` en sus celdas más visitadas (visits > 3) |
| Pulso de muerte | Al morir, la bacteria deposita `0.005 × visits[celda]` en esas mismas celdas |
| Decay | `stigma[celda] × 0.9999` — vida media ~7000 ticks |
| Techo | 60.0 (= STIGMA_SAT × 3) |

---

## Señal de quórum

| Mecanismo | Valor |
|---|---|
| Depósito por bacteria | `12/64 = 0.1875` en celda propia; `0.05625` en 4 vecinos cardinales; escalado por altruismo |
| Boost de alarma | ×(1 + hambre × 3) si hay comida en la celda |
| Boost por comida | factor `1 + food/(3+food)` |
| Boost por agente de comida | +0.1875 en su celda propia |
| Decay | × 0.985 por tick — vida media ~45 ticks |

---

## Agente de comida — Sensores de entrada (N_SENS = 14)

| Índice | Señal | Descripción |
|---|---|---|
| 0–7 | Gradiente de quórum (8 vecinos) | Igual que bacteria — detecta dónde están las bacterias |
| 8 | Presión bacteriana local | `quorum[pos] / 120` clamped a 1 |
| 9 | Nivel de salud | `energy / 150` |
| 10–13 | 0.0 | Reservado (stigma no usado por comida) |

---

## Agente de comida — Ciclo de vida

| Evento | Efecto |
|---|---|
| Regeneración | `+0.15` energía por tick (hasta 150) |
| Bacteria en misma celda | `−FOOD_EATEN` energía |
| Bacteria en vecino cardinal | `−FOOD_EATEN × 0.15` energía |
| Energía ≤ 0 | Muere; su CTRNN se hereda con mutación al 4% |
| Energía > 120 y food_n < 3 | Se divide: hijo en celda vecina aleatoria, ambos con energy/2 |
| Sin agentes de comida | Reaparece uno nuevo con el genoma del último muerto |

---

## Reproducción bacteria (neuroevolución)

Ocurre cuando dos bacterias están en la **misma celda** (posición antes del movimiento) y:
1. Ambas tienen cooldown = 0
2. Ambas tienen hambre < 15%
3. Pasan el gate de selectividad: `rand < exp(−|fit_A − fit_B| × avg_selectivity × 3)`
4. Pasan probabilidad base: 30% × (fitness_promedio / 1.5), clampeado a [5%, 80%]

**Herencia del CTRNN**: gen a gen, cada gen viene del padre A con probabilidad proporcional a la diversidad reciente de A respecto al total (pa = diversidad_A / (diversidad_A + diversidad_B) × 100). Cada gen tiene 4% de probabilidad de mutación ±0.2.

Si la población alcanza MAX_POP = 64, se elimina la bacteria con menor diversidad reciente.

---

## Warm start (entre runs)

Al finalizar el run, los genomas supervivientes se guardan en `population.bin`. Al iniciar el siguiente run:
- Si el archivo existe: se cargan los genomas y se expanden a 8 individuos
  - Primeros N: 70% herencia + 30% pesos random + ruido en rasgos ±0.15
  - Adicionales: 50% herencia + 50% pesos random + ruido en rasgos ±0.40
- Si no existe: 4 bacterias con pesos y rasgos completamente aleatorios
