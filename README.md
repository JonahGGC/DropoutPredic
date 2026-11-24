# Proyecto Final — Machine Learning (MIC1)

**Predicción multiclase de deserción estudiantil con métodos de ensamble** — Jonahtan Gabriel González Cardoza
(matrícula 345491), Maestría en Ingeniería en Computación, Universidad Autónoma de Chihuahua.

Predice la situación final de un estudiante (*Dropout*, *Enrolled*, *Graduate*) con el dataset
*Predict Students' Dropout and Academic Success* (UCI 697) y compara regresión logística, SVM y diez familias de ensambles
con una partición 80/20 estratificada común (`random_state=42`), validación cruzada de 5 folds, validación cruzada repetida
5×5 (Nadeau–Bengio + Holm), bootstrap y McNemar. El reporte técnico (LaTeX, formato IEEE) está en [`reporte/`](reporte/).

## Estructura

| Ruta | Contenido |
|---|---|
| `01_…` a `05_…` `.ipynb`, `config.py`, `referencia.json` | código del proyecto (ver abajo) |
| `data/data.csv` | dataset UCI 697 (4,424 estudiantes × 36 variables + objetivo), tal como se descarga de UCI |
| `outputs/` | resultados de la corrida completa: `tablas/` (CSV), `figs/` (PNG), `data/` (partición), `ejecutados/` (notebooks con sus salidas) |
| `reporte/` | proyecto LaTeX del reporte (`main.tex`, `refs.bib`, figuras, tablas, scripts) y su `README.md` |

No se incluye `outputs/modelos/registro.joblib` (modelos ajustados, pesado): lo regenera el notebook 03.

## Cómo reproducir los resultados

```bash
python -m venv .venv && source .venv/bin/activate      # opcional
pip install -r requirements.txt
jupyter lab            # abrir desde ESTA carpeta (donde está config.py)
```

- **Opción A, todo de una vez:** abre `05_ejecucion_completa.ipynb` y haz **Run All**. Ejecuta 01 → 04 y verifica contra `referencia.json`
  (`OK` si la diferencia es ≤ 0.005).
- **Opción B, paso a paso:** ejecuta `01` → `02` → `03` → `04` en ese orden (cada uno lee lo que dejó el anterior).

Tiempo estimado: ~10–20 min (búsquedas de hiperparámetros del 03; CV repetida y bootstrap del 04).
Los resultados del reporte se obtuvieron **sin** el modo rápido (`PROYECTO_RAPIDO=1` / `MODO_RAPIDO = True` solo sirve para depurar).

**Entorno de la corrida del reporte:** Python 3.14.8, scikit-learn 1.9.1, pandas 2.3.3, NumPy 2.5.4, interpret 0.7.8
(`requirements-lock.txt`). Con otras versiones, las diferencias suelen ser de milésimas; el notebook 05 las marca como `REVISAR`
si exceden la tolerancia.

## Notebooks

| Notebook | Qué hace | Deja en `outputs/` |
|---|---|---|
| `01_configuracion_inicial` | carga y limpia `data/data.csv`, inspecciona, genera la partición 80/20 estratificada | `data/` |
| `02_eda` | EDA (solo Train) y preprocesamiento | `figs/02_*`, `tablas/02_*` |
| `03_modelos` | referencias (RL, SVM), ensambles base y optimización de hiperparámetros (CV de 5 folds, solo Train) | `modelos/registro.joblib`, `figs/03_*`, `tablas/03_*` |
| `04_resultados` | OOF, CV repetida, **evaluación única en Test**, bootstrap, McNemar, interpretabilidad, literatura | `figs/04_*`, `tablas/04_*` |
| `05_ejecucion_completa` | ejecuta 01 → 04 y verifica contra `referencia.json` | `ejecutados/`, `tablas/05_verificacion.csv` |

`config.py` reúne lo compartido: constantes, rutas, carga de datos, preprocesador, métricas, `GradientBoostingBalanceado`,
`EBMBalanceado` y la clase `Registro` (CV y búsquedas con los mismos folds para todos los modelos).

## Reglas metodológicas

- **Test se usa una sola vez**, en el notebook 04; los notebooks 02 y 03 ni siquiera lo cargan.
- El preprocesamiento va **dentro del `Pipeline`**: se ajusta solo con la parte de entrenamiento de cada fold (sin fuga).
- Todos los modelos usan **los mismos folds** (`StratifiedKFold(5, shuffle=True, random_state=42)`).
- La RL y las SVM conservan los hiperparámetros de sus prácticas; el umbral de la RL (0.40) se **recalcula con OOF de Train**.
- Criterio de selección declarado de antemano: F1 macro.

## Resultados principales (Test, n = 885)

Mejor F1 macro puntual: Random Forest 0.716 [IC 95 % bootstrap 0.683–0.747]; Voting (soft) 0.703; RL con regla 0.683.
En validación cruzada repetida ninguna mejora frente a la RL sobrevive a la corrección de Holm; el orden de validación no predice el de Test
(Spearman −0.32). Detalle, tablas y discusión en el reporte (`reporte/main.pdf`).

## Datos y citas

`data/data.csv` — UCI Machine Learning Repository, id 697 (Realinho et al., 2021); descripción en Realinho et al. (2022),
*Data* 7(11):146, doi:10.3390/data7110146. Consulta la licencia y las condiciones de citación en
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
