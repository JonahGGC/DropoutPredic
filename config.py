"""
config.py — configuración y utilidades compartidas por los notebooks 01–05 del Proyecto Final.

Proyecto: predicción de deserción estudiantil (Dropout / Enrolled / Graduate), dataset UCI 697.
Todo lo que se repite entre notebooks vive aquí: constantes, rutas, carga de datos, preprocesamiento,
métricas, modelos auxiliares y la clase `Registro` (CV + búsqueda de hiperparámetros con los mismos folds).

Cada notebook empieza con `from config import *`, que deja disponible el mismo espacio de nombres
(np, pd, plt, sns, clases de sklearn, etc.) que tenía el notebook de ensambles.

Modo rápido (solo para depurar): `PROYECTO_RAPIDO=1` reduce búsquedas, repeticiones y remuestreos.
Los resultados del proyecto se obtienen SIEMPRE con el modo normal (sin esa variable).
"""
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from IPython.display import display

from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict, GridSearchCV,
    RandomizedSearchCV, validation_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report, make_scorer
)
from sklearn.exceptions import UndefinedMetricWarning

# pip install interpret   (EBM)
from interpret.glassbox import ExplainableBoostingClassifier

# Con C/pesos extremos, algún modelo no predice una clase en algún fold; su Precision se reporta como 0.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ----------------------------------------------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DATA_CSV = RAIZ / "data" / "data.csv"
OUT = RAIZ / "outputs"
DIR_DATA = OUT / "data"        # partición Train/Test y lista de columnas (notebook 01)
FIGS = OUT / "figs"            # figuras para la presentación
TABLAS = OUT / "tablas"        # tablas (csv) para la presentación y el paper
MODELOS = OUT / "modelos"      # modelos ajustados, búsquedas y predicciones OOF (notebook 03)
for _d in (DIR_DATA, FIGS, TABLAS, MODELOS):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------------------------
# Constantes generales
# ----------------------------------------------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_JOBS = -1
sns.set_theme(style="whitegrid", rc={"grid.color": "#e1e0d9", "axes.edgecolor": "#c3c2b7"})
pd.set_option("display.max_columns", None)

CLASES = ["Dropout", "Enrolled", "Graduate"]          # códigos 0, 1, 2
COLOR_DROPOUT = "#eb6834"
COLOR_ENROLLED = "#1baf7a"
COLOR_GRADUATE = "#2a78d6"
COLOR_GRIS = "#898781"
PALETA_CLASES = {"Dropout": COLOR_DROPOUT, "Enrolled": COLOR_ENROLLED, "Graduate": COLOR_GRADUATE}
COLORES_CLASES = [PALETA_CLASES[c] for c in CLASES]

NOMINALES = ["Marital status", "Application mode", "Course", "Previous qualification", "Nationality",
             "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]

# Modo rápido: SOLO para depurar los notebooks (no usar para reportar resultados)
RAPIDO = os.environ.get("PROYECTO_RAPIDO", "0") == "1"
K_REPETICIONES = 1 if RAPIDO else 5                     # repeticiones de la CV repetida (5 × 5 folds)
B_BOOT = 100 if RAPIDO else 2000                        # remuestreos bootstrap en Test
N_ARBOLES = [10, 25, 50] if RAPIDO else [10, 25, 50, 100, 200, 400, 800]
N_MAX_GB, N_MAX_HGB = (30, 30) if RAPIDO else (500, 300)
if RAPIDO:
    print("⚠ MODO RÁPIDO activo (PROYECTO_RAPIDO=1): los resultados NO son los del proyecto.")


# ----------------------------------------------------------------------------------------------------------------
# Guardado de figuras y tablas
# ----------------------------------------------------------------------------------------------------------------
def guardar_fig(nombre, mostrar=True, dpi=160):
    """Guarda la figura actual en outputs/figs/<nombre>.png y la muestra."""
    plt.savefig(FIGS / f"{nombre}.png", dpi=dpi, bbox_inches="tight")
    if mostrar:
        plt.show()


def guardar_tabla(tabla, nombre, indice=True):
    """Guarda un DataFrame (o Series) en outputs/tablas/<nombre>.csv (UTF-8 con BOM, abre bien en Excel)."""
    pd.DataFrame(tabla).to_csv(TABLAS / f"{nombre}.csv", index=indice, encoding="utf-8-sig")
    return tabla


# ----------------------------------------------------------------------------------------------------------------
# Datos y partición (notebook 01 los genera; los demás los leen)
# ----------------------------------------------------------------------------------------------------------------
def cargar_datos():
    """Lee data.csv (separador ';', BOM en el primer encabezado, tabulador en un nombre y typo 'Nacionality')."""
    df = pd.read_csv(DATA_CSV, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df.rename(columns={"Nacionality": "Nationality"})


def tipos_columnas(X):
    """Numéricas / nominales (códigos sin orden) / binarias, a partir de X (se calcula con Train)."""
    binarias = [c for c in X.columns if X[c].nunique() == 2]
    numericas = [c for c in X.columns if c not in NOMINALES + binarias]
    return numericas, list(NOMINALES), binarias


def guardar_particion(X_train, X_test, y_train, y_test):
    """Guarda Train/Test en csv y la lista de columnas por tipo. Los csv conservan el índice original."""
    num, nom, binarias = tipos_columnas(X_train)
    with open(DIR_DATA / "columnas.json", "w", encoding="utf-8") as f:
        json.dump({"orden": list(X_train.columns), "numericas": num, "nominales": nom, "binarias": binarias}, f,
                  ensure_ascii=False, indent=2)
    for nombre, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        d = X.copy()
        d["y"] = y
        d.to_csv(DIR_DATA / f"{nombre}.csv", index_label="idx_original")


def cargar_particion():
    """Devuelve X_train, X_test, y_train, y_test exactamente como los generó el notebook 01."""
    if not (DIR_DATA / "train.csv").exists():
        raise FileNotFoundError("Falta outputs/data/train.csv: ejecuta primero 01_configuracion_inicial.ipynb")
    salida = []
    for nombre in ("train", "test"):
        d = pd.read_csv(DIR_DATA / f"{nombre}.csv", index_col="idx_original")
        salida.append((d.drop(columns="y"), d["y"].to_numpy()))
    (X_train, y_train), (X_test, y_test) = salida
    return X_train, X_test, y_train, y_test


def columnas():
    """(numéricas, nominales, binarias) de la partición guardada."""
    with open(DIR_DATA / "columnas.json", encoding="utf-8") as f:
        d = json.load(f)
    return d["numericas"], d["nominales"], d["binarias"]


def crear_preprocessor():
    """ColumnTransformer del Proyecto F_1: StandardScaler (numéricas), OneHot con min_frequency=0.01 (nominales)
    y passthrough (binarias). Se ajusta SIEMPRE dentro del Pipeline, así que no hay fuga de información."""
    numeric_cols, nominal_cols, binary_cols = columnas()
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), nominal_cols),
        ("bin", "passthrough", binary_cols),
    ])


def crear_pipeline(modelo):
    """Pipeline = copia nueva del preprocesador + modelo."""
    return Pipeline([("preprocessor", crear_preprocessor()), ("model", modelo)])


# ----------------------------------------------------------------------------------------------------------------
# Validación cruzada y métricas
# ----------------------------------------------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def auc_ovr_macro(y_true, y_score):
    """ROC-AUC uno contra el resto, macro. y_score: probabilidades o puntajes de decision_function (n, 3)."""
    return np.mean([roc_auc_score(y_true == k, y_score[:, k]) for k in range(len(CLASES))])


scoring = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, average="macro", zero_division=0),
    "Recall": "recall_macro",
    "F1": "f1_macro",
    "F1 ponderado": "f1_weighted",
    "MCC": make_scorer(matthews_corrcoef),
    "Kappa": make_scorer(cohen_kappa_score),
    "ROC-AUC": make_scorer(auc_ovr_macro, response_method=("predict_proba", "decision_function")),
}
METRICAS = list(scoring)


def puntaje_continuo(modelo, X):
    """Probabilidades si el modelo las tiene; si no, decision_function (SVM); None si no hay (Voting hard)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    return None


def contar_sv(pipe):
    """Número de vectores de soporte (solo SVC los guarda; LinearSVC no)."""
    modelo = pipe.named_steps["model"]
    return int(modelo.n_support_.sum()) if hasattr(modelo, "n_support_") else np.nan


def contar_estimadores(pipe):
    """Número de estimadores base: árboles (Bagging/RF/Boosting), modelos del Voting/Stacking."""
    modelo = pipe.named_steps["model"]
    if isinstance(modelo, HistGradientBoostingClassifier):
        return int(modelo.n_iter_) * len(CLASES)          # un árbol por clase en cada iteración
    if isinstance(modelo, ExplainableBoostingClassifier):
        return np.nan
    if hasattr(modelo, "estimators_"):
        est = modelo.estimators_
        return int(np.size(est)) if isinstance(est, np.ndarray) else len(est)
    return np.nan


def metricas(y_true, y_pred, y_score):
    """Las 8 métricas globales; ROC-AUC es NaN si el modelo no tiene puntaje continuo (Voting hard)."""
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro"),
            "F1": f1_score(y_true, y_pred, average="macro"),
            "F1 ponderado": f1_score(y_true, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "Kappa": cohen_kappa_score(y_true, y_pred),
            "ROC-AUC": auc_ovr_macro(y_true, y_score) if y_score is not None else np.nan}


def por_clase(y_true, y_pred, y_score):
    auc = [roc_auc_score(y_true == k, y_score[:, k]) for k in range(len(CLASES))] if y_score is not None else np.nan
    return pd.DataFrame({"Precision": precision_score(y_true, y_pred, average=None, zero_division=0),
                         "Recall": recall_score(y_true, y_pred, average=None),
                         "F1": f1_score(y_true, y_pred, average=None),
                         "ROC-AUC": auc}, index=CLASES)


# ----------------------------------------------------------------------------------------------------------------
# Modelos de referencia (Proyecto F_1 y práctica de SVM) y modelos auxiliares
# ----------------------------------------------------------------------------------------------------------------
BEST_LR = dict(solver="saga", l1_ratio=1.0, C=0.3, class_weight="balanced")   # mejor modelo del Proyecto F_1
NOMBRE_LR = "Regresión Logística (F_1)"

# Mejores SVM de la práctica de SVM (hiperparámetros fijos; sus búsquedas no se repiten aquí)
BEST_SVM = {
    "SVM Lineal optimizado": dict(modelo=LinearSVC, params=dict(C=0.01, class_weight="balanced", max_iter=10_000,
                                                                random_state=RANDOM_STATE)),
    "SVM Polinomial optimizado": dict(modelo=SVC, params=dict(kernel="poly", degree=2, coef0=1, C=1,
                                                              class_weight="balanced", cache_size=1000)),
    "SVM RBF optimizado": dict(modelo=SVC, params=dict(kernel="rbf", C=100, gamma=0.001, class_weight=None,
                                                       cache_size=1000)),
}
NOMBRE_SVM = "SVM RBF optimizado"


def predecir_regla(proba, umbral):
    """Regla del Proyecto F_1: Dropout si P(Dropout) >= umbral; si no, la más probable entre Enrolled y Graduate.
    umbral=None → la clase más probable (regla por defecto)."""
    if umbral is None:
        return proba.argmax(axis=1)
    resto = np.where(proba[:, 1] >= proba[:, 2], 1, 2)
    return np.where(proba[:, 0] >= umbral, 0, resto)


class GradientBoostingBalanceado(GradientBoostingClassifier):
    """GradientBoostingClassifier con pesos de clase 'balanced' calculados dentro de fit()."""
    def fit(self, X, y, sample_weight=None):
        w = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            w = w * sample_weight
        return super().fit(X, y, sample_weight=w)


class EBMBalanceado(ExplainableBoostingClassifier):
    """EBM con pesos de clase 'balanced' calculados dentro de fit()."""
    def fit(self, X, y, sample_weight=None):
        w = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            w = w * sample_weight
        return super().fit(X, y, sample_weight=w)


def crear_ebm(interacciones, n_jobs=1, **kw):
    """El EBM trabaja con las variables SIN escalar y sin one-hot: las 9 nominales se declaran como categóricas."""
    with open(DIR_DATA / "columnas.json", encoding="utf-8") as f:
        d = json.load(f)
    tipos = ["nominal" if c in d["nominales"] else "continuous" for c in d["orden"]]
    return Pipeline([("model", EBMBalanceado(interactions=interacciones, outer_bags=8, n_jobs=n_jobs,
                                             feature_types=tipos, random_state=RANDOM_STATE, **kw))])


# ----------------------------------------------------------------------------------------------------------------
# Registro: CV de 5 folds + predicciones OOF + ajuste final + búsquedas, con los mismos folds para todos los modelos
# ----------------------------------------------------------------------------------------------------------------
class Registro:
    """Guarda, para cada modelo evaluado: métricas de CV, predicciones out-of-fold y el modelo ajustado con todo Train."""

    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.resultados_cv, self.oof, self.modelos_ajustados = {}, {}, {}
        self.searches, self.tiempos_busqueda = {}, {}
        self.oob_res, self.oob_clase = {}, {}
        self.extra = {}                                   # datos sueltos (p. ej. el threshold elegido para la LR)

    # ---------- evaluación por CV ----------
    def evaluar_cv(self, nombre, pipe, guardar=True, n_jobs_cv=N_JOBS, verbose=True):
        """CV de 5 folds (mismos folds para todos los modelos) + predicciones out-of-fold + ajuste final sobre todo Train.
        n_jobs_cv=-1 paraleliza por folds; n_jobs_cv=1 es para modelos que ya paralelizan por dentro."""
        X_train, y_train = self.X_train, self.y_train
        hay_score = hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function")
        scoring_uso = scoring if hay_score else {k: v for k, v in scoring.items() if k != "ROC-AUC"}
        t0 = time.perf_counter()
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring_uso, return_train_score=True,
                           return_estimator=True, return_indices=True, n_jobs=n_jobs_cv)
        t_cv = time.perf_counter() - t0
        # Predicciones out-of-fold con el modelo de cada fold (sin reentrenar)
        pred = np.empty(len(y_train), dtype=int)
        score = np.empty((len(y_train), len(CLASES))) if hay_score else None
        for est, idx in zip(s["estimator"], s["indices"]["test"]):
            pred[idx] = est.predict(X_train.iloc[idx])
            if hay_score:
                score[idx] = puntaje_continuo(est, X_train.iloc[idx])
        t1 = time.perf_counter()
        pipe.fit(X_train, y_train)                       # ajuste final con todo Train (para Test al final)
        t_fit_total = time.perf_counter() - t1
        fila = {}
        for m in METRICAS:
            if f"test_{m}" in s:
                fila[f"{m}_mean"], fila[f"{m}_std"], fila[f"{m}_train"] = (s[f"test_{m}"].mean(), s[f"test_{m}"].std(),
                                                                            s[f"train_{m}"].mean())
            else:
                fila[f"{m}_mean"] = fila[f"{m}_std"] = fila[f"{m}_train"] = np.nan
        fila["gap_F1"] = fila["F1_train"] - fila["F1_mean"]
        fila["n_SV"] = contar_sv(pipe)
        fila["n_est"] = contar_estimadores(pipe)
        fila["fit_s"] = t_fit_total                       # un ajuste con todo Train (1 núcleo si n_jobs=1 en el modelo)
        fila["folds_F1"] = s["test_F1"]
        if guardar:
            self.resultados_cv[nombre], self.oof[nombre], self.modelos_ajustados[nombre] = fila, (pred, score), pipe
        if verbose:
            print(f"{nombre}:  " + "  ".join(f"{m}={fila[f'{m}_mean']:.4f}±{fila[f'{m}_std']:.4f}"
                                             for m in ["Accuracy", "Recall", "F1", "MCC", "ROC-AUC"])
                  + f"  | gap F1={fila['gap_F1']:.3f} | CV {t_cv:.1f}s, ajuste {t_fit_total:.1f}s", flush=True)
        return fila

    def tabla_resumen(self, nombres=None):
        filas = {}
        for n in (nombres or list(self.resultados_cv)):
            r = self.resultados_cv[n]
            f = lambda m: "—" if np.isnan(r[f"{m}_mean"]) else f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
            filas[n] = {**{m: f(m) for m in METRICAS},
                        "Gap F1 (train−CV)": f"{r['gap_F1']:.4f}",
                        "Estimadores": "—" if np.isnan(r["n_est"]) else f"{int(r['n_est']):,}",
                        "Nº SV": "—" if np.isnan(r["n_SV"]) else f"{int(r['n_SV']):,}",
                        "Ajuste con todo Train (s)": f"{r['fit_s']:.2f}"}
        return pd.DataFrame(filas).T

    def tabla_por_clase(self, metrica, nombres=None):
        return pd.DataFrame({n: por_clase(self.y_train, *self.oof[n])[metrica] for n in (nombres or list(self.oof))}).T

    # ---------- evaluación Out-of-Bag (Bagging / Pasting / Random Forest) ----------
    def evaluar_oob(self, pipe):
        """Cada instancia se predice solo con los árboles que NO la vieron al entrenar."""
        pipe = clone(pipe).set_params(model__oob_score=True, model__n_jobs=N_JOBS).fit(self.X_train, self.y_train)
        proba = pipe.named_steps["model"].oob_decision_function_
        ok = ~np.isnan(proba).any(axis=1)                  # instancias que quedaron fuera de al menos un árbol
        pred = proba[ok].argmax(axis=1)
        return ok.sum(), metricas(self.y_train[ok], pred, proba[ok]), por_clase(self.y_train[ok], pred, proba[ok])

    def comparar_oob(self, nombre_oob, pipe_oob, nombre_cv):
        n_ok, m_oob, c_oob = self.evaluar_oob(pipe_oob)
        self.oob_res[nombre_oob], self.oob_clase[nombre_oob] = m_oob, c_oob
        self.oob_res[f"{nombre_cv} (CV 5 folds)"] = {m: self.resultados_cv[nombre_cv][f"{m}_mean"] for m in METRICAS}
        print(f"{nombre_oob}: {n_ok:,} de {len(self.y_train):,} instancias con predicción OOB")
        display(pd.DataFrame({k: self.oob_res[k] for k in (nombre_oob, f"{nombre_cv} (CV 5 folds)")}).T.round(4))
        print("Recall por clase (OOB):", c_oob["Recall"].round(4).to_dict())

    # ---------- búsqueda de hiperparámetros ----------
    METRICA_REFIT = "F1"      # criterio de selección declarado de antemano: F1 macro (igual que en la práctica de SVM)

    def buscar(self, nombre, pipe, espacio, n_iter=None, n_jobs=N_JOBS, sin_auc=False):
        """Búsqueda de hiperparámetros con los mismos folds. n_iter=None → GridSearchCV; si no → RandomizedSearchCV.
        sin_auc=True quita el ROC-AUC de las métricas (Voting hard no tiene puntaje continuo)."""
        if RAPIDO:                                         # solo depuración: búsqueda aleatoria mínima
            n_iter = min(n_iter or 3, 3)
        t0 = time.perf_counter()
        scoring_b = {k: v for k, v in scoring.items() if k != "ROC-AUC"} if sin_auc else scoring
        if n_iter is None:
            s = GridSearchCV(pipe, espacio, scoring=scoring_b, refit=self.METRICA_REFIT, cv=cv, n_jobs=n_jobs,
                             return_train_score=True)
        else:
            s = RandomizedSearchCV(pipe, espacio, n_iter=n_iter, scoring=scoring_b, refit=self.METRICA_REFIT, cv=cv,
                                   n_jobs=n_jobs, return_train_score=True, random_state=RANDOM_STATE)
        s.fit(self.X_train, self.y_train)
        self.searches[nombre], self.tiempos_busqueda[nombre] = s, time.perf_counter() - t0
        r = s.cv_results_
        print(f"{nombre}: {len(r['params'])} combinaciones × 5 folds en {self.tiempos_busqueda[nombre]:.0f} s | "
              f"mejor F1 macro {s.best_score_:.4f} ± {r['std_test_F1'][s.best_index_]:.4f} | "
              f"gap {r['mean_train_F1'][s.best_index_] - s.best_score_:.3f}\n   {s.best_params_}", flush=True)

    # ---------- persistencia ----------
    def guardar(self, nombre="registro"):
        joblib.dump({k: v for k, v in self.__dict__.items() if k not in ("X_train", "y_train")},
                    MODELOS / f"{nombre}.joblib", compress=3)

    @classmethod
    def cargar(cls, X_train, y_train, nombre="registro"):
        ruta = MODELOS / f"{nombre}.joblib"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.relative_to(RAIZ)}: ejecuta primero 03_modelos.ipynb")
        reg = cls(X_train, y_train)
        reg.__dict__.update(joblib.load(ruta))
        return reg


def tabla_busqueda(search, n=8):
    """Las n mejores combinaciones de una búsqueda, con sus métricas de CV."""
    res = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(search.cv_results_["params"]).rename(columns=lambda c: c.replace("model__", ""))
    params = params.astype(object).where(params.notna(), "None")
    cols = ["mean_test_F1", "std_test_F1", "mean_train_F1", "mean_test_Recall", "mean_test_Accuracy",
            "mean_test_MCC", "mean_test_ROC-AUC", "rank_test_F1"]
    t = pd.concat([params, res[[c for c in cols if c in res]]], axis=1)
    t.insert(t.columns.get_loc("mean_train_F1") + 1, "gap_F1", t["mean_train_F1"] - t["mean_test_F1"])
    return t.sort_values("rank_test_F1").head(n)
