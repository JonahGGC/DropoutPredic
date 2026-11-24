#!/usr/bin/env python3
"""Genera tables/*.tex y tables/numbers.tex (macros con las cifras del texto) a partir de outputs/ (CSV de los notebooks).

Toda cifra que aparece en el texto del reporte sale de aquí (macros \\cvF..., \\teF..., etc.), de modo que no hay
transcripción manual. Las únicas cifras copiadas a mano son las comparaciones pareadas del notebook 04 (bootstrap pareado
y McNemar), que no se guardaron en CSV: están en PAREADAS, con su origen indicado.

Uso:  python scripts/make_tables.py        (desde la carpeta del proyecto LaTeX)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
TAB, DAT, OUT = RAIZ / "outputs" / "tablas", RAIZ / "outputs" / "data", RAIZ / "tables"
OUT.mkdir(exist_ok=True)


def rd(nombre):
    return pd.read_csv(TAB / nombre, index_col=0)


def f(x, d=4):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def ms(cadena, d=3):
    """'0.7170 ± 0.0192' -> '0.717$\\pm$0.019'"""
    m, s = [float(t) for t in str(cadena).split("±")]
    return f"{m:.{d}f}$\\pm${s:.{d}f}"


def parse_ic(celda):
    m = re.match(r"\s*([0-9.]+)\s*\[\s*([0-9.]+),\s*([0-9.]+)\s*\]", str(celda))
    return tuple(float(x) for x in m.groups()) if m else (np.nan,) * 3


# ------------------------------------------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------------------------------------------
base_cv, opt_cv, ref_cv = rd("03_modelos_base_cv.csv"), rd("03_modelos_optimizados_cv.csv"), rd("03_referencias_cv.csv")
cv_all = pd.concat([opt_cv, ref_cv, base_cv])
cv_all = cv_all[~cv_all.index.duplicated(keep="first")]
test = rd("04_test_todos_los_modelos.csv")
ic = rd("04_test_intervalos_bootstrap.csv")
f1c, rec, det = rd("04_test_f1_por_clase.csv"), rd("04_test_recall_por_clase.csv"), rd("04_test_clase_dropout.csv")
gen = rd("04_generalizacion.csv")
rep, rep_vs = rd("04_cv_repetida_resumen.csv"), rd("04_cv_repetida_vs_lr_f1.csv")
imp = rd("04_importancia_variables.csv")
hp = json.load(open(TAB / "03_mejores_hiperparametros.json", encoding="utf-8"))
cols = json.load(open(DAT / "columnas.json", encoding="utf-8"))
train = pd.read_csv(DAT / "train.csv", index_col=0)
testd = pd.read_csv(DAT / "test.csv", index_col=0)

# Clave (solo letras, sirve de sufijo de macro) -> nombre en los CSV
NOM = {
    "DUM": "Dummy (referencia)", "LR": "Regresión Logística (F_1)", "LRR": "Regresión Logística (F_1) + P(Dropout) ≥ 0.40",
    "LRD": "Regresión Logística base (por defecto)", "SVL": "SVM Lineal optimizado", "SVP": "SVM Polinomial optimizado",
    "SVR": "SVM RBF optimizado", "SVLD": "SVM Lineal (por defecto)", "SVPD": "SVM Polinomial (por defecto)",
    "SVRD": "SVM RBF (por defecto)", "DT": "Árbol de decisión (1 árbol)",
    "RF": "Random Forest optimizado", "BAG": "Bagging optimizado", "PAS": "Pasting optimizado", "ADA": "AdaBoost optimizado",
    "GB": "Gradient Boosting optimizado", "HGB": "HistGradientBoosting optimizado", "EBM": "EBM optimizado",
    "VH": "Voting (hard) optimizado", "VS": "Voting (soft) optimizado", "STK": "Stacking optimizado",
    "RFB": "Random Forest", "BAGB": "Bagging (árboles)", "PASB": "Pasting (árboles)", "ADAB": "AdaBoost",
    "GBB": "Gradient Boosting (balanceado)", "HGBB": "HistGradientBoosting (balanceado)", "EBMB": "EBM (10 interacciones)",
    "EBMG": "EBM sin interacciones (GAM)", "VHB": "Voting (hard)", "VSB": "Voting (soft)",
    "STKB": "Stacking (LR + SVM + RF → LR)",
}
CORTO = {  # nombres para tablas
    "DUM": "Dummy (clase mayoritaria)", "LR": "Regresión logística (RL)", "LRR": "RL + regla $P(\\mathrm{Dropout})\\geq0.40$",
    "LRD": "RL por defecto", "SVL": "SVM lineal", "SVP": "SVM polinomial", "SVR": "SVM RBF", "SVLD": "SVM lineal por defecto",
    "SVPD": "SVM polinomial por defecto", "SVRD": "SVM RBF por defecto", "DT": "Árbol de decisión",
    "RF": "Random Forest", "BAG": "Bagging", "PAS": "Pasting", "ADA": "AdaBoost", "GB": "Gradient Boosting",
    "HGB": "HistGradientBoosting", "EBM": "EBM", "VH": "Voting (hard)", "VS": "Voting (soft)", "STK": "Stacking",
}

macros = {}


def cvm(k):
    """F1 macro medio en CV con precisión completa (columna 'F1 (CV)' de la tabla de Test); si falta, el valor del CSV de CV."""
    n = NOM[k]
    if n in test.index:
        return float(test.loc[n, "F1 (CV)"])
    return float(str(cv_all.loc[n, "F1"]).split("±")[0])


def cvs(k):
    return float(str(cv_all.loc[NOM[k], "F1"]).split("±")[1])


def mac(nombre, valor):
    macros[nombre] = valor


# ------------------------------------------------------------------------------------------------------------------
# Macros por modelo
# ------------------------------------------------------------------------------------------------------------------
for k, n in NOM.items():
    if n in cv_all.index:
        mean_std = lambda col: [float(t) for t in str(cv_all.loc[n, col]).split("±")]
        mac(f"cvF{k}", f"{cvm(k):.3f}")
        mac(f"cvS{k}", f"{mean_std('F1')[1]:.3f}")
        mac(f"cvAcc{k}", f"{mean_std('Accuracy')[0]:.3f}")
        mac(f"cvMcc{k}", f"{mean_std('MCC')[0]:.3f}")
        mac(f"cvGap{k}", f"{float(cv_all.loc[n, 'Gap F1 (train−CV)']):.3f}")
    if n in test.index:
        t = test.loc[n]
        mac(f"teF{k}", f"{t['F1']:.3f}")
        mac(f"teAcc{k}", f"{t['Accuracy']:.3f}")
        mac(f"teMcc{k}", f"{t['MCC']:.3f}")
        mac(f"teAuc{k}", "---" if pd.isna(t["ROC-AUC"]) else f"{t['ROC-AUC']:.3f}")
        mac(f"teFw{k}", f"{t['F1 ponderado']:.3f}")
        mac(f"dtc{k}", f"{t['Test − CV (F1)']:+.3f}".replace("-", "$-$"))
        if k not in cv_all.index.map(lambda _: None):
            mac(f"cvFq{k}", f"{t['F1 (CV)']:.3f}")
    if n in ic.index:
        _, lo, hi = parse_ic(ic.loc[n, "F1"])
        mac(f"teLo{k}", f"{lo:.3f}")
        mac(f"teHi{k}", f"{hi:.3f}")
    if n in f1c.index:
        for c, a in zip(["Dropout", "Enrolled", "Graduate"], ["D", "E", "G"]):
            mac(f"fu{a}{k}", f"{f1c.loc[n, c]:.3f}")
            mac(f"re{a}{k}", f"{rec.loc[n, c]:.3f}")
    if n in det.index:
        mac(f"det{k}", f"{int(det.loc[n, 'Detectados (de 284)'])}")
        mac(f"falsa{k}", f"{int(det.loc[n, 'Falsas alarmas'])}")

# Optimización: Δ de CV (base → optimizado)
pares_opt = {"RF": "RFB", "BAG": "BAGB", "PAS": "PASB", "ADA": "ADAB", "GB": "GBB", "HGB": "HGBB", "EBM": "EBMB",
             "VH": "VHB", "VS": "VSB", "STK": "STKB", "LR": "LRD", "SVL": "SVLD", "SVP": "SVPD", "SVR": "SVRD"}
delta_cv, delta_te = {}, {}
for o, b in pares_opt.items():
    delta_cv[o] = cvm(o) - cvm(b)
    delta_te[o] = test.loc[NOM[o], "F1"] - test.loc[NOM[b], "F1"]
    mac(f"dcv{o}", f"{delta_cv[o]:+.3f}".replace("-", "$-$"))
    mac(f"dte{o}", f"{delta_te[o]:+.3f}".replace("-", "$-$"))
ens = ["RF", "BAG", "PAS", "ADA", "GB", "HGB", "EBM", "VH", "VS", "STK"]
mac("nOptMejoraCV", str(sum(delta_cv[o] > 0 for o in ens)))
mac("nOptMejoraTest", str(sum(delta_te[o] > 0 for o in ens)))
mac("dcvMin", f"{min(delta_cv[o] for o in ens):+.3f}")
mac("dcvMax", f"{max(delta_cv[o] for o in ens):+.3f}")
mac("dteMedia", f"{np.mean([delta_te[o] for o in ens]):+.3f}".replace("-", "$-$"))
mac("dcvMedia", f"{np.mean([delta_cv[o] for o in ens]):+.3f}")

# Generalización CV -> Test (pérdida media en los 12 modelos optimizados; Spearman sobre los 10 ensambles)
mod12 = [NOM[k] for k in ["LR", "SVR", "RF", "BAG", "PAS", "ADA", "GB", "HGB", "EBM", "VH", "VS", "STK"]]
mod10 = [NOM[k] for k in ens]
g, g10 = gen.loc[mod12], gen.loc[mod10]
rho_cv, p_cv = stats.spearmanr(g10["F1 CV (5 folds)"], g10["F1 Test"])
rho_gap, p_gap = stats.spearmanr(g10["Gap train − CV"], g10["Test − CV (F1)"])
mac("rhoCVTest", f"{rho_cv:.2f}".replace("-", "$-$"))
mac("pCVTest", f"{p_cv:.2f}")
mac("rhoGap", f"{rho_gap:.2f}".replace("-", "$-$"))
mac("pGap", f"{p_gap:.2f}")
mac("nEnsGen", str(len(g10)))
mac("nModGen", str(len(g)))
mac("perdidaMedia", f"{abs(g['Test − CV (F1)'].mean()):.3f}")
mac("perdidaMin", f"{abs(g['Test − CV (F1)']).min():.3f}")
mac("perdidaMax", f"{abs(g['Test − CV (F1)']).max():.3f}")
mac("trainFMin", f"{g['F1 train'].min():.2f}")
mac("trainFMax", f"{g['F1 train'].max():.2f}")
mac("testFMin", f"{g['F1 Test'].min():.3f}")
mac("testFMax", f"{g['F1 Test'].max():.3f}")
mac("cvFMinOpt", f"{g['F1 CV (5 folds)'].min():.3f}")
mac("cvFMaxOpt", f"{g['F1 CV (5 folds)'].max():.3f}")
mac("rangoTresTest", f"{g['F1 Test'].nlargest(3).iloc[0] - g['F1 Test'].nlargest(3).iloc[2]:.3f}")
print(f"[chequeo] Spearman CV-Test ({len(g10)} ensambles): rho={rho_cv:.3f} p={p_cv:.3f} | gap-pérdida rho={rho_gap:.3f} p={p_gap:.3f}")
print(f"[chequeo] pérdida media Test−CV ({len(g)} modelos) = {g['Test − CV (F1)'].mean():.4f}")

# CV repetida 5x5
for k in ["LR", "VS", "VH", "STK", "RF", "HGB", "GB", "EBM", "BAG", "PAS", "ADA", "SVR"]:
    n = NOM[k]
    mac(f"rcF{k}", f"{float(str(rep.loc[n, 'F1']).split('±')[0]):.3f}")
    mac(f"rcS{k}", f"{float(str(rep.loc[n, 'F1']).split('±')[1]):.3f}")
    if n in rep_vs.index:
        r = rep_vs.loc[n]
        mac(f"rvD{k}", f"{r['Diferencia F1 vs LR']:+.4f}".replace("-", "$-$"))
        mac(f"rvLo{k}", f"{r['IC 95% inferior']:+.4f}".replace("-", "$-$"))
        mac(f"rvHi{k}", f"{r['IC 95% superior']:+.4f}".replace("-", "$-$"))
        mac(f"rvP{k}", f"{r['p-valor']:.4f}")
        mac(f"rvH{k}", f"{r['p Holm']:.3f}")
        mac(f"rvG{k}", f"{int(r['Gana (de 25)'])}")
opt_search = rep["Optimismo de la búsqueda (5 folds − 5×5)"].astype(float)
mac("optMin", f"{opt_search.min():.3f}".replace("-", "$-$"))
mac("optMax", f"{opt_search.max():.3f}")
mac("optMaxAbs", f"{opt_search.abs().max():.3f}")
mac("nCompCV", str(len(rep_vs)))
mac("nHolmSig", str(int((rep_vs["p Holm"] < 0.05).sum())))
mac("nPcrudoSig", str(int((rep_vs["p-valor"] < 0.05).sum())))
print("[chequeo] optimismo de la búsqueda:", opt_search.round(4).to_dict())

# Comparaciones pareadas en Test (copiadas del notebook 04, celdas de 'resumen_vs' y 'detalle'; no están en CSV)
PAREADAS = {
    "RF_F1": "+0.0329 [+0.006, +0.060]", "BAG_F1": "+0.0298 [+0.001, +0.059]", "PAS_F1": "+0.0264 [-0.002, +0.054]",
    "VS_F1": "+0.0196 [-0.005, +0.043]", "VS_Acc": "+0.0305 [+0.010, +0.051]", "VS_MCC": "+0.0353 [+0.004, +0.066]",
    "SVR_Acc": "+0.0373 [+0.014, +0.061]", "SVR_AUC": "-0.0310 [-0.044, -0.018]", "VS_McN": "55 / 28 (p=0.004)",
}
mac("parRF", r"$+0.033$ (IC 95\,\% $[0.006,\,0.060]$)")
mac("parBAG", r"$+0.030$ ($[0.001,\,0.059]$)")
mac("parVSF", r"$+0.020$ ($[-0.005,\,0.043]$)")
mac("parVSAcc", r"$+0.031$ ($[0.010,\,0.051]$)")
mac("parVSMcc", r"$+0.035$ ($[0.004,\,0.066]$)")
mac("mcnVS", r"55 contra 28 ($p=0.004$)")
mac("parSVRAcc", r"$+0.037$ ($[0.014,\,0.061]$)")
mac("parSVRAuc", r"$-0.031$ ($[-0.044,\,-0.018]$)")

# Importancia por permutación
for var, k in [("Curricular units 2nd sem (approved)", "ASeg"), ("Tuition fees up to date", "TF"),
               ("Curricular units 1st sem (approved)", "APri")]:
    mac(f"impRF{k}", f"{imp.loc[var, 'Permutación RF (ΔF1)']:.3f}")
    mac(f"impVS{k}", f"{imp.loc[var, 'Permutación Voting (soft) (ΔF1)']:.3f}")
rho_imp, _ = stats.spearmanr(imp["MDI (RF)"], imp["Permutación RF (ΔF1)"])
mac("rhoMDI", f"{rho_imp:.2f}")

# ------------------------------------------------------------------------------------------------------------------
# Datos y EDA (se recalculan desde Train; el texto no usa cifras de memoria)
# ------------------------------------------------------------------------------------------------------------------
Ntr, Nte = len(train), len(testd)
mac("Ntot", f"{Ntr + Nte:,}")
mac("Ntr", f"{Ntr:,}")
mac("Nte", f"{Nte:,}")
cl = ["Dropout", "Enrolled", "Graduate"]
for i, c in enumerate(cl):
    ntr, nte = int((train["y"] == i).sum()), int((testd["y"] == i).sum())
    mac(f"ntr{c[0]}", f"{ntr:,}")
    mac(f"nte{c[0]}", f"{nte:,}")
    mac(f"nt{c[0]}", f"{ntr + nte:,}")
    mac(f"pt{c[0]}", f"{(ntr + nte) / (Ntr + Nte) * 100:.2f}")
    mac(f"ptr{c[0]}", f"{ntr / Ntr * 100:.1f}")
X = train.drop(columns="y")
y = train["y"]
nums, noms, bins = cols["numericas"], cols["nominales"], cols["binarias"]
mac("nNum", str(len(nums)))
mac("nNom", str(len(noms)))
mac("nBin", str(len(bins)))
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

oh = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False).fit(X[noms])
n_oh = oh.get_feature_names_out().shape[0]
n_cat_orig = sum(len(c) for c in oh.categories_)
mac("nOH", str(n_oh))
mac("nCatOrig", str(n_cat_orig))
mac("nCols", str(len(nums) + n_oh + len(bins)))
mac("nCursos", str(train["Course"].nunique()))
print(f"[chequeo] columnas tras el preprocesamiento: {len(nums)} + {n_oh} + {len(bins)} = {len(nums) + n_oh + len(bins)}")


def tasa(mask, clase=0):
    return float((y[mask] == clase).mean() * 100), int(mask.sum())


t_nf, n_nf = tasa(X["Tuition fees up to date"] == 0)
t_db, n_db = tasa(X["Debtor"] == 1)
t_bec, _ = tasa(X["Scholarship holder"] == 1, 2)
t_j, _ = tasa(X["Age at enrollment"] <= 19, 2)
t_m, _ = tasa((X["Age at enrollment"] >= 26) & (X["Age at enrollment"] <= 30), 0)
# Con 0 UC aprobadas se separan los estudiantes que sí cursaron UC de los 148 sin UC registradas (hueco del registro)
m2 = (X["Curricular units 2nd sem (approved)"] == 0) & (X["Curricular units 2nd sem (enrolled)"] > 0)
m1 = (X["Curricular units 1st sem (approved)"] == 0) & (X["Curricular units 1st sem (enrolled)"] > 0)
t_a0, n_a0 = tasa(m2)
t_a1, n_a1 = tasa(m1)
mac("gAprobCero", f"{float((y[m2] == 2).mean() * 100):.1f}")
for nombre, v in dict(tNF=t_nf, tDB=t_db, tBec=t_bec, tJov=t_j, tMay=t_m, tAprob=t_a0, tAprobU=t_a1).items():
    mac(nombre, f"{v:.1f}")
mac("nNF", str(n_nf))
mac("nDB", str(n_db))
mac("nAprobCero", str(n_a0))
mac("nAprobCeroU", str(n_a1))
cero = (X["Curricular units 1st sem (enrolled)"] == 0) & (X["Curricular units 2nd sem (enrolled)"] == 0)
mac("nSinUC", str(int(cero.sum())))
mac("tSinUC", f"{float((y[cero] == 0).mean() * 100):.1f}")
mac("gSinUC", f"{float((y[cero] == 2).mean() * 100):.1f}")
mac("cursoSinUC", ", ".join(str(int(c)) for c in X.loc[cero, "Course"].unique()))
por_curso = (y == 0).groupby(X["Course"]).agg(["mean", "size"])
gr = por_curso[por_curso["size"] >= 100]
mac("cursoMin", f"{gr['mean'].min() * 100:.1f}")
mac("cursoMax", f"{gr['mean'].max() * 100:.1f}")
mac("nCursosGr", str(len(gr)))
mac("rCurso", f"{np.corrcoef(X['Course'], (y == 0).astype(int))[0, 1]:.3f}".replace("-", "$-$"))
out_age = (lambda s: int(((s < s.quantile(.25) - 1.5 * (s.quantile(.75) - s.quantile(.25))) |
                          (s > s.quantile(.75) + 1.5 * (s.quantile(.75) - s.quantile(.25)))).sum()))
_cm = X[nums].corr().abs().values
_iu = np.triu_indices(len(nums), 1)
mac("nParesNum", str(len(_iu[0])))
mac("nCorrAlta", str(int((_cm[_iu] > 0.8).sum())))
mac("rAprobSem", f"{X['Curricular units 1st sem (approved)'].corr(X['Curricular units 2nd sem (approved)']):.2f}")
mac("rInscSem", f"{X['Curricular units 1st sem (enrolled)'].corr(X['Curricular units 2nd sem (enrolled)']):.2f}")
mac("rCredSem", f"{X['Curricular units 1st sem (credited)'].corr(X['Curricular units 2nd sem (credited)']):.2f}")
mac("rAprobSemMax", f"{_cm[_iu].max():.2f}")
mac("nOutEdad", str(out_age(X["Age at enrollment"])))
mac("pOutEdad", f"{out_age(X['Age at enrollment']) / Ntr * 100:.1f}")
mac("edadMin", str(int(X["Age at enrollment"].min())))
mac("edadMax", str(int(X["Age at enrollment"].max())))
print("[chequeo EDA]", dict(NF=(round(t_nf, 1), n_nf), DB=(round(t_db, 1), n_db), Bec=round(t_bec, 1), Jov=round(t_j, 1),
                            May=round(t_m, 1), Aprob0=(round(t_a0, 1), n_a0), SinUC=int(cero.sum()),
                            cursoSinUC=sorted(X.loc[cero, "Course"].unique().tolist()),
                            curso=(round(gr['mean'].min() * 100, 1), round(gr['mean'].max() * 100, 1), len(gr))))


# Comparaciones pareadas en Test contra la RL con regla (extraídas de la salida del notebook 04, celda 14 -> CSV)
par = pd.read_csv(TAB / "04_pareadas_vs_lr_regla.csv", index_col=0)
INV = {v: k for k, v in NOM.items()}
filas_p = []
for n, r in par.iterrows():
    a_, b_ = [int(x) for x in re.match(r"(\d+) / (\d+)", r["McNemar (solo él / solo LR)"]).groups()]
    filas_p.append(dict(k=INV[n], a=a_, b=b_, p=stats.binomtest(a_, a_ + b_, 0.5).pvalue, **{c: r[c] for c in ["F1", "Accuracy", "MCC", "ROC-AUC"]}))
par = pd.DataFrame(filas_p).sort_values("p").reset_index(drop=True)
m_ = len(par)
par["holm"] = np.minimum(1, np.maximum.accumulate((m_ - np.arange(m_)) * par["p"].values))
mac("nPareadas", str(m_))
mac("nMcnHolm", str(int((par["holm"] < 0.05).sum())))
for _, r in par.iterrows():
    mac(f"pmc{r['k']}", f"{r['p']:.3f}")
    mac(f"pho{r['k']}", f"{r['holm']:.3f}")
print("[chequeo] McNemar+Holm:", {r["k"]: (round(r["p"], 4), round(r["holm"], 4)) for _, r in par.iterrows()})
_au = [float(re.match(r"\s*([+-][0-9.]+)", v).group(1)) for k_, v in zip(par["k"], par["ROC-AUC"]) if k_ not in ("SVR", "VH")]
mac("auMin", f"{min(_au):+.3f}".replace("-", "$-$"))
mac("auMax", f"{max(_au):+.3f}")
mac("accLRD", f"{test.loc[NOM['LRD'], 'Accuracy']:.3f}")

# ------------------------------------------------------------------------------------------------------------------
# Escritura de numbers.tex
# ------------------------------------------------------------------------------------------------------------------
with open(OUT / "numbers.tex", "w", encoding="utf-8") as fh:
    fh.write("% Generado por scripts/make_tables.py — no editar a mano\n")
    for k, v in sorted(macros.items()):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"  ✓ numbers.tex ({len(macros)} macros)")


# ------------------------------------------------------------------------------------------------------------------
# Tablas
# ------------------------------------------------------------------------------------------------------------------
def tabla(nombre, contenido):
    (OUT / f"{nombre}.tex").write_text(contenido, encoding="utf-8")
    print("  ✓", nombre)


def negrita(valores, fmt, mayor=True, excluir=()):
    """Devuelve las cadenas formateadas con el mejor valor (por columna) en negritas."""
    vs = [v for i, v in enumerate(valores) if i not in excluir and not pd.isna(v)]
    mejor = (max if mayor else min)(vs)
    return [("\\textbf{" + fmt(v) + "}") if (i not in excluir and not pd.isna(v) and v == mejor) else fmt(v)
            for i, v in enumerate(valores)]


# --- Variables por grupo --------------------------------------------------------------------------------------------
tabla("tab_variables", r"""\begin{table}[t]
\caption{Variables predictoras del conjunto de datos (36) por grupo semántico. Tipo: N = numérica, C = nominal (códigos sin orden), B = binaria.}
\label{tab:variables}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcp{5.6cm}@{}}
\toprule
Grupo & $n$ & Variables (tipo) \\
\midrule
Demográficas & 7 & edad al inscribirse (N); género, extranjero, desplazado, necesidades educativas especiales (B); estado civil, nacionalidad (C) \\
Socioeconómicas & 7 & escolaridad y ocupación de la madre y del padre (C); becario, deudor, colegiatura al corriente (B) \\
Trayectoria al ingreso & 7 & modo de solicitud, curso (17 licenciaturas), escolaridad previa (C); orden de solicitud, calificación previa, nota de admisión (N); turno diurno/nocturno (B) \\
1.\textsuperscript{er} semestre & 6 & unidades curriculares (UC) acreditadas, inscritas, evaluadas, aprobadas, sin evaluación y promedio (N) \\
2.\textsuperscript{o} semestre & 6 & las mismas seis variables (N) \\
Macroeconómicas & 3 & desempleo, inflación y PIB del año de ingreso (N) \\
\midrule
Total & 36 & \textbackslash numNum{} N + \textbackslash numNom{} C + \textbackslash numBin{} B \\
\bottomrule
\end{tabular}
\end{table}
""".replace("\\textbackslash numNum{}", "\\nNum").replace("\\textbackslash numNom{}", "\\nNom").replace("\\textbackslash numBin{}", "\\nBin"))

# --- Espacios de búsqueda -------------------------------------------------------------------------------------------
tabla("tab_espacios", r"""\begin{table*}[t]
\caption{Hiperparámetros evaluados en cada búsqueda (todas con los mismos 5 folds estratificados, puntaje F1 macro y reajuste con el mejor).}
\label{tab:espacios}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{2.7cm}p{12.4cm}r@{}}
\toprule
Modelo & Hiperparámetros evaluados & Combinaciones \\
\midrule
Random Forest & \texttt{max\_features} \{sqrt, log2, 0.2, 0.35, 0.5\}; \texttt{min\_samples\_leaf} \{1, 2, 3, 5, 8, 12, 20\}; \texttt{max\_depth} \{None, 8, 12, 16\}; \texttt{class\_weight} \{None, balanced, balanced\_subsample\}; \texttt{criterion} \{gini, entropy\} & 40 (aleatoria) \\
Bagging y Pasting & \texttt{max\_samples} \{0.3, 0.5, 0.7, 1.0\} (Pasting: hasta 0.9); \texttt{max\_features} \{0.3, 0.6, 1.0\}; \texttt{max\_depth} \{4, 6, 8, 12, None\}; \texttt{min\_samples\_leaf} \{1, 5, 10, 20\}; \texttt{class\_weight} \{None, balanced\} del árbol & 30 + 30 (aleatoria) \\
AdaBoost & \texttt{n\_estimators} \{25, 50, 100, 200, 400\}; \texttt{learning\_rate} \{0.03, 0.1, 0.3, 1.0\}; \texttt{max\_depth} del árbol \{1, 2, 3, 4, 6, 8, 10\} & 140 (malla) \\
Gradient Boosting & \texttt{learning\_rate} \{0.02, 0.05, 0.1, 0.2\}; \texttt{n\_estimators} \{100, 200, 300\}; \texttt{max\_depth} \{2, 3, 4, 5\}; \texttt{subsample} \{0.5, 0.7, 1.0\}; \texttt{min\_samples\_leaf} \{1, 5, 10, 20\}; \texttt{max\_features} \{None, sqrt, 0.3\} & 30 (aleatoria) \\
HistGradientBoosting & \texttt{learning\_rate} \{0.02, 0.05, 0.1, 0.2\}; \texttt{max\_iter} \{100, 200, 400\}; \texttt{max\_leaf\_nodes} \{7, 15, 31\}; \texttt{min\_samples\_leaf} \{10, 20, 40, 80\}; \texttt{l2\_regularization} \{0, 1, 10\}; \texttt{max\_depth} \{None, 3, 5\}; \texttt{class\_weight} \{None, balanced\} & 40 (aleatoria) \\
EBM & \texttt{interactions} \{0, 10, 20\}; \texttt{learning\_rate} \{0.01, 0.03\}; \texttt{max\_leaves} \{2, 3\}; \texttt{min\_samples\_leaf} \{4, 20\} & 24 (malla) \\
Voting & pesos de (RL, SVM, RF, GB, HGB) $\in$ \{(1,1,1,1,1), (1,1,2,2,2), (2,1,1,1,1), (1,1,1,2,2), (1,0,1,1,1)\} & 5 (malla) \\
Stacking & $C$ del meta-modelo (regresión logística balanceada) $\in$ \{0.01, 0.1, 1, 10\}, con validación cruzada interna de 5 folds & 4 \\
RL y SVM & heredados de proyectos previos del curso (búsquedas sobre el mismo Train); aquí se usan fijos & --- \\
\bottomrule
\end{tabular}
\end{table*}
""")

# --- Modelos base (CV) ----------------------------------------------------------------------------------------------
cfg_base = {
    "DUM": "predice la clase mayoritaria",
    "LRD": "parámetros por defecto de scikit-learn",
    "SVLD": "ídem (\\texttt{LinearSVC})", "SVPD": "ídem (\\texttt{SVC}, grado 3)", "SVRD": "ídem (\\texttt{SVC}, RBF)",
    "DT": "profundidad 8, \\texttt{min\\_samples\\_leaf}=10, balanceado",
    "VHB": "voto mayoritario de la RL ($C$=0.3, L1) y la SVM RBF ($C$=100, $\\gamma$=0.001) ya optimizadas, y un Random Forest de 200 árboles",
    "VSB": "promedio de probabilidades de los mismos tres (SVM calibrada con Platt)",
    "BAGB": "200 árboles (el anterior) con remuestreo, \\texttt{max\\_samples}=0.5",
    "PASB": "200 árboles (el anterior) sin reemplazo, \\texttt{max\\_samples}=0.5",
    "RFB": "300 árboles, \\texttt{sqrt}, \\texttt{min\\_samples\\_leaf}=5, \\texttt{balanced\\_subsample}",
    "ADAB": "200 estimadores, $\\eta$=0.5, árboles de profundidad 2 (se detuvo en 10)",
    "GBB": "200 iteraciones, $\\eta$=0.1, profundidad 3, \\texttt{subsample}=0.8, pesos balanceados",
    "HGBB": "200 iteraciones, $\\eta$=0.1, 15 hojas, balanceado",
    "STKB": "los mismos tres $\\rightarrow$ RL balanceada (5 folds internos)",
    "EBMG": "modelo aditivo sin interacciones (\\texttt{interactions}=0)",
    "EBMB": "10 interacciones de a pares",
}
orden_base = ["DUM", "LRD", "SVLD", "SVPD", "SVRD", "DT", "VHB", "VSB", "BAGB", "PASB", "RFB", "ADAB", "GBB", "HGBB", "STKB",
              "EBMG", "EBMB"]
nom_base = {"DUM": "Dummy", "LRD": "Regresión logística", "SVLD": "SVM lineal", "SVPD": "SVM polinomial", "SVRD": "SVM RBF",
            "DT": "Árbol de decisión", "VHB": "Voting (hard)", "VSB": "Voting (soft)", "BAGB": "Bagging", "PASB": "Pasting",
            "RFB": "Random Forest", "ADAB": "AdaBoost", "GBB": "Gradient Boosting", "HGBB": "HistGradientBoosting",
            "STKB": "Stacking", "EBMG": "EBM (GAM)", "EBMB": "EBM"}
fv = [round(cvm(k), 3) for k in orden_base]
mejor_f1 = max(fv[1:])
filas = []
for k, v in zip(orden_base, fv):
    r = base_cv.loc[NOM[k]]
    f1s = f"{v:.3f}$\\pm${cvs(k):.3f}"
    f1s = f"\\textbf{{{f1s}}}" if v == mejor_f1 else f1s
    auc = "---" if pd.isna(r["ROC-AUC"]) or not re.match(r"[0-9]", str(r["ROC-AUC"])) else f"{float(str(r['ROC-AUC']).split('±')[0]):.3f}"
    filas.append(f"{nom_base[k]} & {cfg_base[k]} & {float(str(r['Accuracy']).split('±')[0]):.3f} & {f1s} & "
                 f"{float(str(r['MCC']).split('±')[0]):.3f} & {auc} & {float(r['Gap F1 (train−CV)']):.3f} \\\\")
cuerpo = "\n".join(filas)
tabla("tab_base", r"""\begin{table*}[t]
\caption{Modelos base evaluados con validación cruzada estratificada de 5 folds sobre Train (media; F1 macro con desviación entre folds). La brecha es el F1 de entrenamiento menos el F1 de validación. Negritas: mejor F1 macro (sin contar a Dummy).}
\label{tab:base}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{2.9cm}p{7.6cm}ccccc@{}}
\toprule
Modelo & Configuración inicial & Acc. & F1 macro & MCC & AUC & Brecha \\
\midrule
""" + cuerpo + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# --- Modelos optimizados --------------------------------------------------------------------------------------------
def hp_txt(k):
    t = {"RF": "RF", "BAG": "Bagging", "PAS": "Pasting", "ADA": "AdaBoost", "GB": "Gradient Boosting",
         "HGB": "HistGradientBoosting", "EBM": "EBM"}
    return hp[t[k]] if k in t else None


def fmt_num(v):
    return "None" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v).replace("_", "\\_"))


hps = {
    "LR": "penalización L1 (\\texttt{saga}), $C$=0.3, balanceada",
    "SVL": "$C$=0.01, balanceada",
    "SVP": "grado 2, \\texttt{coef0}=1, $C$=1, balanceada",
    "SVR": "$C$=100, $\\gamma$=0.001",
}
h = hp["Random Forest"]
hps["RF"] = (f"\\texttt{{min\\_samples\\_leaf}}={h['min_samples_leaf']}, \\texttt{{{h['max_features']}}}, profundidad {h['max_depth']}, "
             f"{h['criterion']}, balanceada")
for k, nm in [("BAG", "Bagging"), ("PAS", "Pasting")]:
    h = hp[nm]
    hps[k] = (f"\\texttt{{max\\_samples}}={h['max_samples']:g}, \\texttt{{max\\_features}}={h['max_features']:g}, árbol de profundidad "
              f"{h['estimator__max_depth']} y \\texttt{{min\\_samples\\_leaf}}={h['estimator__min_samples_leaf']}, balanceado")
h = hp["AdaBoost"]
hps["ADA"] = f"árboles de profundidad {h['estimator__max_depth']}, $\\eta$={h['learning_rate']:g}, {h['n_estimators']} estimadores"
h = hp["Gradient Boosting"]
hps["GB"] = (f"$\\eta$={h['learning_rate']:g}, {h['n_estimators']} iteraciones, profundidad {h['max_depth']}, "
             f"\\texttt{{subsample}}={h['subsample']:g}, \\texttt{{max\\_features}}={h['max_features']:g}, hoja mínima {h['min_samples_leaf']}")
h = hp["HistGradientBoosting"]
hps["HGB"] = (f"$\\eta$={h['learning_rate']:g}, {h['max_iter']} iteraciones, {h['max_leaf_nodes']} hojas, hoja mínima "
              f"{h['min_samples_leaf']}, $\\lambda_2$={h['l2_regularization']:g}, balanceado")
h = hp["EBM"]
hps["EBM"] = f"{h['interactions']} interacciones, $\\eta$={h['learning_rate']:g}, {h['max_leaves']} hojas, hoja mínima {h['min_samples_leaf']}"
w = hp["Voting (hard)"]["weights"]
assert hp["Voting (soft)"]["weights"] == w
hps["VH"] = hps["VS"] = f"pesos (RL, SVM, RF, GB, HGB) = ({', '.join(str(x) for x in w)})"
hps["STK"] = "5 modelos base (RL, SVM, RF, GB, HGB) $\\rightarrow$ RL balanceada"
orden_opt = ["LR", "SVL", "SVP", "SVR", "RF", "BAG", "PAS", "ADA", "GB", "HGB", "EBM", "VH", "VS", "STK"]
cvb = {k: cvm(pares_opt[k]) for k in orden_opt}
cvo = {k: cvm(k) for k in orden_opt}
filas = []
for k in orden_opt:
    filas.append(f"{CORTO[k].replace('Regresión logística (RL)', 'RL')} & {hps[k]} & {cvb[k]:.3f} & "
                 f"{cvo[k]:.3f}$\\pm${cvs(k):.3f} & {delta_cv[k]:+.3f} & {test.loc[NOM[pares_opt[k]], 'F1']:.3f} & "
                 f"{test.loc[NOM[k], 'F1']:.3f} & {delta_te[k]:+.3f} \\\\".replace("-0.", "$-$0.").replace("+0.", "+0."))
tabla("tab_opt", r"""\begin{table*}[t]
\caption{Modelos optimizados: hiperparámetros seleccionados, F1 macro en validación cruzada (CV, 5 folds sobre Train) y en Test, frente a los mismos modelos con su configuración inicial (``base''; para RL y SVM, los parámetros por defecto). $\Delta$ = optimizado $-$ base.}
\label{tab:opt}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}p{2.6cm}p{6.5cm}cccccc@{}}
\toprule
 & & \multicolumn{3}{c}{F1 macro en CV} & \multicolumn{3}{c}{F1 macro en Test} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Modelo & Hiperparámetros seleccionados & Base & Optimizado & $\Delta$ & Base & Optimizado & $\Delta$ \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# --- Resultados en Test ---------------------------------------------------------------------------------------------
orden_test = ["DUM", "LR", "LRR", "SVL", "SVP", "SVR", "VH", "VS", "BAG", "PAS", "RF", "ADA", "GB", "HGB", "STK", "EBM"]
cols_t = ["Accuracy", "Precision", "Recall", "F1", "F1 ponderado", "MCC", "Kappa", "ROC-AUC"]
vals = {c: [test.loc[NOM[k], c] for k in orden_test] for c in cols_t}
fm = lambda v: "---" if pd.isna(v) else f"{v:.3f}"
txt = {c: negrita(vals[c], fm, excluir=(0,)) for c in cols_t if c != "F1"}
filas = []
for i, k in enumerate(orden_test):
    n = NOM[k]
    if n in ic.index:
        _, lo, hi = parse_ic(ic.loc[n, "F1"])
        ci = f"[{lo:.3f}, {hi:.3f}]"
    else:
        ci = ""
    f1v = f"{test.loc[n, 'F1']:.3f}"
    if i != 0 and test.loc[n, "F1"] == max(vals["F1"][1:]):
        f1v = f"\\textbf{{{f1v}}}"
    auc = txt["ROC-AUC"][i]
    if k in ("SVL", "SVP", "SVR"):
        auc += "\\textsuperscript{a}"
    filas.append(f"{CORTO[k]} & {txt['Accuracy'][i]} & {txt['Precision'][i]} & {txt['Recall'][i]} & {f1v} {ci} & "
                 f"{txt['F1 ponderado'][i]} & {txt['MCC'][i]} & {txt['Kappa'][i]} & {auc} \\\\")
tabla("tab_test", r"""\begin{table*}[t]
\caption{Evaluación final en Test ($n=885$, evaluada una sola vez por modelo). Precisión, exhaustividad y F1 son promedios macro; entre corchetes, el intervalo de confianza de 95\,\% por bootstrap (2{,}000 remuestreos de los 885 estudiantes) del F1 macro, calculado para los modelos que se compararon formalmente. Negritas: mejor valor de la columna (sin contar a Dummy).}
\label{tab:test}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
Modelo & Acc. & Prec. & Exh. & F1 macro [IC 95\,\%] & F1 pond. & MCC & $\kappa$ & AUC \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\raggedright\scriptsize\textsuperscript{a}\,AUC calculada con \texttt{decision\_function} de scikit-learn; con probabilidades (\texttt{libsvm}) la SVM RBF obtiene 0.884, igual que la RL. Voting (hard) no produce puntajes. Sin IC: modelos que no entraron al bootstrap.
\end{table*}
""")

# --- Por clase en Test ----------------------------------------------------------------------------------------------
orden_cl = ["LR", "LRR", "SVR", "RF", "VS", "EBM"]   # los modelos con tabla de la clase Dropout (los 13 del bootstrap)
filas = []
for k in orden_cl:
    n = NOM[k]
    filas.append(f"{CORTO[k]} & {f1c.loc[n, 'Dropout']:.3f} & {f1c.loc[n, 'Enrolled']:.3f} & {f1c.loc[n, 'Graduate']:.3f} & "
                 f"{rec.loc[n, 'Enrolled']:.3f} & {int(det.loc[n, 'Detectados (de 284)'])} \\\\")
tabla("tab_clases", r"""\begin{table}[t]
\caption{Desempeño por clase en Test. D = Dropout, E = Enrolled, G = Graduate. F1 por clase, exhaustividad (Exh.) de E y desertores detectados (Det., de 284).}
\label{tab:clases}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.55cm}ccccc@{}}
\toprule
Modelo & F1 D & F1 E & F1 G & Exh. E & Det. \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# --- Estado del arte ------------------------------------------------------------------------------------------------
def T(k, c="F1"):
    return f"{test.loc[NOM[k], c]:.3f}"


sota = [
    ("Praptomo et al.\\ (2026) \\cite{praptomo2026}", "RL / RF / XGBoost (búsqueda aleatoria, 15 conf.)",
     "UCI, 4,424; 80/20 estratificada (3,539/885), semilla 42; sin remuestreo",
     "F1 macro", "0.683 / 0.698 / 0.701 (exactitud 0.768 / 0.772 / 0.761)"),
    ("Nashier et al.\\ (2026) \\cite{nashier2026}", "RL, RF, SVM RBF, XGBoost, LightGBM, CatBoost, apilado de los tres últimos",
     "UCI, 4,424; 80/20 estratificada (3,539/885), semilla 42; malla con CV de 5 folds; sin remuestreo",
     "F1 macro (Test)", "0.703 / 0.704 / 0.693 / 0.718 / 0.703 / 0.690 / 0.697 (IC 95\\,\\% del apilado: 0.663--0.729)"),
    ("Villar y de Andrade (2024) \\cite{villar2024}", "SVM, RF, GB, XGBoost, CatBoost, LightGBM",
     "UCI, 4,424; SMOTE; partición y folds no reportados",
     "F1 por clase (media de las tres, calculada por nosotros)", "0.743 / 0.840 / 0.813 / 0.857 / 0.850 / 0.853"),
    ("Putra et al.\\ (2025) \\cite{putra2025}", "RF / XGBoost",
     "4,424 (tras limpieza); 80/20 sin semilla ni estratificación reportadas; selección de 25 variables por correlación",
     "Exactitud; sensibilidad", "0.806 / 0.790; 0.724 / 0.717 (F1 no reportado)"),
    ("Wijiyanto et al.\\ (2026) \\cite{wijiyanto2026}", "Apilado HLR-STACK (XGBoost + CatBoost + LightGBM $\\rightarrow$ RL)",
     "4,424; 80/20 (tamaño y estratificación no reportados), CV de 7 folds, SMOTE en los folds, RFE",
     "Exactitud; F1 (promedio no especificado); AUC", "0.844; 0.725; 0.915"),
    ("\\emph{Contexto:} Martins et al.\\ (2023) \\cite{martins2023}", "SVMSMOTE + RF / Balanced RF",
     "Versión previa (4,433); 90/10, 10 repeticiones; fin del 1.\\textsuperscript{er} semestre",
     "F1 global (ponderado)", "0.745 / 0.744"),
    ("\\textbf{Este trabajo}", "RL con su regla / SVM lineal / Random Forest / Voting (soft)",
     "UCI, 4,424; 80/20 estratificada (3,539/885), semilla 42; CV de 5 folds; Pipeline sin fuga",
     "F1 macro (Test)",
     f"{T('LRR')} / {T('SVL')} / {T('RF')} / {T('VS')} (exactitud {T('LRR', 'Accuracy')} / {T('SVL', 'Accuracy')} / "
     f"{T('RF', 'Accuracy')} / {T('VS', 'Accuracy')})"),
]
filas = [" & ".join(r) + " \\\\" for r in sota]
tabla("tab_sota", r"""\begin{table*}[t]
\caption{Trabajos que usan el mismo conjunto de datos (tres clases). Las condiciones experimentales difieren, por lo que las cifras no son directamente comparables (Sec.~\ref{sec:sota}). Los resultados de cada estudio se listan en el orden de sus modelos.}
\label{tab:sota}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}p{2.55cm}p{3.1cm}p{4.3cm}p{2.2cm}p{4.3cm}@{}}
\toprule
Autores (año) & Modelos & Datos y partición & Métrica & Resultado \\
\midrule
""" + "\n\\addlinespace[2pt]\n".join(filas) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# --- Comparaciones pareadas en Test ---------------------------------------------------------------------------------
def dlt(celda):
    """'+0.0373 [+0.014, +0.061]' -> '+0.037 [+0.014, +0.061]' (con signos tipográficos)."""
    if str(celda).strip() in ("—", "nan"):
        return "---"
    m = re.match(r"\s*([+-][0-9.]+)\s*\[\s*([+-][0-9.]+),\s*([+-][0-9.]+)\s*\]", str(celda))
    v, lo, hi = m.groups()
    sg = lambda x: x.replace("-", "$-$")
    return f"{sg(f'{float(v):+.3f}')} [{sg(lo)}, {sg(hi)}]"


filas = []
orden_p = sorted(par.index, key=lambda i: -float(re.match(r"\s*([+-][0-9.]+)", par.loc[i, "F1"]).group(1)))
for i in orden_p:
    r = par.loc[i]
    auc = dlt(r["ROC-AUC"]) + ("\\textsuperscript{a}" if r["k"] == "SVR" else "")
    hol = f"{r['holm']:.3f}"
    hol = f"\\textbf{{{hol}}}" if r["holm"] < 0.05 else hol
    filas.append(f"{CORTO[r['k']]} & {dlt(r['F1'])} & {dlt(r['Accuracy'])} & {dlt(r['MCC'])} & {auc} & {r['a']} / {r['b']} & {r['p']:.3f} & {hol} \\\\")
tabla("tab_pareadas", r"""\begin{table*}[t]
\caption{Diferencias pareadas en Test frente a la RL con su regla de decisión (modelo final del proyecto previo): diferencia observada e intervalo de 95\,\% por bootstrap pareado (2,000 remuestreos; sin corrección por multiplicidad). McNemar: estudiantes que acierta solo el modelo / solo la RL, $p$ exacto y $p$ ajustado por Holm sobre """ + str(m_) + r""" comparaciones (negritas: $<0.05$).}
\label{tab:pareadas}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Modelo & $\Delta$F1 macro & $\Delta$ exactitud & $\Delta$MCC & $\Delta$AUC & Solo él / solo RL & $p$ McNemar & $p$ Holm \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\multicolumn{8}{@{}l}{\textsuperscript{a}\,AUC de la SVM calculada con \texttt{decision\_function}; con probabilidades no difiere de la RL (0.884 en ambas). El Voting (hard) no produce puntajes.}
\end{tabular}
\end{table*}
""")

print("Listo.")
