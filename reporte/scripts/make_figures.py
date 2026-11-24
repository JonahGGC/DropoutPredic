#!/usr/bin/env python3
"""Genera las figuras del reporte a partir de outputs/ (tablas CSV y partición guardada por los notebooks).

Uso:  python scripts/make_figures.py        (desde la carpeta del proyecto LaTeX)
Salida: figures/*.pdf (vectoriales)
"""
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
TAB = RAIZ / "outputs" / "tablas"
DAT = RAIZ / "outputs" / "data"
FIG = RAIZ / "figures"
FIG.mkdir(exist_ok=True)

# Paleta categórica validada (slots 1-3), texto en tinta neutra y rejilla discreta
AZUL, NARANJA, AQUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#7a7a76"
TINTA, TINTA2, REJILLA = "#1a1a19", "#52514e", "#dcdbd6"
COL = 3.5      # ancho de una columna IEEE (pulgadas)
ANCHO = 7.16   # ancho de página completa

mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "text.color": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.linewidth": 0.6, "grid.color": REJILLA,
    "grid.linewidth": 0.5, "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})

CLASES = ["Dropout", "Enrolled", "Graduate"]
COLORES_CLASE = {"Dropout": NARANJA, "Enrolled": AQUA, "Graduate": AZUL}

# Nombres cortos de los modelos (el CSV usa los nombres completos de los notebooks)
CORTO = {
    "Regresión Logística (F_1)": "Reg. logística (RL)",
    "Regresión Logística (F_1) + P(Dropout) ≥ 0.40": "RL + regla $P(\\mathrm{Dropout})\\geq0.40$",
    "SVM RBF optimizado": "SVM RBF",
    "Voting (soft) optimizado": "Voting (soft)",
    "Voting (hard) optimizado": "Voting (hard)",
    "HistGradientBoosting optimizado": "HistGradientBoosting",
    "Stacking optimizado": "Stacking",
    "Gradient Boosting optimizado": "Gradient Boosting",
    "EBM optimizado": "EBM",
    "Random Forest optimizado": "Random Forest",
    "Pasting optimizado": "Pasting",
    "AdaBoost optimizado": "AdaBoost",
    "Bagging optimizado": "Bagging",
}


def corto(n):
    return CORTO.get(n, n.replace(" optimizado", ""))


def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.pdf")
    plt.close(fig)
    print("  ✓", nombre)


def leer(nombre):
    return pd.read_csv(TAB / nombre, index_col=0)


def parse_ic(celda):
    """'0.7161 [0.6835, 0.7468]' -> (0.7161, 0.6835, 0.7468)"""
    m = re.match(r"\s*([0-9.]+)\s*\[\s*([0-9.]+),\s*([0-9.]+)\s*\]", str(celda))
    return tuple(float(x) for x in m.groups()) if m else (np.nan,) * 3


# ---------------------------------------------------------------------------------------------------
# Fig. 2 — composición de las clases según edad y unidades aprobadas (solo Train)
# ---------------------------------------------------------------------------------------------------
def fig_eda():
    tr = pd.read_csv(DAT / "train.csv", index_col=0)
    y = tr["y"].map(dict(enumerate(CLASES)))
    edad = pd.cut(tr["Age at enrollment"], [16, 19, 22, 25, 30, 40, 100],
                  labels=["17-19", "20-22", "23-25", "26-30", "31-40", "41+"])
    a1 = tr["Curricular units 1st sem (approved)"].clip(upper=10)
    a2 = tr["Curricular units 2nd sem (approved)"].clip(upper=10)
    paneles = [(edad, "Edad al inscribirse", "(a) Edad"),
               (a1, "UC aprobadas, 1er semestre (10 = 10 o más)", "(b) 1er semestre"),
               (a2, "UC aprobadas, 2º semestre (10 = 10 o más)", "(c) 2º semestre")]
    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, 2.0), sharey=True)
    for ax, (g, xl, tit) in zip(axs, paneles):
        comp = pd.crosstab(g, y, normalize="index")[CLASES]
        base = np.zeros(len(comp))
        x = np.arange(len(comp))
        for c in CLASES:                                   # apiladas con 1 px de separación
            ax.bar(x, comp[c].values, bottom=base, color=COLORES_CLASE[c], width=0.82, label=c,
                   edgecolor="white", linewidth=0.5)
            base += comp[c].values
        ax.set_xticks(x, [str(i) for i in comp.index])
        ax.set_xlabel(xl)
        ax.set_title(tit, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("Proporción de estudiantes")
    axs[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(1.75, 1.32), handlelength=1.0,
                  columnspacing=1.2)
    guardar(fig, "fig_eda_clases")


# ---------------------------------------------------------------------------------------------------
# Fig. 3 — efecto de la optimización (F1 macro en CV, base vs optimizado)
# ---------------------------------------------------------------------------------------------------
def fig_antes_despues():
    t = leer("03_optimizacion_base_vs_optimizado.csv").sort_values("F1 optimizado")
    ref = leer("03_referencias_cv.csv")["F1"].loc["Regresión Logística (F_1)"]
    ref = float(str(ref).split("±")[0])
    fig, ax = plt.subplots(figsize=(COL, 2.55))
    y = np.arange(len(t))
    ax.hlines(y, t["F1 base"], t["F1 optimizado"], color=GRIS, linewidth=1.0, zorder=1)
    ax.scatter(t["F1 base"], y, s=22, facecolor="white", edgecolor=GRIS, linewidth=1.0, zorder=3, label="Base")
    ax.scatter(t["F1 optimizado"], y, s=22, color=AZUL, zorder=3, label="Optimizado")
    ax.axvline(ref, color=NARANJA, linestyle="--", linewidth=1.0, label="RL (0.717)")
    ax.set_yticks(y, [corto(n) for n in t.index])
    ax.set_xlabel("F1 macro en validación cruzada (5 folds)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="none")
    guardar(fig, "fig_antes_despues")


# ---------------------------------------------------------------------------------------------------
# Fig. 4 — F1 macro en Test con IC 95% bootstrap
# ---------------------------------------------------------------------------------------------------
def fig_test_ic():
    t = leer("04_test_intervalos_bootstrap.csv")
    f = pd.DataFrame([parse_ic(v) for v in t["F1"]], index=t.index, columns=["f1", "lo", "hi"]).sort_values("f1")
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    y = np.arange(len(f))
    for yi, (n, r) in zip(y, f.iterrows()):
        es_lr = n.startswith("Regresión Logística")
        c = NARANJA if es_lr else AZUL
        ax.hlines(yi, r.lo, r.hi, color=c, linewidth=1.2, zorder=2)
        ax.scatter(r.f1, yi, s=20, color=c, zorder=3, marker="D" if es_lr else "o")
    ax.set_yticks(y, [corto(n) for n in f.index])
    ax.axvline(f.loc["Regresión Logística (F_1) + P(Dropout) ≥ 0.40", "f1"], color=NARANJA, linestyle="--",
               linewidth=0.8, zorder=1)
    ax.set_xlabel("F1 macro en Test (punto: valor observado; barra: IC 95% bootstrap)")
    ax.grid(axis="y", visible=False)
    guardar(fig, "fig_test_ic")


# ---------------------------------------------------------------------------------------------------
# Fig. 5 — diferencia de F1 macro contra la RL en la CV repetida 5×5 (IC de Nadeau–Bengio, p de Holm)
# ---------------------------------------------------------------------------------------------------
def fig_cv_repetida():
    t = leer("04_cv_repetida_vs_lr_f1.csv").sort_values("Diferencia F1 vs LR")
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    y = np.arange(len(t))
    ax.axvline(0, color=TINTA2, linewidth=0.8, zorder=1)
    for yi, (n, r) in zip(y, t.iterrows()):
        ax.hlines(yi, r["IC 95% inferior"], r["IC 95% superior"], color=AZUL, linewidth=1.2, zorder=2)
        ax.scatter(r["Diferencia F1 vs LR"], yi, s=20, color=AZUL, zorder=3)
        ax.text(0.0425, yi, f"{r['p Holm']:.2f}", va="center", ha="right", fontsize=6.8, color=TINTA2)
    ax.text(0.0425, len(t) - 0.2, "$p$ Holm", ha="right", va="bottom", fontsize=6.8, color=TINTA2)
    ax.set_yticks(y, [corto(n) for n in t.index])
    ax.set_xlim(-0.03, 0.045)
    ax.set_xlabel("$\\Delta$F1 macro frente a la RL (CV 5×5; barra: IC 95% corregido)")
    ax.grid(axis="y", visible=False)
    guardar(fig, "fig_cv_repetida")


# ---------------------------------------------------------------------------------------------------
# Fig. 6 — F1 macro en Train, CV y Test
# ---------------------------------------------------------------------------------------------------
def fig_generalizacion():
    t = leer("04_generalizacion.csv").sort_values("F1 Test")
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    y = np.arange(len(t))
    ax.hlines(y, t["F1 Test"], t["F1 train"], color=REJILLA, linewidth=1.0, zorder=1)
    ax.scatter(t["F1 train"], y, s=18, marker="s", color=GRIS, zorder=3, label="Train")
    ax.scatter(t["F1 CV (5 folds)"], y, s=20, marker="o", color=AZUL, zorder=3, label="CV (5 folds)")
    ax.scatter(t["F1 Test"], y, s=24, marker="D", color=NARANJA, zorder=3, label="Test")
    ax.set_yticks(y, [corto(n) for n in t.index])
    ax.set_xlabel("F1 macro")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.45, 1.0), handletextpad=0.3,
              columnspacing=1.2)                      # fuera del área de datos para no tapar puntos
    guardar(fig, "fig_generalizacion")


# ---------------------------------------------------------------------------------------------------
# Fig. 7 — matrices de confusión en Test (conteos transcritos de las matrices de 04_resultados.ipynb)
# ---------------------------------------------------------------------------------------------------
MATRICES = {
    "RL + regla $P(\\mathrm{Dropout})\\geq0.40$": [[203, 56, 25], [35, 91, 33], [15, 79, 348]],
    "Voting (soft)": [[203, 51, 30], [36, 85, 38], [15, 46, 381]],
    "Random Forest": [[201, 54, 29], [25, 97, 37], [13, 55, 374]],
}


def fig_confusion():
    test = leer("04_test_todos_los_modelos.csv")
    esperado = {"RL + regla $P(\\mathrm{Dropout})\\geq0.40$": "Regresión Logística (F_1) + P(Dropout) ≥ 0.40",
                "Voting (soft)": "Voting (soft) optimizado", "Random Forest": "Random Forest optimizado"}
    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, 2.35))
    for ax, (tit, m) in zip(axs, MATRICES.items()):
        m = np.array(m)
        assert m.sum(axis=1).tolist() == [284, 159, 442], tit                      # soporte de cada clase en Test
        acc = np.trace(m) / m.sum()
        assert abs(acc - test.loc[esperado[tit], "Accuracy"]) < 5e-4, (tit, acc)    # coincide con la tabla de Test
        pct = m / m.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{m[i, j]}\n({pct[i, j] * 100:.0f}%)", ha="center", va="center", fontsize=6.8,
                        color="white" if pct[i, j] > 0.55 else TINTA)
        ax.set_xticks(range(3), CLASES)
        ax.set_yticks(range(3), CLASES if ax is axs[0] else [])   # etiquetas de fila solo en el primer panel
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Clase predicha")
        ax.set_title(tit, fontsize=7.6)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Clase real")
    guardar(fig, "fig_confusion")


# ---------------------------------------------------------------------------------------------------
# Fig. 8 — importancia por permutación en Test (RF y Voting soft)
# ---------------------------------------------------------------------------------------------------
ES = {
    "Curricular units 2nd sem (approved)": "UC aprobadas, 2º sem.",
    "Curricular units 1st sem (approved)": "UC aprobadas, 1er sem.",
    "Tuition fees up to date": "Colegiatura al corriente",
    "Curricular units 2nd sem (grade)": "Promedio, 2º sem.",
    "Course": "Curso",
    "Curricular units 2nd sem (evaluations)": "UC evaluadas, 2º sem.",
    "Scholarship holder": "Becario",
    "Curricular units 1st sem (enrolled)": "UC inscritas, 1er sem.",
}


def fig_importancia():
    t = leer("04_importancia_variables.csv")
    rf, vs = "Permutación RF (ΔF1)", "Permutación Voting (soft) (ΔF1)"
    top = t.sort_values(rf, ascending=False).head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(top))
    h = 0.36
    ax.barh(y + h / 2, top[rf], height=h, color=AZUL, label="Random Forest")
    ax.barh(y - h / 2, top[vs], height=h, color=NARANJA, label="Voting (soft)")
    ax.set_yticks(y, [ES.get(i, i) for i in top.index])
    ax.set_xlabel("Caída del F1 macro en Test al permutar la variable")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    guardar(fig, "fig_importancia")


if __name__ == "__main__":
    print("Generando figuras en", FIG)
    fig_eda()
    fig_antes_despues()
    fig_test_ic()
    fig_cv_repetida()
    fig_generalizacion()
    fig_confusion()
    fig_importancia()
