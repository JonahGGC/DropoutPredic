#!/usr/bin/env python3
"""Verificaciones del reporte: contrasta con los CSV las cifras escritas a mano en main.tex, cuenta palabras del
abstract, páginas de contenido principal, citas/referencias y referencias cruzadas. Uso: python scripts/verify.py"""
import json, re, subprocess
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1]
T = R / "outputs" / "tablas"
rd = lambda n: pd.read_csv(T / n, index_col=0)
ok = lambda c, m: print(("  OK   " if c else "  FALLA ") + m) or c
res = []
tex = (R / "main.tex").read_text(encoding="utf-8")
num = (R / "tables" / "numbers.tex").read_text(encoding="utf-8")
mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", num))

print("1) Cifras escritas a mano en la prosa")
f1c, test, ic = rd("04_test_f1_por_clase.csv"), rd("04_test_todos_los_modelos.csv"), rd("04_test_intervalos_bootstrap.csv")
m6 = ["Regresión Logística (F_1)", "Regresión Logística (F_1) + P(Dropout) ≥ 0.40", "SVM RBF optimizado", "Random Forest optimizado",
      "Voting (soft) optimizado", "EBM optimizado"]
f = f1c.loc[m6]
res.append(ok(round(f["Graduate"].min(), 2) == 0.82 and round(f["Graduate"].max(), 2) == 0.86, f"F1 Graduate 0.82–0.86 ({f['Graduate'].min():.3f}–{f['Graduate'].max():.3f})"))
res.append(ok(round(f["Dropout"].min(), 2) == 0.75 and round(f["Dropout"].max(), 2) == 0.77, f"F1 Dropout 0.75–0.77 ({f['Dropout'].min():.3f}–{f['Dropout'].max():.3f})"))
res.append(ok(round(f["Enrolled"].min(), 2) == 0.45 and round(f["Enrolled"].max(), 2) == 0.53, f"F1 Enrolled 0.45–0.53 ({f['Enrolled'].min():.3f}–{f['Enrolled'].max():.3f})"))
tr = pd.read_csv(R / "outputs/data/train.csv", index_col=0); te = pd.read_csv(R / "outputs/data/test.csv", index_col=0)
cnt = pd.concat([tr["y"], te["y"]]).value_counts()
res.append(ok(round(cnt.max() / cnt.min(), 1) == 2.8, f"razón de clases 2.8 a 1 ({cnt.max() / cnt.min():.3f})"))
res.append(ok(len(tr) == 3539 and len(te) == 885 and len(tr) + len(te) == 4424, f"Train {len(tr)}, Test {len(te)}"))
opt = rd("03_modelos_optimizados_cv.csv"); ref = rd("03_referencias_cv.csv")
cv14 = pd.concat([opt, ref])
n14 = ["Regresión Logística (F_1)", "SVM Lineal optimizado", "SVM Polinomial optimizado", "SVM RBF optimizado", "Random Forest optimizado", "Bagging optimizado",
       "Pasting optimizado", "AdaBoost optimizado", "Gradient Boosting optimizado", "HistGradientBoosting optimizado", "EBM optimizado",
       "Voting (hard) optimizado", "Voting (soft) optimizado", "Stacking optimizado"]
cv14 = cv14[~cv14.index.duplicated(keep="first")].loc[n14]
sd = [float(str(v).split("±")[1]) for v in cv14["F1"].values]
res.append(ok(round(min(sd), 3) == 0.011 and round(max(sd), 3) == 0.022, f"desv. entre folds 0.011–0.022 ({min(sd):.4f}–{max(sd):.4f})"))
span = float(mac["cvFMaxOpt"]) - float(mac["cvFMinOpt"])
res.append(ok(abs(span - 0.027) < 0.0011, f"rango de F1 en CV de los optimizados ≈0.027 ({span:.4f})"))
b13 = [tuple(float(x) for x in re.match(r"\s*([0-9.]+)\s*\[\s*([0-9.]+),\s*([0-9.]+)", v).groups()) for v in ic["F1"]]
w = [hi - lo for _, lo, hi in b13]; fs = [v for v, _, _ in b13]
res.append(ok(len(b13) == 13, "13 modelos con IC bootstrap"))
res.append(ok(0.055 < np.mean(w) < 0.07, f"ancho medio del IC ≈0.06 ({np.mean(w):.3f}; {min(w):.3f}–{max(w):.3f})"))
res.append(ok(max(fs) - min(fs) < min(w), f"mejor−peor ({max(fs) - min(fs):.3f}) < IC más angosto ({min(w):.3f})"))
tab5 = ["Dummy (referencia)", "Regresión Logística (F_1)", "Regresión Logística (F_1) + P(Dropout) ≥ 0.40", "SVM Lineal optimizado", "SVM Polinomial optimizado",
        "SVM RBF optimizado", "Voting (hard) optimizado", "Voting (soft) optimizado", "Bagging optimizado", "Pasting optimizado", "Random Forest optimizado",
        "AdaBoost optimizado", "Gradient Boosting optimizado", "HistGradientBoosting optimizado", "Stacking optimizado", "EBM optimizado"]
a_lrd = test.loc["Regresión Logística base (por defecto)", "Accuracy"]
res.append(ok(a_lrd > test.loc[tab5, "Accuracy"].max(), f"RL por defecto ({a_lrd:.4f}) > exactitud de todos los modelos de la Tabla V (máx {test.loc[tab5, 'Accuracy'].max():.4f})"))
res.append(ok(abs(a_lrd - 0.771) < 5e-4 and abs(test.loc['Regresión Logística base (por defecto)', 'Test − CV (F1)']) > 0, "exactitud RL por defecto 0.771"))
best_acc = test.loc[tab5[1:], "Accuracy"].idxmax(); best_mcc = test.loc[tab5[1:], "MCC"].idxmax(); best_auc = test.loc[tab5[1:], "ROC-AUC"].idxmax()
res.append(ok((best_acc, best_mcc, best_auc) == ("SVM Lineal optimizado", "SVM Lineal optimizado", "Voting (soft) optimizado"), f"mejor exactitud/MCC: SVM lineal; mejor AUC: Voting soft ({best_acc}, {best_mcc}, {best_auc})"))
best_f1 = test.loc[tab5[1:], "F1"].idxmax()
res.append(ok(best_f1 == "Random Forest optimizado", f"mejor F1 macro en Test: {best_f1}"))
top3 = test.loc[["Random Forest optimizado", "Bagging optimizado", "Pasting optimizado"], "F1"]
res.append(ok(abs((top3.max() - top3.min()) - float(mac["rangoTresTest"])) < 6e-4, f"RF/Bagging/Pasting difieren en {top3.max() - top3.min():.4f}"))
# 4 modelos con McNemar-Holm < 0.05 y su identidad
par = pd.read_csv(T / "04_pareadas_vs_lr_regla.csv", index_col=0)
ps = []
for n, r in par.iterrows():
    a, b = [int(x) for x in re.match(r"(\d+) / (\d+)", r["McNemar (solo él / solo LR)"]).groups()]
    ps.append((n, stats.binomtest(a, a + b, .5).pvalue))
ps = sorted(ps, key=lambda x: x[1]); mm = len(ps)
ho = np.minimum(1, np.maximum.accumulate([(mm - i) * p for i, (_, p) in enumerate(ps)]))
sig = {n for (n, _), h in zip(ps, ho) if h < 0.05}
res.append(ok(sig == {"SVM RBF optimizado", "Voting (soft) optimizado", "Bagging optimizado", "Random Forest optimizado"} and mac["nMcnHolm"] == "4", f"McNemar+Holm significativos: {sorted(sig)}"))
res.append(ok(mac["nPareadas"] == "11", "11 comparaciones pareadas"))
# solo RF y Bagging tienen IC de F1 que excluye 0
excl = [n for n, r in par.iterrows() if not re.search(r"\[-", r["F1"]) and not re.search(r"\[\+0\.0+,", "")]
exF1 = [n for n, r in par.iterrows() if (lambda lo, hi: lo > 0 or hi < 0)(*[float(x) for x in re.findall(r"[+-][0-9.]+", r["F1"])[1:3]])]
res.append(ok(set(exF1) == {"Random Forest optimizado", "Bagging optimizado"}, f"IC de ΔF1 que excluyen 0: {exF1}"))
au = [(n, [float(x) for x in re.findall(r"[+-][0-9.]+", r["ROC-AUC"])]) for n, r in par.iterrows() if re.search(r"[0-9]", r["ROC-AUC"]) and n != "SVM RBF optimizado"]
res.append(ok(all(lo <= 0 <= hi for _, (v, lo, hi) in au), "AUC pareadas (sin SVM): todos los IC incluyen 0"))
# AUC SVM con probabilidades (salida del notebook 04)
nb = (R.parent / "res/ejecutados/04_resultados.ipynb")
if nb.exists():
    t = nb.read_text(encoding="utf-8")
    res.append(ok("0.8839" in t and "0.8838" in t, "AUC SVM con probabilidades 0.8839 y RL 0.8838 en el notebook 04"))
# Repetida: nada sobrevive Holm; 2 crudos
rv = rd("04_cv_repetida_vs_lr_f1.csv")
res.append(ok(int((rv["p Holm"] < .05).sum()) == 0 and int((rv["p-valor"] < .05).sum()) == 2 and len(rv) == 11, "CV repetida: 11 comparaciones, 0 con Holm<0.05, 2 con p crudo<0.05"))
# Test base/optimizado
res.append(ok(mac["nOptMejoraTest"] == "6" and mac["nOptMejoraCV"] == "10", "10/10 mejoran en CV; 6/10 en Test"))
# Sota: números del propio trabajo en la tabla sota
sota = (R / "tables" / "tab_sota.tex").read_text(encoding="utf-8")
res.append(ok(all(f"{test.loc[k, 'F1']:.3f}" in sota for k in ["Random Forest optimizado", "Voting (soft) optimizado", "SVM Lineal optimizado"]), "tab_sota contiene nuestros F1 de Test"))
# 148 sin UC
res.append(ok(mac["nSinUC"] == "148" and mac["cursoSinUC"] == "171", "148 registros sin UC, curso 171"))

print("2) Abstract")
txt = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(R / "main.pdf"), "-"], capture_output=True, text=True).stdout
a = re.sub(r"-\n", "", txt[txt.index("Resumen"):txt.index("Palabras clave")]).replace("Resumen—", "")
res.append(ok(150 <= len(a.split()) <= 250, f"abstract: {len(a.split())} palabras (150–250)"))

print("3) Páginas")
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(R / "main.pdf")], capture_output=True, text=True).stdout).group(1))
pag_ref = next(i for i in range(1, n + 1) if "EFERENCIAS" in subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(R / "main.pdf"), "-"], capture_output=True, text=True).stdout.replace(" ", ""))
res.append(ok(True, f"PDF de {n} páginas; las referencias empiezan en la {pag_ref} (contenido principal ≈ {pag_ref - 1} páginas + parte de la {pag_ref})"))
res.append(ok(8 <= pag_ref <= 13, "contenido principal dentro de 8–12 páginas"))

print("4) Citas y referencias")
bib = set(re.findall(r"@\w+\{(\w+),", (R / "refs.bib").read_text(encoding="utf-8")))
cit = set(k.strip() for g in re.findall(r"\\cite\{([^}]*)\}", tex + "".join(p.read_text(encoding="utf-8") for p in (R / "tables").glob("tab_*.tex"))) for k in g.split(","))
res.append(ok(cit <= bib, f"todas las citas existen en refs.bib (faltan: {sorted(cit - bib)})"))
res.append(ok(bib <= cit, f"todas las entradas de refs.bib se citan (sin citar: {sorted(bib - cit)})"))
pdftxt = subprocess.run(["pdftotext", str(R / "main.pdf"), "-"], capture_output=True, text=True).stdout
res.append(ok("??" not in pdftxt, "sin referencias cruzadas rotas (??)"))
labs = set(re.findall(r"\\label\{([^}]*)\}", tex + "".join(p.read_text(encoding="utf-8") for p in (R / "tables").glob("tab_*.tex"))))
refs = set(re.findall(r"\\ref\{([^}]*)\}", tex + "".join(p.read_text(encoding="utf-8") for p in (R / "tables").glob("tab_*.tex"))))
res.append(ok(refs <= labs, f"todas las \\ref tienen \\label (faltan: {sorted(refs - labs)})"))
res.append(ok(labs <= refs | {"eq:f1", "sec:intro", "sec:conc", "sec:disc"}, f"etiquetas sin uso: {sorted(labs - refs)}"))
figs = set(re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex))
res.append(ok(all((R / "figures" / f).exists() for f in figs), f"{len(figs)} figuras existen"))
print(f"\n{sum(res)}/{len(res)} verificaciones OK")
raise SystemExit(0 if all(res) else 1)
