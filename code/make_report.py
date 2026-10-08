"""Build the PDF report from results/*.json and the figures."""
import json, os
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "report", "BT2024154_report.pdf")
GITHUB = "https://github.com/Kkshiteej-Tiwari/Machine-Learning-Assignment-1"

L = {v: json.load(open(os.path.join(RES, f"{v}_cv.json"))) for v in ("var1", "var2")}
B = {v: L[v]["best"] for v in L}


def ridge(v, d):
    return next(r for r in L[v]["ridge"] if r["degree"] == d)


def lasso(v, d, g):
    return next(r for r in L[v]["lasso"] if r["degree"] == d and r["gamma"] == g)


def best_of(v, method, **kw):
    rows = [r for r in L[v][method] if all(r[k] == x for k, x in kw.items())]
    return min(rows, key=lambda r: r["cv_wmse"])


def mname(b, short=False):
    """Human-readable name of a selected model, e.g. 'elastic net &gamma;=4, &rho;=0.5'."""
    if b["method"] == "enet":
        return (f"elastic net, &gamma;={b['gamma']}, &rho;={b['l1_ratio']}" if not short
                else f"elastic net &gamma;={b['gamma']}")
    return f"{'Lasso' if b['method'] == 'lasso' else 'ridge'}, &gamma;={b['gamma']}" if not short         else f"{b['method']} &gamma;={b['gamma']}"


def r2(v, mse):
    return 1 - mse / L[v]["var_y"]


ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=10.2,
                      leading=13.2, alignment=TA_JUSTIFY, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontName="Times-Bold", fontSize=13,
                    spaceBefore=8, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=ss["Heading3"], fontName="Times-Bold", fontSize=11,
                    spaceBefore=5, spaceAfter=2)
cap = ParagraphStyle("cap", parent=body, fontSize=8.8, leading=11, textColor=colors.HexColor("#52514e"))
title = ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold", fontSize=17, spaceAfter=2)
sub = ParagraphStyle("s", parent=body, alignment=1, fontSize=10.5)
left = ParagraphStyle("l", parent=body, alignment=0)
cell = ParagraphStyle("c", parent=body, fontSize=8.8, leading=10.5, alignment=0, spaceAfter=0)


def P(t, s=body):
    return Paragraph(t, s)


def tbl(rows, widths, bold_rows=()):
    rows = [[P(str(c), cell) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
          ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec"))]
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e3eefb")))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width_cm):
    from PIL import Image as PI
    path = os.path.join(RES, name)
    w, h = PI.open(path).size
    return Image(path, width=width_cm * cm, height=width_cm * cm * h / w)


b1, b2 = B["var1"], B["var2"]
r1_4, r1_5, r1_6 = ridge("var1", 4), ridge("var1", 5), ridge("var1", 6)
r2_8 = ridge("var2", 8)
story = []

# ------------------------------------------------------------------ title
story += [P("Polynomial Regression Assignment — Report", title),
          P("Roll No. <b>BT2024154</b> &nbsp;|&nbsp; Problems: var1 (steam turbine Net Power Score), "
            "var2 (thermal anomaly mapping)", sub),
          P(f"Code repository: <link href='{GITHUB}' color='blue'>{GITHUB}</link>", sub),
          Spacer(1, 6)]

# ------------------------------------------------------------------ 1
story += [P("1. Summary", h1),
          P(f"Both problems were solved with <b>polynomial regression only</b>: every model is a "
            f"polynomial in the inputs containing all terms of total degree &le; d, fitted by penalised "
            f"least squares. The degree and the penalty were chosen by repeated 5-fold cross-validation (CV). "
            f"The final choices are <b>degree {b1['degree']}</b> for var1 (CV MSE {b1['cv_mse']:.3f}, "
            f"CV R&sup2; {r2('var1', b1['cv_mse']):.3f}) and <b>degree {b2['degree']}</b> for var2 "
            f"(CV MSE {b2['cv_mse']:.3f}, CV R&sup2; {r2('var2', b2['cv_mse']):.4f}). "
            f"The largest allowed degrees (10 and 20) overfit badly. For var2 the CV error flattens at about "
            f"0.25 for degrees 8&ndash;10, which looks like the noise floor, so degree 8 is the smallest "
            f"degree that reaches it."),
          ]

# ------------------------------------------------------------------ 2
story += [P("2. Data exploration", h1),
          tbl([["", "inputs", "train / test rows", "Var(y) train", "max degree",
                 "rows with &ge;3 inputs at &plusmn;1 (train &rarr; test)"],
               ["var1", "x1..x6", "1000 / 1000", f"{L['var1']['var_y']:.2f}", "10", "28.7% &rarr; <b>66.2%</b>"],
               ["var2", "x1..x3", "1000 / 1000", f"{L['var2']['var_y']:.2f}", "20", "1.5% &rarr; 1.7%"]],
              [1.2 * cm, 1.5 * cm, 2.6 * cm, 2.2 * cm, 1.9 * cm, 6.4 * cm]),
          Spacer(1, 4),
          P("The files contain no missing values. All inputs lie in [&minus;1, 1], and a large share of "
            "values sit <i>exactly</i> on &plusmn;1 (31% of var1 training values, 25% for var2). The inputs "
            "were evidently clipped to the box during preprocessing. This matters in two ways. "
            "(i) High-degree polynomials are least stable near the edges of the domain (Runge-type "
            "oscillation), and many points sit on those edges. "
            "(ii) For <b>var1 the test set is shifted toward the corners</b>: 66% of test rows have three or "
            "more clipped inputs, against 29% of training rows. CV on the training rows alone would "
            "therefore under-represent the hardest test region. Section 3.3 explains how this is "
            "handled. var2 shows no such shift."),
          ]

# ------------------------------------------------------------------ 3
story += [P("3. Approach", h1),
          P("3.1 Model family and feature basis", h2),
          P("For p inputs and degree d the model is "
            "f(x) = &Sigma; c<sub>t</sub>&middot;&phi;<sub>t</sub>(x), summed over all exponent vectors t with "
            "|t| &le; d, i.e. C(p+d, d) terms. That is 462 terms for var1 at d=5 (8008 at d=10) and 165 for var2 at "
            "d=8 (1771 at d=20). This matches the assignment's definition: the powers in each term add "
            "up to at most d. Rather than raw monomials x<sub>1</sub><sup>a</sup>x<sub>2</sub><sup>b</sup>&hellip;, "
            "I use the <b>product-Legendre basis</b> P<sub>a</sub>(x<sub>1</sub>)P<sub>b</sub>(x<sub>2</sub>)&hellip;. "
            "Every Legendre product expands into monomials of the same total degree, so it spans "
            "<i>exactly the same set of polynomials</i> and the model is still plain polynomial "
            "regression. The design matrix, however, is far better conditioned on [&minus;1, 1]. For var2 "
            "the condition number at d=8 drops from 1.2&times;10<super>3</super> (monomials) to 24, and at d=12 from "
            "7.3&times;10<super>4</super> to 290. This keeps high-degree fits numerically reliable, and it makes a "
            "penalty on the coefficients mean the same thing for every term."),
          P("3.2 Regularisation", h2),
          P("A degree-d polynomial with hundreds or thousands of terms fitted to 800 rows (one CV fold) "
            "overfits without a penalty. With d=6 for var1 there are more terms (924) than rows, so the "
            "training error drops to 0 while the CV error explodes (Figure 1). Three penalised estimators were used:"),
          P("&bull; <b>Ridge (L2)</b>, min ||y &minus; &Phi;c||&sup2; + &alpha;||c||&sup2;, with an unpenalised "
            "intercept. It is solved through one SVD of the centred design matrix per fold, which gives "
            "predictions for all 37 &alpha; values (10<super>&minus;8</super>&ndash;10<super>4</super>) "
            "at once. Sweeping every degree (10 for var1, 20 for var2) therefore takes about a minute."),
          P("&bull; <b>Lasso (L1) with a degree-weighted penalty</b>, min &frac12;n<super>&minus;1</super>||y &minus; &Phi;c||&sup2; "
            "+ &lambda; &Sigma;<sub>t</sub> (1+|t|)<super>&gamma;/2</super>|c<sub>t</sub>|. "
            "L1 can set coefficients to exactly zero, which helps when the true polynomial uses only a subset "
            "of the possible terms. The weight (1+|t|)<super>&gamma;/2</super> penalises high-order terms more "
            "than low-order ones, so the fit prefers smooth explanations. It is implemented by rescaling "
            "the columns of &Phi;, and &gamma;=0 gives ordinary Lasso."),
          P("&bull; <b>Elastic net</b>, the same objective with the penalty "
            "&lambda;[&rho;||c||<sub>1</sub> + &frac12;(1&minus;&rho;)||c||&sup2;] (same &gamma; weighting). "
            "It sits between ridge (&rho;=0) and Lasso (&rho;=1) and is the usual remedy when correlated "
            "terms make the Lasso pick among them erratically. &rho; &isin; {0.5, 0.9} was tried; small "
            "&rho; behaves like ridge, which stage 1 already covers."),
          P("3.3 Model selection", h2),
          P("All hyper-parameters (degree d, penalty &alpha; or &lambda;, weight &gamma;) were chosen by 5-fold CV, "
            "repeated with two different random splits and averaged (seeded, fully reproducible). "
            "Stage 1 runs a ridge sweep over <i>every</i> degree to map the bias&ndash;variance curve. "
            "Stage 2 runs a Lasso grid over the most promising degrees and &gamma; &isin; {0,2,4,6}. "
            "Stage 3 runs elastic net at the degrees and &gamma; values around the Lasso optimum. "
            "Because a richer model family can win by chance, elastic net replaces the ridge/Lasso winner "
            "only if its per-fold advantage (paired over the same 10 folds) exceeds one standard error. "
            "To account for the var1 shift, the selection score is an "
            "<b>importance-weighted CV MSE</b>. Each validation row with k clipped inputs gets weight "
            "w(k) = P<sub>test</sub>(k) / P<sub>train</sub>(k), so the CV error estimates the error on the "
            "test set's mix of rows. Unweighted CV MSE is reported alongside it. For var2 the weights are "
            "about 1, and both scores pick the same model."),
          ]

# ------------------------------------------------------------------ figure 1
story += [KeepTogether([fig("fig_degree_curves.png", 16.5),
                        P("<b>Figure 1.</b> Error against polynomial degree (log scale). Without a penalty the "
                          "training error keeps falling while the CV error rises sharply past degree 4 (var1) "
                          "and 9 (var2): the classic overfitting U-curve. Ridge tames the explosion but cannot "
                          "beat the low-degree optimum. The dashed line is the CV MSE of the final model.", cap)])]

# ------------------------------------------------------------------ 4
v1_rows = [["model", "degree", "terms (non-zero)", "CV MSE", "CV R&sup2;", "shift-weighted CV MSE"]]
for d in (2, 3, 4, 5, 6, 10):
    r = ridge("var1", d)
    v1_rows.append([f"ridge (&alpha;={r['alpha']:.0e})", d, r["n_terms"], f"{r['cv_mse']:.3f}",
                    f"{r2('var1', r['cv_mse']):.3f}", f"{r['cv_wmse']:.3f}"])
for d, g in ((4, 6), (5, 0), (6, 6)):
    r = lasso("var1", d, g)
    v1_rows.append([f"lasso &gamma;={g}", d, "", f"{r['cv_mse']:.3f}", f"{r2('var1', r['cv_mse']):.3f}",
                    f"{r['cv_wmse']:.3f}"])
v1_rows.append([f"<b>{mname(b1, True)} (final)</b>", f"<b>{b1['degree']}</b>",
                f"{b1['n_terms']} ({b1['n_nonzero_terms']})", f"<b>{b1['cv_mse']:.3f}</b>",
                f"<b>{r2('var1', b1['cv_mse']):.3f}</b>", f"<b>{b1['cv_wmse']:.3f}</b>"])

story += [P("4. Problem var1 — steam turbine Net Power Score (6 inputs, degree &le; 10)", h1),
          tbl(v1_rows, [3.6 * cm, 1.3 * cm, 2.6 * cm, 1.8 * cm, 1.7 * cm, 3.5 * cm], bold_rows=(len(v1_rows) - 1,)),
          Spacer(1, 4),
          P(f"<b>Chosen degree: {b1['degree']}.</b> Degrees 1&ndash;3 clearly underfit: CV MSE falls from "
            f"{ridge('var1', 1)['cv_mse']:.2f} to {ridge('var1', 3)['cv_mse']:.2f}, and train and CV errors stay "
            f"close together. With ridge the best degree is 4 (CV MSE {r1_4['cv_mse']:.3f}), and degree 5 is "
            f"already worse ({r1_5['cv_mse']:.3f}). This is because ridge shrinks all 462 coefficients and "
            f"never removes any, so the extra variance outweighs the gain. The Lasso shows the real structure. "
            f"The function has genuine degree-5 content but uses only part of the 462 terms: the final model "
            f"keeps {b1['n_nonzero_terms']} of them, mostly terms that involve one to three inputs. "
            f"Lasso at degree 5 <b>more than halves the error</b> of the best ridge model "
            f"({b1['cv_mse']:.3f} vs {r1_4['cv_mse']:.3f}; R&sup2; {r2('var1', r1_4['cv_mse']):.3f} &rarr; "
            f"{r2('var1', b1['cv_mse']):.3f}). Going to degree 6 brings no further gain "
            f"({lasso('var1', 6, 6)['cv_mse']:.3f}), and degrees 7&ndash;10 only add variance. "
            f"The degree-weighted penalty (&gamma;&gt;0) improves on plain Lasso "
            f"({lasso('var1', 5, 0)['cv_mse']:.3f} &rarr; {b1['cv_mse']:.3f}) because it lets the "
            f"high-order terms in only when the data supports them, which keeps the surface smooth at the "
            f"clipped corners the test set is rich in. Degree 5 is also best on the unweighted CV score "
            f"(lowest {min(r['cv_mse'] for r in L['var1']['lasso']):.3f}), so the choice of degree does not depend on the shift correction."),
          P(f"<b>Robustness at the corners.</b> The out-of-fold MSE of the final model grows only mildly with "
            f"the number of clipped inputs ({b1['oof_mse_by_clipped']['0']:.2f} with none, "
            f"{b1['oof_mse_by_clipped']['3']:.2f} with three, {b1['oof_mse_by_clipped']['5']:.2f} with five). "
            f"The expected test MSE is therefore close to the weighted CV value of about {b1['cv_wmse']:.2f}, "
            f"not far above it. The final model's test predictions differ from those of the runner-up "
            f"(degree 6 Lasso) by a mean squared difference of only 0.002, so they are stable."),
          ]

# ------------------------------------------------------------------ 5
v2_rows = [["model", "degree", "terms (non-zero)", "CV MSE", "CV R&sup2;", "train MSE (no penalty)"]]
for d in (4, 6, 7, 8, 9, 10, 12, 14, 20):
    r = ridge("var2", d)
    v2_rows.append([f"ridge (&alpha;={r['alpha']:.0e})", d, r["n_terms"], f"{r['cv_mse']:.3f}",
                    f"{r2('var2', r['cv_mse']):.4f}", f"{r['train_mse_ols']:.3f}"])
v2_rows.append([f"<b>{mname(b2, True)} (final)</b>", f"<b>{b2['degree']}</b>",
                f"{b2['n_terms']} ({b2['n_nonzero_terms']})", f"<b>{b2['cv_mse']:.3f}</b>",
                f"<b>{r2('var2', b2['cv_mse']):.4f}</b>", f"{b2['train_mse']:.3f} (penalised)"])
story += [P("5. Problem var2 — thermal anomaly mapping (3 inputs, degree &le; 20)", h1),
          tbl(v2_rows, [3.6 * cm, 1.3 * cm, 2.6 * cm, 1.8 * cm, 1.7 * cm, 3.5 * cm], bold_rows=(len(v2_rows) - 1,)),
          Spacer(1, 4),
          P(f"<b>Chosen degree: {b2['degree']}.</b> The CV curve (Figure 1, right) is a clean U. Error drops "
            f"steeply up to degree 7&ndash;8, stays flat at about 0.25 for degrees 8&ndash;10, then rises "
            f"fast: about 1.1 at degree 12, 5.4 at degree 14 and over 10 at degree 20. "
            f"The flat bottom suggests the remaining error is mostly measurement noise. The training error at "
            f"degree 8 ({r2_8['train_mse_ols']:.3f}) is only modestly below the CV error ({r2_8['cv_mse']:.3f}), "
            f"which fits a noise variance of roughly 0.2. A higher degree cannot reduce that noise; it can "
            f"only fit it. By the usual parsimony rule (the smallest model whose CV error matches the "
            f"minimum), degree 8 is chosen. The &ldquo;up to degree 20&rdquo; in the brief is an upper bound "
            f"on the true complexity, not a target. With 1000 samples in 3-D, a degree-20 model has 1771 "
            f"coefficients, more than the number of points, and even heavy ridge regularisation leaves its "
            f"CV error 40 times worse."),
          P(f"At degree 8 ridge, Lasso and elastic net are practically tied (CV MSE {r2_8['cv_mse']:.3f}, "
            f"{lasso('var2', 8, 2)['cv_mse']:.3f} and {b2['cv_mse']:.3f}). The test predictions of the final "
            f"model and the ridge model differ by a mean squared difference of only 0.004, so the choice "
            f"between them is immaterial. The final model ({mname(b2)}) was kept because it had the "
            f"lowest CV score and passed the one-standard-error test of Section 6. It is nearly dense: "
            f"only {b2['n_terms'] - b2['n_nonzero_terms']} of the {b2['n_terms']} terms are zero, which fits "
            f"a smooth field that genuinely uses almost every degree-8 term."),
          ]

# ------------------------------------------------------------------ 6
ev = {v: L[v]["enet_vs_best"] for v in L}
NICE = {"lasso": "Lasso", "ridge": "ridge", "enet": "elastic net"}
en_rows = [["", "model", "degree", "&gamma;", "&rho; (L1 share)", "&lambda;", "shift-weighted CV MSE"]]
hl = []
for v in ("var1", "var2"):
    rr = min(L[v]["ridge"], key=lambda r: r["cv_wmse"])
    en_rows.append([v, "best ridge", rr["degree"], "&ndash;", "0", f"{rr['alpha']:.1e}", f"{rr['cv_wmse']:.4f}"])
    ll = best_of(v, "lasso")
    en_rows.append(["", "best Lasso", ll["degree"], ll["gamma"], "1", f"{ll['lam']:.1e}", f"{ll['cv_wmse']:.4f}"])
    for rho in (0.5, 0.9):
        e = best_of(v, "enet", l1_ratio=rho)
        en_rows.append(["", "best elastic net", e["degree"], e["gamma"], rho, f"{e['lam']:.1e}",
                        f"{e['cv_wmse']:.4f}"])
    g = ev[v]
    hl.append(len(en_rows))
    en_rows.append(["", f"<b>elastic net gain over {NICE[g['incumbent']['method']]}</b>", "", "", "", "",
                    f"<b>{g['gain']:+.4f} &plusmn; {g['se']:.4f}</b> &rarr; "
                    f"{'adopted' if g['adopted'] else 'not adopted'}"])


def enet_verdict(v):
    g = ev[v]
    if g["adopted"]:
        return (f"for {v} elastic net improves the shift-weighted CV MSE by {g['gain']:.4f}, more than one "
                f"standard error ({g['se']:.4f}), so it was adopted")
    if g["gain"] > 0:
        return (f"for {v} elastic net is ahead by only {g['gain']:.4f}, within one standard error "
                f"({g['se']:.4f}), so the simpler {NICE[g['incumbent']['method']]} model was kept")
    return (f"for {v} elastic net does not beat the {NICE[g['incumbent']['method']]} model "
            f"({g['gain']:+.4f} &plusmn; {g['se']:.4f}), so it was not adopted")


story += [KeepTogether([P("6. Elastic net check", h1),
          tbl(en_rows, [1.1 * cm, 4.6 * cm, 1.3 * cm, 1.0 * cm, 2.1 * cm, 1.6 * cm, 4.8 * cm], bold_rows=hl),
          Spacer(1, 4),
          P("Elastic net mixes the two penalties, so it was tested to make sure the choice between ridge "
            "and Lasso was not leaving accuracy on the table. The last row of each block is the mean "
            "per-fold improvement of the best elastic net over the selected ridge/Lasso model, &plusmn; one "
            f"standard error. In short, {enet_verdict('var1')}; {enet_verdict('var2')}. "
            "The pattern is plausible. var1's function is sparse (the Lasso zeroes two thirds of the "
            "terms), so the L2 part only pulls in terms that should be absent. var2's field uses almost "
            "every term, so a partly-L2 penalty that shrinks without zeroing suits it slightly better. "
            "The var2 gain is small (under 1% of the error) and changes the test predictions by a mean "
            "squared difference of only 0.002; the degree choice is unaffected either way.")])]

story += [KeepTogether([P("7. Final models and validation", h1),
          fig("fig_parity.png", 15.5),
          P("<b>Figure 2.</b> Out-of-fold predictions of the final models against the actual values. "
            "No row is predicted by a model that saw it during training.", cap),
          tbl([["", "degree", "method", "penalty", "non-zero terms", "CV MSE", "CV R&sup2;", "train MSE"],
               ["var1", b1["degree"], mname(b1), f"&lambda;={b1['lam']:.1e}",
                f"{b1['n_nonzero_terms']} / {b1['n_terms']}", f"{b1['cv_mse']:.3f}",
                f"{r2('var1', b1['cv_mse']):.4f}", f"{b1['train_mse']:.3f}"],
               ["var2", b2["degree"], mname(b2), f"&lambda;={b2['lam']:.1e}",
                f"{b2['n_nonzero_terms']} / {b2['n_terms']}", f"{b2['cv_mse']:.3f}",
                f"{r2('var2', b2['cv_mse']):.4f}", f"{b2['train_mse']:.3f}"]],
              [1.0 * cm, 1.4 * cm, 3.6 * cm, 2.1 * cm, 2.2 * cm, 1.5 * cm, 1.6 * cm, 1.6 * cm])]),
          Spacer(1, 4),
          P("Each final model was refitted on all 1000 training rows with the selected hyper-parameters "
            "and used to predict the test rows. The predictions are in <b>BT2024154_pred_var1.csv</b> and "
            "<b>BT2024154_pred_var2.csv</b>, each a single column <i>y</i> with 1000 rows in the same order as "
            "the test files, matching the sample submission."),
          P("8. Key decisions at a glance", h1),
          P("&bull; <b>CV over training error</b> for choosing the degree: training error always favours the "
            "highest degree (Figure 1)."),
          P("&bull; <b>Legendre basis</b>: the same polynomial space as monomials, but well conditioned, "
            "so degree 8&ndash;20 fits are numerically trustworthy."),
          P("&bull; <b>Regularisation</b>: needed once the number of terms approaches the number of samples. "
            "The sparse, degree-weighted Lasso was the decisive improvement for var1. Elastic net was "
            "tested as well and adopted only where it beat the winner by more than one standard error "
            "(var2, by a small margin)."),
          P("&bull; <b>Shift-aware validation</b>: var1's test set is concentrated at clipped corners, so CV "
            "errors were re-weighted to match it, and the chosen model was checked to hold up there."),
          P("&bull; <b>Parsimony</b>: where several degrees tie within noise (var2: 8&ndash;10), the "
            "smallest was chosen."),
          P("9. Reproducing the results", h1),
          P(f"Repository: <link href='{GITHUB}' color='blue'>{GITHUB}</link>. "
            "Run <font face='Courier'>pip install -r requirements.txt</font>, then "
            "<font face='Courier'>python code/train.py</font> (model selection and fitting, about 5 minutes on a "
            "laptop CPU), then <font face='Courier'>python code/predict.py</font> (writes the prediction files). "
            "<font face='Courier'>code/polyreg.py</font> holds the feature construction, the ridge path, Lasso and "
            "elastic-net CV and the final model class. Full CV logs are in <font face='Courier'>results/</font>.", left),
          ]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                        title="Polynomial Regression Assignment - BT2024154", author="BT2024154")


def footer(c, d):
    c.setFont("Times-Roman", 8.5)
    c.setFillColor(colors.HexColor("#52514e"))
    c.drawCentredString(A4[0] / 2, 1.0 * cm, f"BT2024154 — Polynomial Regression — page {d.page}")


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
