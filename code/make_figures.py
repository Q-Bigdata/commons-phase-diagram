"""Main and Extended Data figures from the processed tables in proc/ (self-contained; style helpers in fig_style.py).

usage: python3 make_figures.py [proc_dir] [fig_dir]
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import apply_figure_style, panel_letter, META_GREY

PROC = sys.argv[1] if len(sys.argv) > 1 else "proc"
FIG = sys.argv[2] if len(sys.argv) > 2 else "fig"
os.makedirs(FIG, exist_ok=True)
apply_figure_style(sizes=(7, 6, 6))
FAMC = {"F1": "#2166ac", "F2": "#d6604d", "F3": "#1b7837"}
FAML = {"F1": "Evolving payoff-maximisers (F1)", "F2": "Calibrated responders (F2)", "F3": "Q-learners (F3)"}
FAMS = ["F1", "F2", "F3"]
RULEC = {"equal": "#9e9e9e", "mixed": "#bdbdbd", "proportional": "#4393c3", "power22": "#762a83",
         "step97": "#c51b7d", "capstep": "#000000"}
RULEL = {"equal": "Equal", "mixed": "Mixed (w = 0.5)", "proportional": "Proportional",
         "power22": "Distilled power rule", "step97": "Step at 0.97 capacity", "capstep": "Step at capacity"}
W2 = 7.2


def rd(name):
    return pd.read_csv(os.path.join(PROC, name))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def fig1():
    br = rd("theory_br.csv")
    fp = rd("theory_fixedpoints.csv")
    dv = rd("theory_signmap.csv")
    ct = rd("theory_contest.csv")
    fig = plt.figure(figsize=(W2, 4.3))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.5, left=0.07, right=0.985, top=0.93, bottom=0.1)
    # a rule shapes
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0, 1, 501)
    ax.plot(x, np.ones_like(x), color=RULEC["equal"], lw=1)
    ax.plot(x, 0.5 * np.ones_like(x), color=RULEC["mixed"], lw=1)
    ax.plot(x, np.zeros_like(x), color=RULEC["proportional"], lw=1)
    ax.plot(x, x ** 22, color=RULEC["power22"], lw=1.3)
    ax.step(x, np.where(x >= 0.97, 1, 0), where="post", color=RULEC["step97"], lw=1.3)
    for k, (yy, lab) in {"equal": (1.03, "equal"), "mixed": (0.53, "mixed"), "proportional": (0.03, "proportional")}.items():
        ax.text(0.02, yy, lab, color=RULEC[k], fontsize=6, va="bottom")
    ax.text(0.40, 0.70, "distilled $(R/R_0)^{22}$", color=RULEC["power22"], fontsize=6)
    ax.text(0.40, 0.30, "step at $0.97R_0$", color=RULEC["step97"], fontsize=6)
    ax.set_xlabel("Pool relative to capacity, $R/R_0$")
    ax.set_ylabel("Equal-sharing weight, $w(R)$")
    ax.set_title("Allocation rules compared")
    ax.margins(0.04)
    panel_letter(ax, "a")
    # b best responses at r = 0.4, T = 40
    ax = fig.add_subplot(gs[0, 1])
    d = br[(br.r.round(3) == 0.4) & (br["T"] == 40) & (br.p == 4)]
    for rule in ["proportional", "mixed", "power22", "step97", "capstep"]:
        q = d[d.rule == rule].sort_values("phi_res")
        ax.plot(q.phi_res, q.br, color=RULEC[rule], lw=1.1 if rule != "capstep" else 0.9,
                ls="--" if rule == "capstep" else "-", label=RULEL[rule])
    ax.plot([0.3, 1], [0.3, 1], color=META_GREY, lw=0.6, ls=":")
    ax.axvline(1 / 1.4, color=META_GREY, lw=0.6)
    ax.text(1 / 1.4 - 0.01, 0.05, "$1/(1+r)$", color=META_GREY, fontsize=6, ha="right")
    ax.set_xlabel("Residents' return fraction")
    ax.set_ylabel("Best response")
    ax.set_title("Best responses, $r=0.4$, $T=40$")
    ax.legend(loc="lower left", fontsize=5, handlelength=1.4)
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.0, 1.02)
    panel_letter(ax, "b")
    # c fixed-point offset from the boundary
    ax = fig.add_subplot(gs[0, 2])
    m = fp[(fp["T"] == 40) & (fp.p == 4) & fp.downcrossing]
    for rule in ["capstep", "step97", "power22"]:
        q = m[m.rule == rule].sort_values("r")
        ax.plot(q.r, q.fixed_point - q.boundary, "o-", ms=2.5, color=RULEC[rule], lw=1, label=RULEL[rule])
    ax.axhline(0, color=META_GREY, lw=0.6)
    ax.axvline(8 / 39, color=META_GREY, lw=0.6, ls=":")
    ax.text(8 / 39 + 0.02, -0.006, "$r(T-1)=2p$", fontsize=6, color=META_GREY)
    ax.set_xlabel("Growth rate, $r$")
    ax.set_ylabel("Fixed point minus $1/(1+r)$")
    ax.set_title("Exact boundary if $r(T-1)>2p$")
    ax.legend(loc="lower right", fontsize=5)
    panel_letter(ax, "c")
    # d region in which the boundary attracts (left derivative positive), analytic with numerical check
    ax = fig.add_subplot(gs[1, 0])
    rr = np.linspace(0.04, 1.0, 300)
    ax.fill_between(rr, 1 + 2 / rr, 400, color="#d9d9d9", lw=0)
    ax.fill_between(rr, 1 + 8 / rr, 400, color="#c994c7", lw=0, alpha=0.8)
    ax.plot(rr, 1 + 2 / rr, color="k", lw=0.8, ls="--")
    ax.plot(rr, 1 + 8 / rr, color=RULEC["step97"], lw=0.9)
    q = dv[(dv.p == 4) & (dv.rule == "step97")]
    ok = (np.sign(q.D_minus) == np.sign(q.D_minus_analytic)).mean()
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80, 160]); ax.set_yticklabels(["5", "10", "20", "40", "80", "160"])
    ax.set_ylim(4, 200); ax.set_xlim(0.04, 1.0)
    ax.text(0.5, 60, "boundary attracts:\nall contingent rules", fontsize=5.5, ha="center")
    ax.text(0.75, 7.0, "capacity\nstep only", fontsize=5.5, ha="center")
    ax.text(0.12, 5.0, "neither", fontsize=5.5)
    ax.set_xlabel("Growth rate, $r$")
    ax.set_ylabel("Horizon, $T$")
    ax.set_title("Where the boundary attracts ($p=4$)")
    panel_letter(ax, "d")
    # e contest efficiency vs horizon
    ax = fig.add_subplot(gs[1, 1])
    c4 = ct[(ct.p == 4) & (ct.r.round(3) == 0.4)].sort_values("T")
    TT = np.linspace(10, 320, 300)
    ps = 1 / (1 + 2 / ((TT - 1) * 0.75))
    ax.plot(TT, (1 - ps) * 1.4 / 0.4, color=RULEC["proportional"], lw=1.1, label="singular strategy (Prop. 2)")
    c4 = c4[c4.in_domain]
    ax.plot(c4["T"], c4.eff_sim, "o", ms=3, color="k", label="simulated, monomorphic")
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 160, 320]); ax.set_xticklabels(["10", "20", "40", "80", "160", "320"])
    ax.set_xlabel("Horizon, $T$")
    ax.set_ylabel("Efficiency")
    ax.set_title("Proportional allocation dissipates surplus")
    ax.legend(fontsize=5, loc="upper right")
    panel_letter(ax, "e")
    # f free riders
    ax = fig.add_subplot(gs[1, 2])
    fr = rd("theory_freeriders.csv")
    z = fr[(fr.r.round(3) == 0.4) & (fr.phi_c == 1.0)]
    for rule in ["equal", "proportional", "power22", "step97", "capstep"]:
        q = z[z.rule == rule].sort_values("f")
        ax.plot(q.f, q.minR, color=RULEC[rule], lw=1.1 if rule != "capstep" else 0.9,
                ls="--" if rule == "capstep" else "-", label=RULEL[rule])
    ax.axvline(0.4 / 1.4, color=META_GREY, lw=0.6, ls=":")
    ax.text(0.4 / 1.4 + 0.02, 0.42, "$r/(1+r)$", fontsize=6, color=META_GREY)
    ax.set_xlabel("Share of committed free riders, $f$")
    ax.set_ylabel("Minimum pool over 200 rounds")
    ax.set_title("Contingency tolerates free riders")
    ax.legend(fontsize=5, loc="lower left")
    panel_letter(ax, "f", dx=-0.34)
    save(fig, "Fig1_theory")


# ------------------------------------------------------------------------------------------------
SLICE_ROWS = [("S1_r_f", "f", [0.0, 0.1, 0.2, 0.3, 0.4], "free riders $f$"), ("S2_r_noise", "sig", [0.0, 0.1, 0.2], "noise $\\sigma$"),
              ("S3_r_p", "p", [3, 8, 16], "group size $p$"), ("S4_r_growthnoise", "eps", [0.1, 0.2, 0.3], "growth noise $\\varepsilon$"),
              ("S5_r_horizon", "T", [20, 80], "horizon $T$")]
R7 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def fig2():
    S = rd("env_optimum_summary.csv")
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.55, wspace=0.12, left=0.105, right=0.9, top=0.93, bottom=0.09)
    norm = TwoSlopeNorm(vmin=-0.05, vcenter=0.0, vmax=0.12)
    ylabels = []
    for j, fam in enumerate(FAMS):
        ax = fig.add_subplot(gs[0, j])
        M = []
        marks = []
        ylabels = []
        for sl, col, vals, lab in SLICE_ROWS:
            for v in vals:
                rowv, rowm = [], []
                for r in R7:
                    q = S[(S.family == fam) & (S.slice == sl) & np.isclose(S[col], v) & np.isclose(S.r, r)]
                    rowv.append(q.dT.iloc[0] if len(q) else np.nan)
                    rowm.append(q.status.iloc[0] if len(q) else "")
                M.append(rowv); marks.append(rowm)
                ylabels.append(f"{lab.split(' ')[-1]} = {v:g}")
        M = np.array(M)
        im = ax.imshow(M, cmap="PuOr", norm=norm, aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if marks[a][b] == "T~":
                    ax.text(b, a, "~", ha="center", va="center", fontsize=6, color="k")
                elif marks[a][b] == "nonT":
                    ax.text(b, a, "x", ha="center", va="center", fontsize=6, color="k", fontweight="bold")
        ax.set_xticks(range(7)); ax.set_xticklabels([f"{v:g}" for v in R7])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels if j == 0 else [], fontsize=5.5)
        ax.set_xlabel("Growth rate, $r$")
        ax.set_title(FAML[fam], color=FAMC[fam], fontsize=6.5)
        k = 0
        for sl, col, vals, lab in SLICE_ROWS[:-1]:
            k += len(vals)
            ax.axhline(k - 0.5, color="white", lw=1.2)
        if j == 0:
            panel_letter(ax, "a", dx=-0.42)
    cax = fig.add_axes([0.915, 0.47, 0.012, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Contingency advantage $\\Delta_T$\n(best contingent minus best other)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    # b advantage versus noise-to-margin ratio
    gsb = gs[1, :].subgridspec(1, 3, wspace=0.42)
    ax = fig.add_subplot(gsb[0, 0:2])
    for fam in FAMS:
        q = S[S.family == fam]
        e = q.eta.clip(lower=0.02)
        ax.scatter(e, q.dT, s=6, color=FAMC[fam], alpha=0.75, lw=0, label=fam)
    ax.axhline(0, color=META_GREY, lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0 (no noise)", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("Noise-to-margin ratio $\\eta=\\sigma_{tot}/(\\sqrt{p}\\,r/(1+r))$ (heuristic)")
    ax.set_ylabel("$\\Delta_T$ (fresh seeds)")
    ax.set_title("The advantage narrows as noise outgrows the growth margin")
    ax.legend(fontsize=5, loc="upper right", markerscale=1.5)
    panel_letter(ax, "b", dx=-0.08)
    # c status counts
    ax = fig.add_subplot(gsb[0, 2])
    cats = [("T", "contingent best", "#542788"), ("T~", "contingent tied", "#b2abd2"), ("nonT", "other regime best", "#e08214")]
    bottom = np.zeros(3)
    for code, lab, col in cats:
        v = np.array([(S[S.family == f].status == code).sum() for f in FAMS])
        ax.bar(range(3), v, bottom=bottom, color=col, label=lab, width=0.65)
        bottom += v
    ax.set_xticks(range(3)); ax.set_xticklabels(FAMS)
    ax.set_ylabel("Environments (of 112)")
    ax.set_title("Fresh-seed regime status")
    ax.set_ylim(0, 135)
    ax.legend(fontsize=4.8, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    panel_letter(ax, "c")
    save(fig, "Fig2_phase_diagram")


# ------------------------------------------------------------------------------------------------
def fig3():
    A = rd("amnesty_decomposition.csv")
    TH = rd("threshold_location.csv")
    PR = rd("threshold_profile.csv")
    CP = rd("common_policy_regret.csv")
    PE = rd("common_policy_per_env.csv")
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.075, right=0.985, top=0.86, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    fs = [0.0, 0.1, 0.2, 0.3, 0.4]
    for k, fam in enumerate(FAMS):
        q = A[(A.family == fam) & (A.slice == "S1_r_f")]
        med = [q[np.isclose(q.f, f)]["diff"].median() for f in fs]
        for i, f in enumerate(fs):
            v = q[np.isclose(q.f, f)]["diff"]
            ax.scatter(np.full(len(v), i + (k - 1) * 0.22), v, s=5, color=FAMC[fam], alpha=0.6, lw=0)
        ax.plot(np.arange(5) + (k - 1) * 0.22, med, "_", ms=8, color=FAMC[fam], mew=1.4, label=fam)
    ax.axhline(0, color=META_GREY, lw=0.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{f:g}" for f in fs])
    ax.set_xlabel("Share of committed free riders, $f$")
    ax.set_ylabel("Best re-inclusion minus best\npermanent-exclusion rule")
    ax.set_title("Re-inclusion adds to contingent weighting")
    ax.legend(fontsize=5, loc="upper left")
    panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    th = sorted(PR.theta.unique())
    med = PR.groupby("theta").loss.median()
    q90 = PR.groupby("theta").loss.quantile(0.9)
    ax.fill_between(med.index, 0, q90.values, color="#d8daeb", lw=0, label="90th percentile")
    ax.plot(med.index, med.values, "o-", color="#542788", ms=3, lw=1, label="median")
    ax2 = ax.twinx()
    cnt = TH.theta.value_counts().sort_index()
    ax2.bar(cnt.index, cnt.values, width=0.012, color=META_GREY, alpha=0.6)
    ax2.set_ylabel("Cells with this optimal trigger", color=META_GREY, fontsize=6)
    ax2.tick_params(axis="y", labelsize=5.5, colors=META_GREY)
    ax2.spines["right"].set_visible(True)
    ax.set_xlabel("Step trigger $\\theta$ (pool / capacity)")
    ax.set_ylabel("Efficiency loss vs best trigger")
    ax.set_title("Near-capacity triggers form a plateau")
    ax.legend(fontsize=5, loc="upper left")
    ax.set_xlim(0.25, 1.03)
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    W = pd.read_csv(os.path.join(PROC, "common_policy_worst.csv"))
    order = ["global_minimax", "step97", "distilled", "proportional", "equal"]
    lab = {"global_minimax": "minimax rule", "step97": "step 0.97", "distilled": "distilled", "proportional": "proportional",
           "equal": "equal"}
    col = {"global_minimax": "#542788", "step97": RULEC["step97"], "distilled": RULEC["power22"],
           "proportional": RULEC["proportional"], "equal": RULEC["equal"]}
    for i, nm in enumerate(order):
        v = W[W.policy == nm].worst_regret
        ax.scatter(np.random.default_rng(i).uniform(-0.18, 0.18, len(v)) + i, v, s=4, color=col[nm], alpha=0.6, lw=0)
        ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="k", lw=1.2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[o] for o in order], rotation=30, ha="right")
    ax.set_ylabel("Worst-family regret (efficiency)")
    ax.set_title("A single contingent rule is robust")
    ax.set_ylim(-1.0, 0.05)
    panel_letter(ax, "c")
    save(fig, "Fig3_mechanism")


# ------------------------------------------------------------------------------------------------
def fig4():
    cur = rd("es_curves.csv")
    S = rd("es_summary.csv")
    gaps = rd("es_planner_gaps.csv")
    fig = plt.figure(figsize=(W2, 2.8))
    gs = fig.add_gridspec(1, 4, wspace=0.62, left=0.065, right=0.99, top=0.86, bottom=0.2, width_ratios=[1.1, 1.2, 1.0, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    lam1 = os.path.exists(os.path.join(PROC, "es_curves_lam1.csv"))
    for tag, col, lab in [("best", "#542788", "trained on data-calibrated F2"), ("lam1", "#e08214", "trained on $\\lambda=1$ F2")]:
        c_ = rd("es_curves.csv") if tag == "best" else (rd("es_curves_lam1.csv") if lam1 else None)
        if c_ is None:
            continue
        c_ = c_[c_.tag == tag]
        for (r, f, s), q in c_.groupby(["r", "f", "seed"]):
            ax.plot(q.Rmid, q.w_med, color=col, lw=0.4, alpha=0.35)
        g = c_.groupby("Rmid").w_med.median()
        ax.plot(g.index, g.values, color=col, lw=1.6, label=lab)
    ax.axhline(0, color=META_GREY, lw=0.5); ax.axhline(1, color=META_GREY, lw=0.5, ls=":")
    ax.set_xlabel("Pool, $R/R_0$")
    ax.set_ylabel("Effective equal-sharing weight")
    ax.set_title("What the planners learned")
    ax.set_ylim(-0.6, 2.6)
    ax.legend(fontsize=5, loc="upper left")
    panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    rows = [("best", gaps)]
    if os.path.exists(os.path.join(PROC, "es_lam1_wide.csv")):
        L = rd("es_lam1_wide.csv")
        L = L.assign(F2=L["F2_lam1"] - L["step_lam1"])
        rows.append(("lam1", L))
    for k, (tag, d) in enumerate(rows):
        col = "#542788" if tag == "best" else "#e08214"
        for i, fam in enumerate(["F2", "F1", "F3"]):
            if tag == "best":
                v = d[fam]
            else:
                stepF = rd("es_crosseval_paired.csv")
                stepm = stepF[stepF.family == fam].groupby(["r", "f"]).step.first().reset_index()
                if fam == "F2":
                    v = d["F2"]
                else:
                    dd = d.merge(stepm, on=["r", "f"])
                    v = dd[f"{fam}_best"] - dd["step"] if f"{fam}_best" in dd else dd[fam + "_best"] - dd.step
            x = i + (k - 0.5) * 0.3
            ax.scatter(np.random.default_rng(i + 10 * k).uniform(-0.08, 0.08, len(v)) + x, v, s=5, color=col, alpha=0.7, lw=0)
            ax.plot([x - 0.12, x + 0.12], [np.median(v)] * 2, color="k", lw=1.1)
    ax.axhline(0, color=META_GREY, lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["training\n(F2)", "transfer\n(F1)", "transfer\n(F3)"])
    ax.set_ylabel("Planner minus step rule (efficiency)")
    ax.set_title("Transfer depends on the lesson")
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    ctl = rd("es_adaptation_control.csv")
    sub = ctl[ctl.cls == "contingent"]
    for i, fam in enumerate(["F1", "F3"]):
        q = sub[sub.family == fam]
        if len(q):
            ax.bar(i - 0.17, q.pre_frozen.iloc[0], width=0.32, color="#b2abd2", label="recipients frozen" if i == 0 else None)
            ax.bar(i + 0.17, q.pre_adapt.iloc[0], width=0.32, color="#542788", label="recipients adapt" if i == 0 else None)
    if os.path.exists(os.path.join(PROC, "es_lam1_control.csv")):
        lc = rd("es_lam1_control.csv")
        for i, fam in enumerate(["F1", "F3"]):
            q = lc[(lc.family == fam) & (lc.cls == "compensatory")]
            if len(q):
                ax.scatter([i - 0.17], [q.pre_frozen.iloc[0]], marker="D", s=12, color="#e08214", zorder=3,
                           label="$\\lambda=1$ compensatory, frozen" if i == 0 else None)
                ax.scatter([i + 0.17], [q.pre_adapt.iloc[0]], marker="D", s=12, facecolor="white", edgecolor="#e08214",
                           zorder=3, label="$\\lambda=1$ compensatory, adapt" if i == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["F1", "F3"])
    ax.set_ylabel("Efficiency")
    ax.set_title("Adaptation control")
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=4.4, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    panel_letter(ax, "c")
    ax = fig.add_subplot(gs[0, 3])
    kn = json.load(open(os.path.join(PROC, "key_numbers.json")))
    hr = kn["human_recip"]
    est = [("within-player,\nExp. 1 baselines", hr["exp1_baselines"]), ("within-player,\nall", hr["all_exp1to3"])]
    for i, (lab, e) in enumerate(est):
        ax.errorbar([e["lam"]], [i], xerr=[[1.96 * e["lam_se"]], [1.96 * e["lam_se"]]], fmt="o", color="k", ms=3, lw=0.8)
        ax.errorbar([e["gam"]], [i + 0.25], xerr=[[1.96 * e["gam_se"]], [1.96 * e["gam_se"]]], fmt="s", color=META_GREY, ms=3, lw=0.8)
    cb = kn["calib_best"]
    l1 = [x for x in kn["calib_profile_lam"] if abs(x["lam"] - 1) < 1e-9][0]
    ax.axvline(cb["lam"], color="#542788", lw=1); ax.axvline(1.0, color="#e08214", lw=1)
    ax.axvline(cb["gam"], color="#542788", lw=1, ls=":")
    ax.text(cb["lam"] + 0.02, 1.7, "$\\lambda$ (best fit)", fontsize=5, color="#542788")
    ax.text(0.62, 1.7, "$\\lambda=1$ fit", fontsize=5, color="#e08214")
    ax.text(cb["gam"] + 0.02, -0.55, "$\\gamma$ (fit)", fontsize=5, color="#542788")
    ax.set_yticks([0.12, 1.12]); ax.set_yticklabels([e[0] for e in est], fontsize=5)
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.8, 2.0)
    ax.set_xlabel("Coefficient")
    ax.set_title("Human data: weak matching")
    ax.text(0.02, -0.75, "circles: matching $\\lambda$; squares: generosity $\\gamma$", fontsize=4.6)
    panel_letter(ax, "d")
    save(fig, "Fig4_planners")


# ------------------------------------------------------------------------------------------------
def fig5():
    C = rd("t3_contrasts.csv")
    H = rd("human_facts.csv")
    LV = rd("t3_levels.csv")
    fig = plt.figure(figsize=(W2, 3.0))
    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.2, right=0.985, top=0.88, bottom=0.15, width_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, 0])
    ids = list(H.id)
    for i, fid in enumerate(ids):
        h = H[H.id == fid].iloc[0]
        y = len(ids) - 1 - i
        ax.errorbar([h.human_diff], [y], xerr=[[h.human_diff - h.lo], [h.hi - h.human_diff]], fmt="D", ms=3.5,
                    color="k", lw=0.9, zorder=3, label="human (40 groups per arm)" if i == 0 else None)
        for k, fam in enumerate(FAMS):
            q = C[(C.family == fam) & (C.integer == 0) & (C.id == fid)]
            if len(q):
                ax.errorbar([q.sim_diff.iloc[0]], [y + (k - 1) * 0.22 - 0.0], xerr=[[1.96 * q.sim_se.iloc[0]]] * 2,
                            fmt="o", ms=2.5, color=FAMC[fam], lw=0.7, label=FAML[fam] if i == 0 else None)
    ax.axvline(0, color=META_GREY, lw=0.6)
    labs = [f"{fid} {H[H.id == fid].fact.iloc[0]}" + ("  (fitted)" if H[H.id == fid].calibrated_on.iloc[0] else "") for fid in ids]
    ax.set_yticks(range(len(ids))[::-1]); ax.set_yticklabels(labs, fontsize=5.2)
    ax.set_xlabel("Difference between conditions (direction of the published fact > 0)")
    ax.set_title("Eleven published contrasts: human estimates and simulations")
    ax.legend(fontsize=4.8, loc="lower left")
    panel_letter(ax, "a", dx=-0.62)
    ax = fig.add_subplot(gs[0, 1])
    st_m = {"eff": "o", "gini": "s", "sust": "^", "all4": "v", "act": "D"}
    for fam in FAMS:
        q = LV[(LV.family == fam) & (LV.integer == 0)]
        for s, mk in st_m.items():
            z = q[q.stat == s]
            ax.errorbar(z.human, z.sim, xerr=[z.human - z.human_lo, z.human_hi - z.human], fmt=mk, ms=2.8, color=FAMC[fam],
                        lw=0.5, alpha=0.85)
    ax.plot([0, 1], [0, 1], color=META_GREY, lw=0.6, ls=":")
    ax.set_xlabel("Human estimate (95% CI)")
    ax.set_ylabel("Simulated value")
    ax.set_title("Levels by condition and outcome")
    for j, (s, mk) in enumerate(st_m.items()):
        ax.scatter([], [], marker=mk, color="k", s=6, label={"eff": "efficiency", "gini": "Gini", "sust": "pool sustained",
                                                             "all4": "all four active", "act": "active share"}[s])
    ax.legend(fontsize=4.8, loc="lower right")
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.05)
    panel_letter(ax, "b")
    save(fig, "Fig5_human")


# ------------------------------------------------------------------------------------------------
def fig6():
    R = rd("field_results.csv")
    ODC = rd("field_outcome_dependent_completions.csv")
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.075, right=0.985, top=0.86, bottom=0.22)
    cods = ["complete cases", "missing = absent", "missing = present"]
    short = ["complete\n(n = 27)", "missing =\nabsent (69)", "missing =\npresent (69)"]
    ax = fig.add_subplot(gs[0, 0])
    from math import sqrt

    def wil(k, n):
        if n == 0:
            return (np.nan, np.nan)
        z = 1.96; p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
        h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - h, c + h
    for i, cd in enumerate(cods):
        r = R[R.coding == cd].iloc[0]
        for k, (num, den, col, lab) in enumerate([(r.core_succ, r.core_n, "#542788", "2A + 2B + 5 present"),
                                                  (r.rest_succ, r.rest_n, META_GREY, "otherwise")]):
            p = num / den; lo, hi = wil(num, den)
            ax.errorbar([i + (k - 0.5) * 0.3], [p], yerr=[[max(p - lo, 0)], [max(hi - p, 0)]], fmt="o", color=col, ms=3.5, lw=0.9,
                        label=lab if i == 0 else None)
            ax.text(i + (k - 0.5) * 0.3, -0.12, f"{int(num)}/{int(den)}", ha="center", fontsize=4.8, color=col)
    ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=5.2)
    ax.set_ylim(-0.17, 1.08)
    ax.set_ylabel("Share of successful commons")
    ax.set_title("Association with the three codes")
    ax.legend(fontsize=5, loc="center", bbox_to_anchor=(0.55, 0.68))
    panel_letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    comps = [("H4_score", "simulation class score", "#542788"), ("count11", "count of 11 principles", "#4393c3"),
             ("count_core3", "count of 2A, 2B, 5", "#92c5de")]
    for i, cd in enumerate(cods[1:]):
        r = R[R.coding == cd].iloc[0]
        for k, (c_, lab, col) in enumerate(comps):
            x = i + (k - 1) * 0.25
            ax.errorbar([x], [r[f"auc_{c_}"]], yerr=[[r[f"auc_{c_}"] - r[f"auc_{c_}_lo"]], [r[f"auc_{c_}_hi"] - r[f"auc_{c_}"]]],
                        fmt="o", ms=3.2, color=col, lw=0.9, label=lab if i == 0 else None)
        if "ridge11_loo_auc" in r:
            ax.scatter([i + 0.42], [r.ridge11_loo_auc], marker="x", s=12, color="k", label="ridge on 11 codes (LOO)" if i == 0 else None)
    ax.axhline(0.5, color=META_GREY, lw=0.6, ls=":")
    ax.set_xticks(range(2)); ax.set_xticklabels(short[1:], fontsize=5.2)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("AUC (95% bootstrap CI)")
    ax.set_title("Simulation score is not more\ndiscriminating than simple counts")
    ax.legend(fontsize=4.6, loc="lower left")
    panel_letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    mis = rd("field_misclassification.csv")
    lab = ["S: absent\nF: absent", "S: absent\nF: present", "S: present\nF: absent", "S: present\nF: present"]
    ax.bar(range(4), ODC.rd, color=["#542788", "#b2abd2", "#b2abd2", "#542788"], width=0.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=4.8)
    ax.set_ylabel("Risk difference, core vs other")
    ax.set_title("Positive under outcome-dependent\ncompletions of missing codes")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(ODC.rd):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=5)
    panel_letter(ax, "c")
    save(fig, "Fig6_field")


# ------------------------------------------------------------------------------------------------
def ed_calibration():
    kn = json.load(open(os.path.join(PROC, "key_numbers.json")))
    tg = rd("calibration_targets.csv")
    c = json.load(open(os.path.join(PROC, "calib.json")))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.45, left=0.07, right=0.985, top=0.85, bottom=0.25))
    ax = axs[0]
    keys = [f"{m}|{s}" for m, s in zip(tg.mech, tg.stat)]
    x = np.arange(len(keys))
    ax.errorbar(x - 0.15, tg.target, yerr=1.96 * tg.se, fmt="D", color="k", ms=3, lw=0.8, label="human (95% CI)")
    ax.scatter(x + 0.05, [c["best_fit_values"][k] for k in keys], s=10, color="#542788", label="best fit", zorder=3)
    ax.scatter(x + 0.2, [c["micro_fit_values"][k] for k in keys], s=10, marker="s", color="#4393c3", label="micro-consistent fit", zorder=3)
    ax.scatter(x - 0.3, tg.approx_published_target_v1, s=8, marker="_", color="#e08214", label="approximate targets (earlier)")
    ax.set_xticks(x); ax.set_xticklabels([k.replace("proportional", "P").replace("equal", "E").replace("mixed", "M").replace("|", "\n") for k in keys], fontsize=5)
    ax.set_ylabel("Value")
    ax.set_title("Calibration targets (Exp. 1)")
    ax.legend(fontsize=4.6, loc="upper left")
    panel_letter(ax, "a")
    for ax, par, lab in [(axs[1], "lam", "matching $\\lambda$"), (axs[2], "sd_int", "intrinsic noise $s_{int}$")]:
        prof = kn[f"calib_profile_{'sdint' if par == 'sd_int' else par}"]
        ax.plot([q[par] for q in prof], [q["loss"] for q in prof], "o-", color="#542788", ms=3, lw=1)
        g = kn["calib_profile_gam"]
        ax.set_xlabel(lab)
        ax.set_ylabel("Profile loss (weighted SSE)")
        ax.set_title(f"Profile over {lab}")
        panel_letter(ax, "b" if par == "lam" else "c")
    save(fig, "ED_Fig1_calibration")


def ed_robust():
    if not os.path.exists(os.path.join(PROC, "robust_summary.csv")):
        return
    RB = rd("robust_summary.csv")
    labs = {"X1_bevholt": "density-dependent regrowth", "X2_obsnoise": "noisy stock observation", "X2_obsdelay": "delayed stock observation",
            "X3_recordnoise": "noisy contribution records", "X4_corrnoise": "correlated execution noise", "X5_integer": "integer tokens",
            "X6_sig_x_eps": "noise x growth noise", "X7_sig_x_p": "noise x group size", "X8_f_x_T": "free riders x horizon"}
    sl = list(labs)
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.6))
    for k, fam in enumerate(FAMS):
        for i, s in enumerate(sl):
            q = RB[(RB.slice == s) & (RB.family == fam)]
            if not len(q):
                continue
            y = i + (k - 1) * 0.25
            ax.scatter(q.dT, np.full(len(q), y), s=6, color=FAMC[fam], alpha=0.7, lw=0, label=fam if i == 0 else None)
            frac = (q.best_regime == "T").mean()
            ax.text(0.17, y, f"{int((q.best_regime == 'T').sum())}/{len(q)}", fontsize=4.6, color=FAMC[fam], va="center")
    ax.axvline(0, color=META_GREY, lw=0.6)
    ax.set_yticks(range(len(sl))); ax.set_yticklabels([labs[s] for s in sl], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Contingency advantage $\\Delta_T$ (grid)")
    ax.set_title("Structural and interaction tests", pad=12)
    ax.set_xlim(-0.26, 0.22)
    ax.legend(fontsize=5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False)
    ax.text(0.17, -0.9, "contingent\nwinners", fontsize=4.6, color=META_GREY, ha="left")
    save(fig, "ED_Fig2_robustness")


def ed_es():
    H = rd("es_hist.csv")
    S = rd("es_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.45, left=0.07, right=0.985, top=0.85, bottom=0.22))
    ax = axs[0]
    for (tag, r, f, s), q in H[H.tag == "best"].groupby(["tag", "r", "f", "seed"]):
        ax.plot(q.it, q.eff, lw=0.4, color=plt.cm.viridis((r - 0.2) / 0.6), alpha=0.6)
    ax.set_xlabel("ES iteration"); ax.set_ylabel("Held-out efficiency in F2")
    ax.set_title("Training curves (50 planners)")
    for rr in [0.2, 0.4, 0.8]:
        ax.plot([], [], color=plt.cm.viridis((rr - 0.2) / 0.6), lw=1, label=f"r = {rr:g}")
    ax.legend(fontsize=4.8, loc="lower right")
    panel_letter(ax, "a", dx=-0.3)
    ax = axs[1]
    sb = S[S.tag == "best"]
    ax.hist(sb.R2, bins=15, color="#542788")
    ax.set_xlabel("Projection $R^2$ (equal/proportional line)")
    ax.set_ylabel("Planners")
    ax.set_title("Distillation fit")
    panel_letter(ax, "b")
    ax = axs[2]
    tv = rd("es_table_vs_net.csv")
    for fam in FAMS:
        q = tv[tv.family == fam]
        ax.scatter(q.net, q.table, s=6, color=FAMC[fam], label=fam, lw=0)
    ax.plot([0, 1], [0, 1], color=META_GREY, lw=0.6, ls=":")
    ax.set_xlabel("Network planner efficiency")
    ax.set_ylabel("Executable distilled table")
    ax.set_title("Distilled rules reproduce the networks")
    ax.legend(fontsize=5)
    panel_letter(ax, "c")
    save(fig, "ED_Fig3_planners")


def ed_classscores():
    sc = json.load(open(os.path.join(PROC, "class_scores_all.json")))
    order = ["EQUAL", "MIXED-FORGIVING", "PROPORTIONAL", "THRESHOLD-PERMANENT", "THRESHOLD-AMNESTY"]
    keys = [k for k in sc if k.endswith("|mean")]
    fig, ax = plt.subplots(figsize=(W2 * 0.7, 2.6))
    M = np.array([[sc[k][o] for o in order] for k in keys])
    im = ax.imshow(M, aspect="auto", cmap="Purples", vmin=0, vmax=max(1.0, M.max()))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.6, color="k" if M[i, j] < 0.6 else "w")
    ax.set_xticks(range(5)); ax.set_xticklabels(["equal", "mixed\nforgiving", "proportional", "contingent\npermanent", "contingent\nre-inclusion"], fontsize=5)
    def _lab(k):
        env, est = k.replace("|mean", "").split("|")
        nm = {"prod": "sust x incl.", "joint": "joint", "sust": "sustained", "act": "inclusion", "eff": "efficiency"}[est]
        if env == "ref":
            e = "reference"
        else:
            r_, f_, s_, p_ = env.split("_")
            ch = []
            if r_ != "r0.4": ch.append("r = " + r_[1:])
            if f_ != "f0.2": ch.append("f = " + f_[1:])
            if s_ != "s0.05": ch.append("sigma = " + s_[1:])
            if p_ != "p4": ch.append("p = " + p_[1:])
            e = ", ".join(ch)
        return f"{e} | {nm}"
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([_lab(k) for k in keys], fontsize=4.6)
    ax.set_title("Simulated class scores by environment and estimand (three-family mean)")
    save(fig, "ED_Fig4_class_scores")


def ed_multiobjective():
    MO = rd("multiobjective.csv")
    pols = ["step97", "distilled", "proportional", "mixed", "equal", "step97_perm"]
    plab = {"step97": "step 0.97", "distilled": "distilled", "proportional": "proportional", "mixed": "mixed", "equal": "equal",
            "step97_perm": "step, permanent"}
    stats_ = [("eff", "efficiency"), ("sust", "pool sustained"), ("minR", "minimum pool"), ("act", "active share"),
              ("worst", "worst-off share"), ("gini", "Gini"), ("ever_excl", "ever excluded")]
    fig, axs = plt.subplots(1, len(stats_), figsize=(W2, 2.2), sharey=True,
                            gridspec_kw=dict(wspace=0.12, left=0.1, right=0.99, top=0.82, bottom=0.2))
    for j, (s, lab) in enumerate(stats_):
        ax = axs[j]
        for k, fam in enumerate(FAMS):
            for i, pl in enumerate(pols):
                v = MO[(MO.family == fam) & (MO.policy == pl)][s]
                ax.scatter([v.median()], [i + (k - 1) * 0.22], s=8, color=FAMC[fam], lw=0, label=fam if (j == 0 and i == 0) else None)
        ax.set_title(lab, fontsize=5.5)
        ax.tick_params(axis="x", labelsize=5)
        if j == 0:
            ax.set_yticks(range(len(pols))); ax.set_yticklabels([plab[p] for p in pols], fontsize=5.5)
    axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.5, loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=3, frameon=False)
    fig.suptitle("Median across the 112 environments: surplus, ecological risk, inclusion and equity", fontsize=6.5, x=0.1, ha="left")
    save(fig, "ED_Fig5_tradeoffs")


def ed_sensitivity():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.5, left=0.08, right=0.985, top=0.84, bottom=0.24))
    ax = axs[0]
    CV = rd("eta_crossvalidation.csv")
    q = CV[(CV.families == "pooled") & (CV.holdout == "slice")]
    ax.bar(range(len(q)), q.auc, color="#542788")
    nice = {"eta": "$\\eta$", "eta_eff": "$\\eta$ with $p_{eff}$", "eta_eps": "$\\eta$ with $p_{eff}$, $\\varepsilon$",
            "eta_emp": "measured\nvariability", "raw": "raw\nvariables", "eta_plus": "$\\eta$ + $\\varepsilon$, $f$, $T$"}
    ax.set_xticks(range(len(q))); ax.set_xticklabels([nice[v] for v in q.predictor], rotation=40, ha="right", fontsize=5)
    ax.set_ylabel("Held-out AUC (leave one slice out)")
    ax.set_title("Predicting the sign of $\\Delta_T$")
    ax.set_ylim(0, 1)
    panel_letter(ax, "a", dx=-0.3)
    ax = axs[1]
    E = rd("eta_table.csv")
    for fam in FAMS:
        z = E[E.family == fam]
        ax.scatter(z.eta.clip(lower=0.02), 1 / z.conc_T, s=5, color=FAMC[fam], lw=0, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("$\\eta$ (nominal)"); ax.set_ylabel("Effective group size $1/\\sum a_i^2$")
    ax.set_title("Allocation concentration")
    ax.legend(fontsize=5)
    panel_letter(ax, "b")
    ax = axs[2]
    cs = rd("calibration_phase_sensitivity.csv")
    g = cs.groupby("fit").apply(lambda d: (d.best_regime == "T").sum())
    ax.bar(range(len(g)), g.values, color="#d6604d")
    alt = {}
    kn = json.load(open(os.path.join(PROC, "key_numbers.json")))
    for tag in ["gridlam1", "gridv1"]:
        if f"{tag}_T" in kn:
            alt[tag] = kn[f"{tag}_T"]
    xs = list(g.index.str.strip('"')) + list(alt)
    ax.bar(range(len(g), len(g) + len(alt)), list(alt.values()), color="#e08214")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([x.replace("plaus", "P").replace("gridlam1", "$\\lambda$=1").replace("gridv1", "earlier fit") for x in xs],
                                                      rotation=60, fontsize=4.8)
    ax.set_ylabel("F2 cells with contingent winner")
    ax.set_title("Calibration dependence (of 112)")
    panel_letter(ax, "c")
    save(fig, "ED_Fig6_sensitivity")


if __name__ == "__main__":
    failures = 0
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, ed_calibration, ed_robust, ed_es, ed_classscores, ed_multiobjective, ed_sensitivity]:
        try:
            f()
            print("ok", f.__name__, flush=True)
        except Exception as ex:
            failures += 1
            import traceback
            traceback.print_exc()
            print("FAILED", f.__name__, ex, flush=True)

    sys.exit(1 if failures else 0)
