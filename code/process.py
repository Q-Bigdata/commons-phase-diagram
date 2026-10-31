"""Processing pipeline: raw stage outputs (out/, out_human/, out_field/) -> processed tables (proc/) and the
single source of quoted numbers (proc/key_numbers.json, consumed by the manuscript build).

usage: python3 process.py <section|all>    sections: theory phase reeval eta threshold t3 es robust sens field
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

OUT, HUM, FLD, PROC = "out", "out_human", "out_field", "proc"
os.makedirs(PROC, exist_ok=True)
KN_PATH = os.path.join(PROC, "key_numbers.json")
KN = json.load(open(KN_PATH)) if os.path.exists(KN_PATH) else {}
FAMS = ["F1", "F2", "F3"]
DELTA = 0.01   # practical-equivalence band (fraction of the stationary-surplus reference)


def kn(**kw):
    for k, v in kw.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        KN[k] = v


def save_kn():
    json.dump(KN, open(KN_PATH, "w"), indent=1, sort_keys=True)


def insts_table(tag="fine"):
    fn = os.path.join(OUT, f"insts_{tag}_ext.csv")
    I = pd.read_csv(fn if (tag == "fine" and os.path.exists(fn)) else os.path.join(OUT, f"insts_{tag}.csv"))
    I["regime"] = [sim.regime((r.kind, r.theta, r.wlo, r.whi, r.k, r.amnesty)) for r in I.itertuples()]
    return I


def sig_tot(fam, e, sd_int):
    s = e["sig"]
    return float(np.sqrt(sd_int ** 2 + s ** 2)) if fam == "F2" else float(s)


# ------------------------------------------------------------------------------------------------
def sec_theory():
    fp = pd.read_csv(os.path.join(OUT, "theory_fixedpoints.csv"))
    dv = pd.read_csv(os.path.join(OUT, "theory_derivatives.csv"))
    ct = pd.read_csv(os.path.join(OUT, "theory_contest.csv"))
    fr = pd.read_csv(os.path.join(OUT, "theory_freeriders.csv"))
    fp.to_csv(os.path.join(PROC, "theory_fixedpoints.csv"), index=False)
    m = fp[(fp["T"] == 40) & (fp.p == 4)]
    rows = []
    for rule in ["step97", "power22", "capstep"]:
        d = m[(m.rule == rule) & m.downcrossing]
        d = d.assign(offset=d.fixed_point - d.boundary)
        rows.append(d)
        kn(**{f"fp_{rule}_maxabs_offset_rge025": float(d[d.r >= 0.25].offset.abs().max()),
              f"fp_{rule}_offset_r01": float(d[np.isclose(d.r, 0.1)].offset.iloc[0]),
              f"fp_{rule}_offset_r005": float(d[np.isclose(d.r, 0.05)].offset.iloc[0]),
              f"fp_{rule}_offset_r02": float(d[np.isclose(d.r, 0.2)].offset.iloc[0]),
              f"fp_{rule}_mineff": float(d.eff_mono.min())})
    pd.concat(rows).to_csv(os.path.join(PROC, "theory_fixedpoint_offsets.csv"), index=False)
    # analytic one-sided derivatives versus numerical ones
    x = dv.dropna(subset=["D_minus_analytic"])
    kn(deriv_max_rel_err=float(((x.D_minus - x.D_minus_analytic).abs() / x.D_minus_analytic.abs().clip(lower=1.0)).max()))
    # sign map of the left derivative at the boundary
    sm = dv[dv.rule.isin(["capstep", "step97", "power22"])].assign(pos=lambda q: q.D_minus > 0)
    sm.to_csv(os.path.join(PROC, "theory_signmap.csv"), index=False)
    c4 = ct[(ct.p == 4) & np.isclose(ct.r, 0.4)].set_index("T")
    kn(contest_phi_T40=float(c4.loc[40, "phi_star"]), contest_eff_T20=float(c4.loc[20, "eff_analytic"]),
       contest_eff_T40=float(c4.loc[40, "eff_analytic"]), contest_eff_T320=float(c4.loc[320, "eff_analytic"]),
       contest_maxdiff_sim=float((ct[ct.in_domain].eff_analytic - ct[ct.in_domain].eff_sim).abs().max()))
    ct.to_csv(os.path.join(PROC, "theory_contest.csv"), index=False)
    fr.to_csv(os.path.join(PROC, "theory_freeriders.csv"), index=False)
    # free-rider tolerance checks: equal sharing sustains capacity iff (1-f) phi_c >= 1/(1+r)
    eq = fr[fr.rule == "equal"].copy()
    eq["pred_capacity"] = (1 - eq.f) * eq.phi_c >= 1 / (1 + eq.r) - 1e-9
    kn(fr_equal_capacity_match=float((eq.pred_capacity == eq.recovers).mean()))
    st = fr[(fr.rule == "capstep") & (fr.q >= 1 - 1e-9) & (fr.f < 1)]
    kn(fr_capstep_positive_all=bool((st.minR > 0).all()))
    br = pd.read_csv(os.path.join(OUT, "theory_br.csv"))
    br.to_csv(os.path.join(PROC, "theory_br.csv"), index=False)


# ------------------------------------------------------------------------------------------------
def load_grid(name="grid", tag="fine"):
    I = insts_table(tag)
    G = []
    for f in FAMS:
        fn = os.path.join(OUT, f"{name}_{f}.csv.gz")
        if os.path.exists(fn):
            g = pd.read_csv(fn).merge(I, on="inst")
            G.append(g)
    return pd.concat(G), I


def cell_summary(G, E, sd_int):
    rows = []
    for (fam, env), d in G.groupby(["family", "env"]):
        e = E.loc[env]
        d = d.sort_values("eff", ascending=False)
        best = d.iloc[0]
        bt = d[d.regime == "T"]
        bn = d[d.regime != "T"]
        row = dict(family=fam, env=env, slice=e["slice"], r=e.r, p=e.p, T=e["T"], sig=e.sig, eps=e.eps, f=e.f,
                   best_inst=int(best.inst), best_regime=best.regime, best_eff=best.eff, best_theta=best.theta,
                   best_wlo=best.wlo, best_whi=best.whi, best_am=int(best.amnesty), best_kind=int(best.kind),
                   bestT_eff=bt.eff.max() if len(bt) else np.nan, bestnonT_eff=bn.eff.max() if len(bn) else np.nan,
                   bestnonT_regime=bn.iloc[0].regime if len(bn) else "")
        for rg in "PEITM":
            q = d[d.regime == rg]
            row[f"best_{rg}"] = q.eff.max() if len(q) else np.nan
        for a in (0, 1):
            q = bt[bt.amnesty == a]
            row[f"bestT_am{a}"] = q.eff.max() if len(q) else np.nan
        row["dT"] = row["bestT_eff"] - row["bestnonT_eff"]
        st = sig_tot(fam, e, sd_int)
        row["sig_tot"] = st
        row["eta"] = st / (np.sqrt(e.p) * e.r / (1 + e.r))
        rows.append(row)
    return pd.DataFrame(rows)


def sec_phase():
    E = pd.read_csv(os.path.join(OUT, "envs.csv")).set_index("env")
    c = json.load(open(os.path.join(OUT, "calib.json")))
    sd_int = c["best"]["sd_int"]
    G, I = load_grid()
    C = cell_summary(G, E, sd_int)
    C.to_csv(os.path.join(PROC, "env_optimum_summary_grid.csv"), index=False)
    cnt = C[C.best_regime == "T"].groupby("family").size()
    agree = C.pivot(index="env", columns="family", values="best_regime")
    kn(grid_T_F1=int(cnt.get("F1", 0)), grid_T_F2=int(cnt.get("F2", 0)), grid_T_F3=int(cnt.get("F3", 0)),
       grid_T_total=int(cnt.sum()), grid_cells=int(len(C)),
       grid_all3_T=int((agree == "T").all(1).sum()), n_inst_fine=int(len(I)),
       n_evals=int(len(G)), grid_counts=I.regime.value_counts().to_dict())
    # classifier cut-off sensitivity on the grid winners
    sens = []
    for pc, dc in [(0.1, 0.2), (0.05, 0.1), (0.2, 0.3), (0.25, 0.5)]:
        lab = [sim.regime((r.kind, r.theta, r.wlo, r.whi, r.k, r.amnesty), p_cut=pc, d_cut=dc)
               for r in I.itertuples()]
        L = dict(zip(I.inst, lab))
        sens.append(dict(p_cut=pc, d_cut=dc, **{f"T_{f}": int(sum(L[i] == "T" for i in C[C.family == f].best_inst))
                                                 for f in FAMS}))
    pd.DataFrame(sens).to_csv(os.path.join(PROC, "classifier_cutoff_sensitivity.csv"), index=False)
    kn(cutoff_sens=sens)
    # common policies on the common grid: regret relative to each family's best, per environment
    piv = {f: G[G.family == f].pivot(index="env", columns="inst", values="eff") for f in FAMS}
    reg = {f: piv[f].sub(piv[f].max(1), axis=0) for f in FAMS}
    worst = np.minimum(np.minimum(reg["F1"].to_numpy(), reg["F2"].to_numpy()), reg["F3"].to_numpy())
    insts = sim.inst_grid(True)
    gmm = int(np.argmax(worst.min(0)))
    gmean = int(np.argmax(worst.mean(0)))
    rowsC = []
    for nm, ii in [("global_minimax", gmm), ("global_mean_worst", gmean), ("step97", insts.index(sim.STEP)),
                   ("distilled", insts.index(sim.HEUR)), ("proportional", insts.index(sim.PROP)),
                   ("equal", insts.index(sim.EQ))]:
        w = worst[:, ii]
        rowsC.append(dict(policy=nm, inst=ii, regime=sim.regime(insts[ii]), inst_def=str(insts[ii]),
                          worst_regret_median=float(np.median(w)), worst_regret_min=float(w.min()),
                          frac_within_delta=float((w >= -DELTA).mean()), frac_within_005=float((w >= -0.05).mean()),
                          **{f"regret_{f}_median": float(np.median(reg[f].to_numpy()[:, ii])) for f in FAMS}))
    long = []
    for nm, ii in [("global_minimax", gmm), ("step97", insts.index(sim.STEP)), ("distilled", insts.index(sim.HEUR)),
                   ("proportional", insts.index(sim.PROP)), ("equal", insts.index(sim.EQ))]:
        for j, env in enumerate(piv["F1"].index):
            long.append(dict(policy=nm, env=env, worst_regret=float(worst[j, ii])))
    pd.DataFrame(long).to_csv(os.path.join(PROC, "common_policy_worst.csv"), index=False)
    CP = pd.DataFrame(rowsC)
    CP.to_csv(os.path.join(PROC, "common_policy_regret.csv"), index=False)
    per_env = [dict(env=env, inst=int(np.argmax(worst[j])), regime=sim.regime(insts[int(np.argmax(worst[j]))]),
                    worst_regret=float(worst[j].max())) for j, env in enumerate(piv["F1"].index)]
    PE = pd.DataFrame(per_env)
    PE.to_csv(os.path.join(PROC, "common_policy_per_env.csv"), index=False)
    kn(common_global_inst=str(insts[gmm]), common_global_regime=sim.regime(insts[gmm]),
       common_global_worst_median=float(np.median(worst[:, gmm])), common_global_worst_min=float(worst[:, gmm].min()),
       common_perenv_T=int((PE.regime == "T").sum()), common_perenv_n=int(len(PE)),
       common_step_worst_median=float(np.median(worst[:, insts.index(sim.STEP)])),
       common_step_worst_min=float(worst[:, insts.index(sim.STEP)].min()),
       common_heur_worst_median=float(np.median(worst[:, insts.index(sim.HEUR)])))


# ------------------------------------------------------------------------------------------------
def sec_reeval():
    E = pd.read_csv(os.path.join(OUT, "envs.csv")).set_index("env")
    c = json.load(open(os.path.join(OUT, "calib.json")))
    sd_int = c["best"]["sd_int"]
    I = insts_table()
    R = pd.read_csv(os.path.join(OUT, "reeval.csv.gz")).merge(I, on="inst")
    Gd, _ = load_grid()
    insts = sim.inst_grid(True) + [sim.MIX]
    iS, iH = insts.index(sim.STEP), insts.index(sim.HEUR)
    rows, amn, mo = [], [], []
    for (fam, env), d in R.groupby(["family", "env"]):
        e = E.loc[env]
        W = d.pivot(index="rep", columns="inst", values="eff")
        mu = W.mean()
        win = int(mu.idxmax())
        D = W.sub(W[win], axis=0)
        se = D.std(ddof=1) / np.sqrt(len(W))
        equiv = [int(i) for i in mu.index if (mu[i] - mu[win]) >= -max(DELTA, 2 * se[i])]
        regs = set(I.set_index("inst").loc[equiv, "regime"])
        rgm = I.set_index("inst").regime
        Tm = [i for i in mu.index if rgm[i] == "T"]
        nT = [i for i in mu.index if rgm[i] != "T"]
        bT = max(Tm, key=lambda i: mu[i]) if Tm else None
        bN = max(nT, key=lambda i: mu[i]) if nT else None
        dT = W[bT] - W[bN] if (bT is not None and bN is not None) else pd.Series([np.nan])
        g = Gd[(Gd.family == fam) & (Gd.env == env)].set_index("inst").eff
        status = ("T" if regs == {"T"} else ("T~" if "T" in regs else "nonT"))
        row = dict(family=fam, env=env, slice=e["slice"], r=e.r, p=e.p, T=e["T"], sig=e.sig, eps=e.eps, f=e.f,
                   winner=win, winner_regime=rgm[win], winner_eff=mu[win], n_equiv=len(equiv),
                   equiv_regimes="".join(sorted(regs)), status=status,
                   dT=float(dT.mean()), dT_se=float(dT.std(ddof=1) / np.sqrt(len(dT))) if len(dT) > 1 else np.nan,
                   bestT=bT, bestnonT=bN, bestnonT_regime=rgm[bN] if bN is not None else "",
                   grid_best_eff=float(g.max()), optimism=float(g.max() - mu[int(g.idxmax())]) if int(g.idxmax()) in mu.index else np.nan,
                   step_rel=float(mu[iS] / mu[win]) if mu[win] > 0 else np.nan,
                   heur_rel=float(mu[iH] / mu[win]) if mu[win] > 0 else np.nan,
                   step_gap=float(mu[iS] - mu[win]), heur_gap=float(mu[iH] - mu[win]),
                   theta_win=float(I.set_index("inst").theta[win]), am_win=int(I.set_index("inst").amnesty[win]),
                   whi_win=float(I.set_index("inst").whi[win]), wlo_win=float(I.set_index("inst").wlo[win]))
        st = sig_tot(fam, e, sd_int)
        row.update(sig_tot=st, eta=st / (np.sqrt(e.p) * e.r / (1 + e.r)))
        dd = d[d.inst == (bT if bT is not None else win)]
        row.update(conc_T=float(dd.conc.mean()), sdPhi_T=float(dd.sdPhi.mean()))
        rows.append(row)
        # amnesty decomposition within the contingent regime
        TA = [i for i in Tm if I.set_index("inst").amnesty[i] == 1]
        TP = [i for i in Tm if I.set_index("inst").amnesty[i] == 0]
        if TA and TP:
            a, b = max(TA, key=lambda i: mu[i]), max(TP, key=lambda i: mu[i])
            df_ = W[a] - W[b]
            amn.append(dict(family=fam, env=env, slice=e["slice"], f=e.f, r=e.r, best_amnesty=a, best_permanent=b,
                            diff=float(df_.mean()), se=float(df_.std(ddof=1) / np.sqrt(len(df_))),
                            excl_spell_perm=float(d[d.inst == b].spell.mean()),
                            ever_excl_perm=float(d[d.inst == b].ever_excl.mean())))
        # multi-objective profile of the winner and canonical rules
        for nm, ii in [("winner", win), ("step97", iS), ("distilled", iH), ("proportional", insts.index(sim.PROP)),
                       ("equal", insts.index(sim.EQ)), ("mixed", insts.index(sim.MIX)), ("step97_perm", insts.index(sim.STEP_PERM))]:
            q = d[d.inst == ii]
            if len(q):
                mo.append(dict(family=fam, env=env, slice=e["slice"], r=e.r, f=e.f, sig=e.sig, eps=e.eps, p=e.p, T=e["T"],
                               policy=nm, **{k: float(q[k].mean()) for k in ["eff", "sust", "act", "gini", "minR",
                                                                             "termR", "below_half", "worst", "spell",
                                                                             "ever_excl", "act_alive", "sust_act"]}))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(PROC, "env_optimum_summary.csv"), index=False)
    A = pd.DataFrame(amn)
    A.to_csv(os.path.join(PROC, "amnesty_decomposition.csv"), index=False)
    MO = pd.DataFrame(mo)
    MO.to_csv(os.path.join(PROC, "multiobjective.csv"), index=False)
    for f in FAMS:
        s = S[S.family == f]
        kn(**{f"re_T_{f}": int((s.status == "T").sum()), f"re_Ttie_{f}": int((s.status == "T~").sum()),
              f"re_nonT_{f}": int((s.status == "nonT").sum()), f"re_winT_{f}": int((s.winner_regime == "T").sum()),
              f"heur_rel_med_{f}": float(s.heur_rel.median()), f"step_rel_med_{f}": float(s.step_rel.median()),
              f"optimism_med_{f}": float(s.optimism.median())})
    kn(re_T=int((S.status == "T").sum()), re_Ttie=int((S.status == "T~").sum()), re_nonT=int((S.status == "nonT").sum()),
       re_winT=int((S.winner_regime == "T").sum()), re_cells=int(len(S)),
       re_all3_T_or_tie=int(S.assign(ok=S.status != "nonT").pivot(index="env", columns="family", values="ok").all(1).sum()),
       re_all3_T=int(S.assign(ok=S.status == "T").pivot(index="env", columns="family", values="ok").all(1).sum()),
       re_nonT_regimes=S[S.status == "nonT"].winner_regime.value_counts().to_dict(),
       re_nonT_short_med=float(-S[S.status == "nonT"].dT.median()) if (S.status == "nonT").any() else np.nan,
       optimism_med=float(S.optimism.median()), optimism_max=float(S.optimism.max()))
    if len(A):
        A["cls"] = np.where(A["diff"] > np.maximum(DELTA, 2 * A.se), "amnesty",
                            np.where(A["diff"] < -np.maximum(DELTA, 2 * A.se), "permanent", "equivalent"))
        A.to_csv(os.path.join(PROC, "amnesty_decomposition.csv"), index=False)
        tab = A.groupby(["family", "cls"]).size().unstack(fill_value=0)
        kn(amnesty_table=tab.to_dict(), amnesty_n=int(len(A)),
           amnesty_better=int((A.cls == "amnesty").sum()), amnesty_equiv=int((A.cls == "equivalent").sum()),
           amnesty_worse=int((A.cls == "permanent").sum()),
           amnesty_better_fpos=int(((A.cls == "amnesty") & (A.f > 0)).sum()), amnesty_n_fpos=int((A.f > 0).sum()),
           amnesty_better_f0=int(((A.cls == "amnesty") & (A.f == 0)).sum()), amnesty_n_f0=int((A.f == 0).sum()))


# ------------------------------------------------------------------------------------------------
def logit_fit(X, y, l2=1e-6):
    from scipy.optimize import minimize
    X = np.c_[np.ones(len(X)), X]

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)
    b = minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    return b


def logit_pred(b, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ b)))


def sec_eta():
    S = pd.read_csv(os.path.join(PROC, "env_optimum_summary.csv"))
    S = S[np.isfinite(S.dT)].copy()
    S["y"] = (S.dT > 0).astype(int)
    S["margin"] = S.r / (1 + S.r)
    S["eta_eff"] = S.sig_tot * np.sqrt(S.conc_T) / S.margin              # allocation-weighted (1/p_eff)
    S["eta_eps"] = np.sqrt(S.sig_tot ** 2 * S.conc_T + S.eps ** 2) / S.margin
    S["eta_emp"] = S.sdPhi_T / S.margin                                    # measured aggregate-return variability
    S.to_csv(os.path.join(PROC, "eta_table.csv"), index=False)
    out = {}
    F2 = S[S.family == "F2"]
    # pre-specified replication: logistic of the sign of dT on log eta within F2, with case bootstrap
    x = np.log(np.clip(F2.eta.to_numpy(float), 1e-3, None))[:, None]
    b = logit_fit(x, F2.y.to_numpy())
    star = float(np.exp(-b[0] / b[1])) if b[1] != 0 else np.nan
    acc = float(((logit_pred(b, x) > 0.5) == F2.y).mean())
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, len(F2), len(F2))
        bb = logit_fit(x[ii], F2.y.to_numpy()[ii])
        if bb[1] < 0:
            bs.append(np.exp(-bb[0] / bb[1]))
    kn(eta_star=star, eta_star_lo=float(np.quantile(bs, 0.025)), eta_star_hi=float(np.quantile(bs, 0.975)),
       eta_acc_F2=acc, eta_base_F2=float(max(F2.y.mean(), 1 - F2.y.mean())), eta_nF2=int(len(F2)),
       eta_F2_pos=int(F2.y.sum()))
    # descriptive counts by eta band, all families
    lo = S[S.eta < 0.5]; hi = S[S.eta > 1.0]
    kn(eta_low_n=int(len(lo)), eta_low_Tge=int((lo.dT >= -1e-9).sum()), eta_low_nonT=int((lo.status == "nonT").sum()),
       eta_high_n=int(len(hi)), eta_high_loss=int((hi.dT < 0).sum()), eta_high_F2=int((hi.family == "F2").sum()),
       eta_high_nonT=int((hi.status == "nonT").sum()))
    # grouped cross-validation: leave one slice out and leave one growth rate out, comparing predictors
    L = lambda v: np.log(np.clip(np.asarray(v, float), 1e-3, None))
    feats = {"eta": lambda d: L(d.eta)[:, None],
             "eta_eff": lambda d: L(d.eta_eff)[:, None],
             "eta_eps": lambda d: L(d.eta_eps)[:, None],
             "eta_emp": lambda d: L(d.eta_emp)[:, None],
             "raw": lambda d: np.c_[L(d.sig_tot), L(d.r), L(d.p), d.eps.to_numpy(), d.f.to_numpy(), L(d["T"])],
             "eta_plus": lambda d: np.c_[L(d.eta), d.eps.to_numpy(), d.f.to_numpy(), L(d["T"])]}
    cv = []
    for fam_set, nm in [(["F2"], "F2"), (["F1", "F2", "F3"], "pooled")]:
        d = S[S.family.isin(fam_set)].reset_index(drop=True)
        fx = pd.get_dummies(d.family).to_numpy(float)[:, 1:] if nm == "pooled" else np.zeros((len(d), 0))
        for grp in ["slice", "r"]:
            for fn, ff in feats.items():
                X = np.c_[ff(d), fx]
                pred = np.zeros(len(d))
                for gv in d[grp].unique():
                    te = (d[grp] == gv).to_numpy()
                    if d.y[~te].nunique() < 2:
                        pred[te] = d.y[~te].mean(); continue
                    bb = logit_fit(X[~te], d.y.to_numpy()[~te], l2=1e-2)
                    pred[te] = logit_pred(bb, X[te])
                yv = d.y.to_numpy()
                pos, neg = pred[yv == 1], pred[yv == 0]
                aucv = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else np.nan
                cv.append(dict(families=nm, holdout=grp, predictor=fn, acc=float(((pred > 0.5) == yv).mean()),
                               auc=aucv, base=float(max(yv.mean(), 1 - yv.mean())), n=len(d)))
    from scipy import stats as _st
    rc = []
    for f in FAMS + ["all"]:
        d = S if f == "all" else S[S.family == f]
        d = d[d.eta > 0]
        for v in ["eta", "eta_eff", "eta_eps", "eta_emp"]:
            rc.append(dict(family=f, var=v, rho=float(_st.spearmanr(d[v], d.dT).correlation), n=len(d)))
    RC = pd.DataFrame(rc)
    RC.to_csv(os.path.join(PROC, "eta_rankcorr.csv"), index=False)
    kn(eta_rankcorr=RC.to_dict("records"))
    CV = pd.DataFrame(cv)
    CV.to_csv(os.path.join(PROC, "eta_crossvalidation.csv"), index=False)
    kn(eta_cv=CV.to_dict("records"))
    # matched-eta counterexamples (identical eta, opposite sign of dT within a family)
    ce = []
    for (fam, et), d in S.groupby(["family", S.eta.round(6)]):
        if d.y.nunique() == 2 and et > 0:
            ce.append(dict(family=fam, eta=et, n=len(d), envs=";".join(map(str, d.env)), Ts=";".join(map(str, d["T"])),
                           dTs=";".join(f"{v:.4f}" for v in d.dT)))
    CE = pd.DataFrame(ce)
    CE.to_csv(os.path.join(PROC, "eta_matched_counterexamples.csv"), index=False)
    kn(eta_matched_sets=int(len(CE)))
    # per-family p_eff
    kn(peff_med={f: float((1 / S[S.family == f].conc_T).median()) for f in FAMS})


# ------------------------------------------------------------------------------------------------
def sec_threshold():
    G, I = load_grid()
    E = pd.read_csv(os.path.join(OUT, "envs.csv")).set_index("env")
    S = pd.read_csv(os.path.join(PROC, "env_optimum_summary.csv"))
    rows, prof = [], []
    for (fam, env), d in G.groupby(["family", "env"]):
        s = S[(S.family == fam) & (S.env == env)]
        if not len(s) or s.winner_regime.iloc[0] != "T":
            continue
        e = E.loc[env]
        st = d[(d.kind == 0) & (d.theta > 0) & (d.theta <= 1.0)]
        best = st.loc[st.eff.idxmax()]
        rows.append(dict(family=fam, env=env, slice=e["slice"], r=e.r, f=e.f, theta=best.theta, whi=best.whi,
                         wlo=best.wlo, am=best.amnesty, eff=best.eff, sust_frac=1 / (1 + e.r)))
        top = st.eff.max()
        for th, q in st.groupby("theta"):
            prof.append(dict(family=fam, env=env, theta=th, loss=float(top - q.eff.max())))
    TH = pd.DataFrame(rows)
    TH.to_csv(os.path.join(PROC, "threshold_location.csv"), index=False)
    PR = pd.DataFrame(prof)
    PR.to_csv(os.path.join(PROC, "threshold_profile.csv"), index=False)
    pm = PR.groupby("theta").loss.median()
    from scipy import stats
    kn(theta_med=float(TH.theta.median()), theta_q25=float(TH.theta.quantile(0.25)), theta_q75=float(TH.theta.quantile(0.75)),
       theta_min=float(TH.theta.min()), theta_max=float(TH.theta.max()), theta_n=int(len(TH)),
       theta_frac_ge09=float((TH.theta >= 0.9).mean()),
       theta_spearman_r=float(stats.spearmanr(TH.theta, TH.r).correlation),
       theta_profile_median=pm.to_dict())
    w = TH[(TH.family == "F1") & (TH.env.isin(E[E.slice == "S1_r_f"].index))].groupby("f").whi.median()
    kn(whi_by_f_F1=w.to_dict())


# ------------------------------------------------------------------------------------------------
def sec_t3():
    H = pd.read_csv(os.path.join(HUM, "human_facts.csv"))
    HC = pd.read_csv(os.path.join(HUM, "human_condition_summary.csv"))
    T3 = pd.read_csv(os.path.join(OUT, "t3.csv"))
    cond = {"E1_equal": "equal", "E1_mixed": "mixed", "E1_proportional": "proportional",
            "E2_heuristic": "heuristic", "E2_proportional": "proportional"}
    rows = []
    for tag, d0 in T3.groupby(["family", "integer"]):
        fam, integer = tag
        for _, f in H.iterrows():
            a = d0[d0.mech == cond[f.cond_a]].sort_values("rep")[f.outcome].to_numpy()
            b = d0[d0.mech == cond[f.cond_b]].sort_values("rep")[f.outcome].to_numpy()
            n = min(len(a), len(b))
            dif = a[:n] - b[:n]
            m, se = dif.mean(), dif.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            row = dict(family=fam, integer=integer, id=f.id, fact=f.fact, calibrated_on=f.calibrated_on, nrep=n,
                       sim_diff=m, sim_se=se, human_diff=f.human_diff, human_lo=f.lo, human_hi=f.hi,
                       human_supported=f.human_sign_supported)
            for k_se, tol in itertools.product([1, 2, 3], [0.0, 0.005, 0.01, 0.02]):
                thr = max(k_se * (se if np.isfinite(se) else 0), tol)
                row[f"cls_{k_se}se_{tol}"] = "yes" if m > thr else ("no" if m < -thr else "tie")
            row["cls"] = row["cls_2se_0.005"]
            row["in_human_ci"] = bool(f.lo <= m <= f.hi)
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(PROC, "t3_contrasts.csv"), index=False)
    # quantitative level comparison: simulated condition means versus human estimates
    lv = []
    for tag, d0 in T3.groupby(["family", "integer"]):
        for c_, mech in cond.items():
            if c_ == "E2_proportional":
                continue
            for s in ["eff", "gini", "sust", "all4", "act"]:
                h = HC[(HC.cond == c_) & (HC.stat == s)]
                if not len(h):
                    continue
                v = d0[d0.mech == mech][s]
                lv.append(dict(family=tag[0], integer=tag[1], cond=c_, stat=s, sim=float(v.mean()),
                               human=float(h["mean"].iloc[0]), human_lo=float(h.lo.iloc[0]), human_hi=float(h.hi.iloc[0])))
    LV = pd.DataFrame(lv)
    LV["abs_err"] = (LV.sim - LV.human).abs()
    LV["in_ci"] = (LV.sim >= LV.human_lo) & (LV.sim <= LV.human_hi)
    LV.to_csv(os.path.join(PROC, "t3_levels.csv"), index=False)
    for (fam, integer), d in R.groupby(["family", "integer"]):
        key = f"t3_{fam}_{'int' if integer else 'cont'}"
        kn(**{key + "_yes": int((d.cls == "yes").sum()), key + "_no": int((d.cls == "no").sum()),
              key + "_tie": int((d.cls == "tie").sum()),
              key + "_heldout_yes": int(((d.cls == "yes") & (~d.calibrated_on)).sum()),
              key + "_inci": int(d.in_human_ci.sum())})
    for (fam, integer), d in LV.groupby(["family", "integer"]):
        kn(**{f"lv_{fam}_{'int' if integer else 'cont'}_mae": float(d.abs_err.mean()),
              f"lv_{fam}_{'int' if integer else 'cont'}_inci": int(d.in_ci.sum()), "lv_n": int(len(d))})
    kn(human_facts=H[["id", "human_diff", "lo", "hi", "perm_p", "human_sign_supported"]].to_dict("records"),
       human_facts_supported=int(H.human_sign_supported.sum()))
    # threshold-rule sensitivity of the family scores
    sens = {}
    for col in [c for c in R.columns if c.startswith("cls_")]:
        sens[col] = R[R.integer == 0].groupby("family")[col].apply(lambda q: int((q == "yes").sum())).to_dict()
    kn(t3_threshold_sensitivity=sens)
    rec = json.load(open(os.path.join(HUM, "human_reciprocation.json")))
    kn(human_recip=rec)
    import shutil as _sh
    for fn in ["human_facts.csv", "human_condition_summary.csv", "calibration_targets.csv", "human_reciprocation.json"]:
        _sh.copy(os.path.join(HUM, fn), os.path.join(PROC, fn))
    _sh.copy(os.path.join(OUT, "calib.json"), os.path.join(PROC, "calib.json"))
    cg = pd.read_csv(os.path.join(OUT, "calib_grid.csv.gz"))
    cg.sort_values("loss").head(2000).to_csv(os.path.join(PROC, "calib_grid_top2000.csv"), index=False)
    c = json.load(open(os.path.join(OUT, "calib.json")))
    kn(calib_best=c["best"], calib_micro=c["micro"], calib_chi2dof=c["chi2_dof"], calib_plaus_n=len(c["plausible"]),
       calib_plaus_cov=c["plausible_coverage"], calib_best_fit_values=c["best_fit_values"],
       calib_profile_lam=c["profile_lam"], calib_profile_gam=c["profile_gam"], calib_profile_sdint=c["profile_sd_int"])


# ------------------------------------------------------------------------------------------------
def es_class(rw, comp=1.0, cont_lo=0.3, rise=0.2, prop=0.2):
    if rw["w_lt05"] > comp:
        return "compensatory"
    if rw["w_lt05"] < cont_lo and rw["w_ge096"] - rw["w_lt05"] >= rise:
        return "contingent"
    if max(rw["w_lt05"], rw["w_ge096"]) < prop:
        return "proportional"
    return "other"


def sec_es():
    S = pd.read_json(os.path.join(OUT, "es_summary.json"))
    S["r"] = S.r.round(3); S["f"] = S.f.round(3)
    for k in ["w_lt05", "w_ge096"]:
        S[k] = S[k].fillna(S[k].median())
    S["cls"] = S.apply(es_class, axis=1)
    stab = []
    for comp, cl, rise in [(1.0, 0.3, 0.2), (0.9, 0.25, 0.15), (1.1, 0.35, 0.25), (1.2, 0.2, 0.3)]:
        q = S.apply(lambda rw: es_class(rw, comp, cl, rise), axis=1)
        stab.append(dict(comp=comp, cont_lo=cl, rise=rise, agree=float((q == S.cls).mean()), **q.value_counts().to_dict()))
    pd.DataFrame(stab).to_csv(os.path.join(PROC, "es_class_stability.csv"), index=False)
    S.drop(columns=["wtab"]).to_csv(os.path.join(PROC, "es_summary.csv"), index=False)
    import shutil as _sh
    for fn in ["es_curves.csv", "es_hist.csv", "es_curves_lam1.csv", "es_hist_lam1.csv"]:
        if os.path.exists(os.path.join(OUT, fn)):
            _sh.copy(os.path.join(OUT, fn), os.path.join(PROC, fn))
    X = pd.read_csv(os.path.join(OUT, "cross_eval.csv.gz"))
    X["r"] = X.r.round(3); X["f"] = X.f.round(3)
    X["fit"] = X.tag.str.split("|").str[1].fillna("best")
    X["ptag"] = X.tag.str.split("|").str[0]
    ad = X[(X["mode"] == "adapt") & (X.fit == "best")]
    agg = ad.groupby(["kind", "key", "ptag", "r", "f", "family"]).eff.agg(["mean", "std", "count"]).reset_index()
    step = agg[agg.key == "step97"][["r", "f", "family", "mean"]].rename(columns={"mean": "step"})
    P = agg[agg.kind.isin(["planner", "table"]) & agg.key.str.match(r"(es|tab)_s")].merge(step, on=["r", "f", "family"])
    P["seed"] = P.key.str[-1].astype(int)
    P["gap"] = P["mean"] - P.step
    P = P.merge(S[["tag", "r", "f", "seed", "cls", "R2", "med_resid_frac", "frac_discarded", "frac_negative_projection"]]
                .rename(columns={"tag": "ptag"}), on=["ptag", "r", "f", "seed"], how="left")
    P.to_csv(os.path.join(PROC, "es_crosseval_paired.csv"), index=False)
    pl = P[(P.kind == "planner") & (P.ptag == "best")]
    W = pl.pivot_table(index=["r", "f", "seed", "cls"], columns="family", values="gap").reset_index()
    W["worst_paired"] = W[["F1", "F3"]].min(1)
    W.to_csv(os.path.join(PROC, "es_planner_gaps.csv"), index=False)
    cc = S[S.tag == "best"].cls.value_counts().to_dict()
    lv = pl.pivot_table(index=["r", "f", "seed", "cls"], columns="family", values="mean").reset_index()
    kn(es_n=int((S.tag == "best").sum()), es_classes=cc, es_classes_micro=S[S.tag == "micro"].cls.value_counts().to_dict(),
       es_win_F2=int((W.F2 > 0).sum()), es_lose_F1=int((W.F1 < 0).sum()), es_lose_F3=int((W.F3 < 0).sum()),
       es_class_means={k: {f: float(v[f].mean()) for f in FAMS} for k, v in lv.groupby("cls")},
       es_class_gap_med={k: {f: float(v[f].median()) for f in FAMS} for k, v in W.groupby("cls")},
       es_R2_med=float(S[S.tag == "best"].R2.median()), es_resid_med=float(S[S.tag == "best"].med_resid_frac.median()),
       es_discard_med=float(S[S.tag == "best"].frac_discarded.median()),
       es_negproj_med=float(S[S.tag == "best"].frac_negative_projection.median()),
       es_class_stability=stab, step_means={f: float(step[step.family == f]["step"].mean()) for f in FAMS})
    # paired within-environment comparison of compensatory and contingent planners
    pe = []
    for (r, f), d in W.groupby(["r", "f"]):
        for cl in ["compensatory", "contingent"]:
            q = d[d.cls == cl]
            if len(q):
                pe.append(dict(r=r, f=f, cls=cl, n=len(q), **{fam: float(q[fam].mean()) for fam in FAMS}))
    pd.DataFrame(pe).to_csv(os.path.join(PROC, "es_paired_by_env.csv"), index=False)
    # distilled executable tables versus the networks
    tb = P[(P.kind == "table") & (P.ptag == "best")][["r", "f", "seed", "family", "mean"]].rename(columns={"mean": "table"})
    nt = pl[["r", "f", "seed", "family", "mean", "cls"]].rename(columns={"mean": "net"})
    TN = nt.merge(tb, on=["r", "f", "seed", "family"])
    TN.to_csv(os.path.join(PROC, "es_table_vs_net.csv"), index=False)
    kn(table_vs_net_mad={f: float((TN[TN.family == f].table - TN[TN.family == f].net).abs().median()) for f in FAMS})
    # adaptation controls: recipients pre-adapted to the step rule, then frozen or adapting
    ct = X[X["mode"].isin(["pre_frozen", "pre_adapt"]) & (X.kind == "planner") & (X.ptag == "best")]
    ct = ct.merge(S[S.tag == "best"][["r", "f", "seed", "cls"]].assign(key=lambda q: "es_s" + q.seed.astype(str)),
                  on=["r", "f", "key"])
    ctrl = ct.groupby(["cls", "family", "mode"]).eff.mean().unstack().reset_index()
    ctrl.to_csv(os.path.join(PROC, "es_adaptation_control.csv"), index=False)
    kn(es_control=ctrl.to_dict("records"))
    fx = X[(X["mode"].isin(["pre_frozen", "pre_adapt"])) & (X.key == "step97")].groupby(["family", "mode"]).eff.mean()
    kn(es_control_step=fx.reset_index().to_dict("records"))
    # calibration sensitivity of the in-model (F2) comparison
    cs = X[(X["mode"] == "adapt") & (X.family == "F2") & (X.ptag == "best")]
    cs = cs.groupby(["kind", "key", "fit", "r", "f"]).eff.mean().reset_index()
    stp = cs[cs.key == "step97"][["fit", "r", "f", "eff"]].rename(columns={"eff": "step"})
    pcs = cs[cs.kind == "planner"].merge(stp, on=["fit", "r", "f"])
    pcs["seed"] = pcs.key.str[-1].astype(int)
    pcs = pcs.merge(S[S.tag == "best"][["r", "f", "seed", "cls"]], on=["r", "f", "seed"])
    pcs["gap"] = pcs.eff - pcs.step
    pcs.to_csv(os.path.join(PROC, "es_calibration_sensitivity.csv"), index=False)
    kn(es_calsens=pcs.groupby(["fit", "cls"]).gap.median().unstack().to_dict())
    mi = X[(X["mode"] == "adapt") & (X.ptag == "micro") & (X.kind == "planner") & (X.fit == "best")]
    mi_cls = S[S.tag == "micro"][["r", "f", "seed", "cls"]]
    H = pd.read_csv(os.path.join(OUT, "es_hist.csv"))
    last = H[H.it >= H.it.max() - 200].groupby(["tag", "r", "f", "seed"]).eff.agg(lambda v: v.iloc[-1] - v.iloc[0])
    kn(es_last200_gain_med=float(last.median()), es_last200_gain_max=float(last.max()))


# ------------------------------------------------------------------------------------------------
def sec_robust():
    E = pd.read_csv(os.path.join(OUT, "envs_robust.csv")).set_index("env")
    c = json.load(open(os.path.join(OUT, "calib.json")))
    G, I = load_grid("robust", "robust")
    rows = []
    for (fam, env), d in G.groupby(["family", "env"]):
        e = E.loc[env]
        d = d.sort_values("eff", ascending=False)
        b = d.iloc[0]
        bt = d[d.regime == "T"].eff.max()
        bn = d[d.regime != "T"].eff.max()
        rows.append(dict(family=fam, env=env, slice=e["slice"], r=e.r, f=e.f, sig=e.sig, eps=e.eps, p=e.p, T=e["T"],
                         best_regime=b.regime, best_theta=b.theta, best_eff=b.eff, dT=bt - bn,
                         best_am=int(b.amnesty), whi=b.whi, wlo=b.wlo))
    RB = pd.DataFrame(rows)
    RB.to_csv(os.path.join(PROC, "robust_summary.csv"), index=False)
    tab = RB.groupby(["slice", "family"]).apply(lambda q: f"{int((q.best_regime == 'T').sum())}/{len(q)}").unstack()
    tab.to_csv(os.path.join(PROC, "robust_table.csv"))
    kn(robust_table=tab.to_dict(), robust_T=int((RB.best_regime == "T").sum()), robust_n=int(len(RB)),
       bh_theta_med=float(RB[(RB.slice == "X1_bevholt") & (RB.best_regime == "T")].best_theta.median())
       if ((RB.slice == "X1_bevholt") & (RB.best_regime == "T")).any() else np.nan)


def sec_sens():
    X = pd.read_csv(os.path.join(OUT, "sensitivity.csv.gz"))
    I = insts_table()
    rg = I.set_index("inst").regime
    f3 = X[X.kind == "f3var"]
    insts = sim.inst_grid(True) + [sim.MIX]
    iS = insts.index(sim.STEP)
    rows = []
    for (v, env), d in f3.groupby(["variant", "env"]):
        m = d.groupby("inst").eff.mean()
        mz = d.groupby("inst").fz_eff.mean()
        isT = np.array([rg[i] == "T" for i in m.index])
        rows.append(dict(variant=v, env=env, best_regime=rg[m.idxmax()], T_minus_nonT=float(m[isT].max() - m[~isT].max()),
                         T_minus_nonT_frozen=float(mz[isT].max() - mz[~isT].max()),
                         step_minus_bestother=float(m[iS] - m.drop(iS).max()),
                         best_regime_frozen=rg[mz.idxmax()], step_minus_bestother_frozen=float(mz[iS] - mz.drop(iS).max())))
    F3V = pd.DataFrame(rows)
    F3V.to_csv(os.path.join(PROC, "f3_variant_sensitivity.csv"), index=False)
    kn(f3var_T_best=int((F3V.T_minus_nonT >= 0).sum()), f3var_T_best_frozen=int((F3V.T_minus_nonT_frozen >= 0).sum()),
       f3var_T_within=int((F3V.T_minus_nonT >= -DELTA).sum()),
       f3var_by_variant=F3V.groupby("variant").apply(lambda q: f"{int((q.T_minus_nonT >= 0).sum())}/{len(q)}").to_dict(),
       f3var_step_best=int((F3V.step_minus_bestother >= 0).sum()), f3var_n=int(len(F3V)),
       f3var_step_best_frozen=int((F3V.step_minus_bestother_frozen >= 0).sum()))
    cs = X[X.kind == "calsens"]
    out = []
    for (v, env), d in cs.groupby(["variant", "env"]):
        m = d.set_index("inst").eff
        out.append(dict(fit=v, env=env, best_regime=rg[m.idxmax()],
                        dT=float(m[rg[m.index] == "T"].max() - m[rg[m.index] != "T"].max())))
    CS = pd.DataFrame(out)
    CS.to_csv(os.path.join(PROC, "calibration_phase_sensitivity.csv"), index=False)
    kn(calsens_T={k: int((d.best_regime == "T").sum()) for k, d in CS.groupby("fit")})


def sec_field():
    R = pd.read_csv(os.path.join(FLD, "field_results.csv"))
    R.to_csv(os.path.join(PROC, "field_results.csv"), index=False)
    import shutil as _sh
    _sh.copy(os.path.join(FLD, "class_scores_all.json"), os.path.join(PROC, "class_scores_all.json"))
    _sh.copy(os.path.join(FLD, "baggio2016_cases_reconstructed.csv"), os.path.join(PROC, "baggio2016_cases_reconstructed.csv"))
    kn(field=R.to_dict("records"))
    for nm in ["field_outcome_dependent_completions.csv", "class_decision_table.csv", "coding_agreement_A2.csv",
               "reconstruction_audit_A4.csv", "field_misclassification.csv", "field_sensitivity_mapping_score.csv",
               "class_scores.csv"]:
        fn = os.path.join(FLD, nm)
        if os.path.exists(fn):
            pd.read_csv(fn).to_csv(os.path.join(PROC, nm), index=False)
    au = pd.read_csv(os.path.join(FLD, "reconstruction_audit_A4.csv"))
    kn(audit_rows=int(len(au)), audit_unique=int(au.unique.sum()), audit_mixed=int(au.mixed.sum()))
    sc = pd.read_csv(os.path.join(FLD, "class_scores.csv"))
    kn(class_scores=dict(zip(sc.cls, sc.score)))
    cm = pd.read_csv(os.path.join(FLD, "field_outcome_dependent_completions.csv"))
    kn(odc_rd_min=float(cm.rd.min()), odc_rd_max=float(cm.rd.max()), odc_auc_min=float(cm.H4_auc.min()),
       odc_auc_max=float(cm.H4_auc.max()))
    ms = pd.read_csv(os.path.join(FLD, "field_misclassification.csv"))
    kn(mis_rd_posfrac_min=float(ms.rd_pos_frac.min()))
    ss = pd.read_csv(os.path.join(FLD, "field_sensitivity_mapping_score.csv"))
    full = ss[ss.coding != "complete cases"]
    kn(sens_auc_min=float(full.auc.min()), sens_auc_max=float(full.auc.max()),
       sens_amnesty_gt_perm_frac=float(ss.drop_duplicates("score").amnesty_gt_permanent.mean()))
    kn(complete_vs_incomplete=json.load(open(os.path.join(FLD, "field_complete_vs_incomplete.json"))))


def sec_alt():
    """Calibration dependence: F2 phase diagram under the lam = 1 and submitted-version fits; planners trained
    against the lam = 1 fit and their transfer."""
    E = pd.read_csv(os.path.join(OUT, "envs.csv")).set_index("env")
    I = insts_table()
    for tag, sd_int in [("gridlam1", None), ("gridv1", 0.3)]:
        fn = os.path.join(OUT, f"{tag}_F2.csv.gz")
        if not os.path.exists(fn):
            continue
        c = json.load(open(os.path.join(OUT, "calib.json")))
        if sd_int is None:
            sd_int = [x for x in c["profile_lam"] if abs(x["lam"] - 1) < 1e-9][0]["sd_int"]
        G = pd.read_csv(fn).merge(I, on="inst")
        C = cell_summary(G, E, sd_int)
        C.to_csv(os.path.join(PROC, f"env_optimum_summary_{tag}.csv"), index=False)
        y = (C.dT > 0).astype(int).to_numpy()
        x = np.log(np.clip(C.eta.to_numpy(float), 1e-3, None))[:, None]
        star = np.nan
        if 0 < y.sum() < len(y):
            b = logit_fit(x, y)
            star = float(np.exp(-b[0] / b[1])) if b[1] < 0 else np.nan
        kn(**{f"{tag}_T": int((C.best_regime == "T").sum()), f"{tag}_n": int(len(C)), f"{tag}_eta_star": star,
              f"{tag}_regimes": C.best_regime.value_counts().to_dict()})
    fs = os.path.join(OUT, "es_summary_lam1.json")
    if os.path.exists(fs):
        S = pd.read_json(fs)
        S["r"] = S.r.round(3); S["f"] = S.f.round(3)
        for k in ["w_lt05", "w_ge096"]:
            S[k] = S[k].fillna(S[k].median())
        S["cls"] = S.apply(es_class, axis=1)
        S.drop(columns=["wtab"]).to_csv(os.path.join(PROC, "es_summary_lam1.csv"), index=False)
        X = pd.read_csv(os.path.join(OUT, "cross_eval_lam1.csv.gz"))
        X["r"] = X.r.round(3); X["f"] = X.f.round(3)
        X["fit"] = X.tag.str.split("|").str[1].fillna("best")
        ad = X[X["mode"] == "adapt"]
        g = ad.groupby(["kind", "key", "r", "f", "family", "fit"]).eff.mean().reset_index()
        pl = g[g.kind == "planner"].copy()
        pl["seed"] = pl.key.str[-1].astype(int)
        pl = pl.merge(S[["r", "f", "seed", "cls"]], on=["r", "f", "seed"])
        stl = g[(g.kind == "inst") & (g.key == "step97")][["r", "f", "eff"]].rename(columns={"eff": "step_lam1"})
        pl.to_csv(os.path.join(PROC, "es_lam1_eval.csv"), index=False)
        W = pl.pivot_table(index=["r", "f", "seed", "cls"], columns=["family", "fit"], values="eff")
        W.columns = [f"{a}_{b}" for a, b in W.columns]
        W = W.reset_index().merge(stl, on=["r", "f"], how="left")
        W.to_csv(os.path.join(PROC, "es_lam1_wide.csv"), index=False)
        Sb = pd.read_csv(os.path.join(PROC, "es_crosseval_paired.csv")) if os.path.exists(os.path.join(PROC, "es_crosseval_paired.csv")) else None
        out = dict(lam1_classes=S.cls.value_counts().to_dict(), lam1_n=int(len(S)))
        for cl, d in W.groupby("cls"):
            out[f"lam1_{cl}_means"] = {k: float(d[k].mean()) for k in d.columns if k.startswith("F") or k == "step_lam1"}
        ct = X[X["mode"].isin(["pre_frozen", "pre_adapt"])].copy()
        ct["seed"] = ct.key.str[-1].astype(int)
        ct = ct.merge(S[["r", "f", "seed", "cls"]], on=["r", "f", "seed"])
        out["lam1_control"] = ct.groupby(["cls", "family", "mode"]).eff.mean().reset_index().to_dict("records")
        ct.groupby(["cls", "family", "mode"]).eff.mean().unstack().reset_index().to_csv(
            os.path.join(PROC, "es_lam1_control.csv"), index=False)
        kn(**out)


SECTIONS = dict(theory=sec_theory, phase=sec_phase, reeval=sec_reeval, eta=sec_eta, threshold=sec_threshold,
                t3=sec_t3, es=sec_es, robust=sec_robust, sens=sec_sens, field=sec_field, alt=sec_alt)

if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if which == ["all"]:
        which = list(SECTIONS)
    for s in which:
        try:
            SECTIONS[s]()
            print("section", s, "ok", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("section", s, "FAILED", ex, flush=True)
        save_kn()
