"""Human evidence from the public per-round records of Koster, Pislar et al. (2025): game-level outcomes,
calibration targets with group-bootstrap uncertainty, the eleven ordinal facts with human estimates and
intervals, and individual-level tests of the reciprocation equation used by the F2 family.

usage: python3 human.py human_rounds.csv.gz out/
"""
import sys, os, json, zlib
import numpy as np
import pandas as pd

R0, R_GROW, T0 = 200.0, 0.4, 40
CEIL = T0 * R0 * R_GROW / (1 + R_GROW)
MECH = {"Equal Baseline Exp 1": "E1_equal", "Mixed Baseline Exp 1": "E1_mixed",
        "Proportional Baseline Exp 1": "E1_proportional", "RL Agent (M1) Exp 1": "E1_M1",
        "Interpolating Baseline Exp 2": "E2_heuristic", "Proportional Baseline Exp 2": "E2_proportional",
        "RL Agent (M1) Exp 2": "E2_M1", "Interpolating Baseline Exp 3": "E3_heuristic", "RL Agent (M2) Exp 3": "E3_M2"}


def gini(x):
    x = np.sort(np.asarray(x, float))
    n, s = len(x), x.sum()
    if s <= 1e-12:
        return 0.0
    return 2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * s) - (n + 1.0) / n


def game_table(h):
    h = h[h.mech_name_by_player.isin(MECH)].copy()
    h["cond"] = h.mech_name_by_player.map(MECH)
    h["game"] = h.mech_name_by_player + "|" + h.launch_id.astype(str)
    rows, prow = [], []
    for g, d in h.groupby("game", sort=False):
        d = d.sort_values("round_id")
        pool = d["mechanism_observation.pool"].to_numpy(float)
        nxt = d["next_environment_state.pool"].to_numpy(float)
        off = d[[f"offer_{i}" for i in range(4)]].to_numpy(float)
        rec = d[[f"player_action_{i}" for i in range(4)]].to_numpy(float)
        rec = np.nan_to_num(rec)
        off = np.nan_to_num(off)
        kept = np.clip(off - rec, 0, None)
        T = len(d)
        alive = (pool >= 1.0)
        sust = bool(alive.all() and nxt[-1] >= 1.0)
        act_r = ((off >= 1.0).sum(1) / 4.0) * alive
        all4 = bool(sust and (off[-1] >= 1.0).all())
        pay = kept.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fr = np.where(off >= 1.0, rec / np.maximum(off, 1e-9), np.nan)
        rows.append(dict(game=g, cond=d.cond.iloc[0], T=T, sust=float(sust), all4=float(all4),
                         gini=gini(pay), eff=kept.sum() / CEIL, act=act_r[:T0].sum() / T0,
                         phi=float(np.nanmean(fr)) if np.isfinite(fr).any() else np.nan,
                         minpool=float(min(pool.min(), nxt[-1])) / R0, collapse_round=int(np.argmax(~alive)) + 1 if not alive.all() else 0))
        # player-round records for the individual-level reciprocation analysis
        for t in range(1, T):
            if pool[t] < 1.0:
                break
            for i in range(4):
                if off[t, i] >= 1.0 and off[t - 1, i] >= 0.0:
                    others = [j for j in range(4) if j != i and off[t - 1, j] >= 1.0]
                    if not others:
                        continue
                    ob = float(np.mean([rec[t - 1, j] / off[t - 1, j] for j in others]))
                    own_prev = rec[t - 1, i] / off[t - 1, i] if off[t - 1, i] >= 1.0 else np.nan
                    rho = off[t, i] / (pool[t] / 4.0)
                    prow.append(dict(game=g, cond=d.cond.iloc[0], pid=f"{g}|{i}", t=t + 1, y=rec[t, i] / off[t, i],
                                     obar=ob, d=float(np.clip(rho - 1.0, -1, 1)), rho=rho, own_prev=own_prev))
    return pd.DataFrame(rows), pd.DataFrame(prow)


def fe_ols(df, ycol, xcols, gcol, ccol):
    """Within (fixed-effect) OLS with cluster-robust (CR1) standard errors."""
    d = df.dropna(subset=[ycol] + xcols).copy()
    for c in [ycol] + xcols:
        d[c + "_w"] = d[c] - d.groupby(gcol)[c].transform("mean")
    X = d[[c + "_w" for c in xcols]].to_numpy()
    y = d[ycol + "_w"].to_numpy()
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    u = y - X @ b
    cl = d[ccol].to_numpy()
    meat = np.zeros((len(xcols), len(xcols)))
    for k in np.unique(cl):
        s = (X[cl == k] * u[cl == k, None]).sum(0)
        meat += np.outer(s, s)
    G, n, kx = len(np.unique(cl)), len(y), len(xcols)
    adj = G / (G - 1) * (n - 1) / (n - kx - d[gcol].nunique())
    V = np.linalg.inv(XtX) @ meat @ np.linalg.inv(XtX) * max(adj, 1.0)
    return dict(zip(xcols, b)), dict(zip(xcols, np.sqrt(np.diag(V)))), int(n), int(G)


FACTS = [  # id, label, outcome, cond_a, cond_b (fact asserts a > b), source, used in calibration
    ("O1", "Surplus: mixed > equal", "eff", "E1_mixed", "E1_equal", "Exp. 1", False),
    ("O2", "Surplus: proportional > equal", "eff", "E1_proportional", "E1_equal", "Exp. 1", False),
    ("O3", "Surplus: proportional > mixed", "eff", "E1_proportional", "E1_mixed", "Exp. 1", False),
    ("O4", "Gini: proportional > mixed", "gini", "E1_proportional", "E1_mixed", "Exp. 1", True),
    ("O5", "Gini: proportional > equal", "gini", "E1_proportional", "E1_equal", "Exp. 1", True),
    ("O6", "All four active at end: mixed > equal", "all4", "E1_mixed", "E1_equal", "Exp. 1", True),
    ("O7", "All four active at end: equal > proportional", "all4", "E1_equal", "E1_proportional", "Exp. 1", True),
    ("O8", "Pool sustained: proportional > mixed", "sust", "E1_proportional", "E1_mixed", "Exp. 1", True),
    ("O9", "Surplus: distilled rule > proportional", "eff", "E2_heuristic", "E2_proportional", "Exp. 2", False),
    ("O10", "Gini: proportional > distilled rule", "gini", "E2_proportional", "E2_heuristic", "Exp. 2", False),
    ("O11", "Active share: distilled rule > proportional", "act", "E2_heuristic", "E2_proportional", "Exp. 2", False),
]
TARGET_KEYS = [(m, s) for m in ["equal", "mixed", "proportional"] for s in ["sust", "all4", "gini"]]


def main(path, out):
    os.makedirs(out, exist_ok=True)
    h = pd.read_csv(path, low_memory=False)
    G, P = game_table(h)
    G.to_csv(os.path.join(out, "human_games.csv"), index=False, float_format="%.6f")
    rng = np.random.default_rng(20261019)
    B = 2000
    by = {c: d for c, d in G.groupby("cond")}
    summ = []
    for c, d in by.items():
        for s in ["eff", "gini", "sust", "all4", "act", "phi"]:
            v = d[s].to_numpy(float)
            bs = np.array([np.nanmean(rng.choice(v, len(v))) for _ in range(B)])
            summ.append(dict(cond=c, stat=s, n=len(v), mean=float(np.nanmean(v)), se=float(bs.std(ddof=1)),
                             lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975))))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(out, "human_condition_summary.csv"), index=False, float_format="%.6f")
    # calibration targets and bootstrap replicates (resampling groups within condition)
    tv = []
    boot = np.zeros((1000, 9))
    for j, (m, s) in enumerate(TARGET_KEYS):
        v = by["E1_" + m][s].to_numpy(float)
        tv.append(dict(mech=m, stat=s, n=len(v), target=float(v.mean()),
                       se=float(S[(S.cond == "E1_" + m) & (S.stat == s)].se.iloc[0])))
    rb = np.random.default_rng(7)
    for b in range(1000):
        for m in ["equal", "mixed", "proportional"]:
            d = by["E1_" + m]
            idx = rb.integers(0, len(d), len(d))
            for s in ["sust", "all4", "gini"]:
                boot[b, TARGET_KEYS.index((m, s))] = d[s].to_numpy(float)[idx].mean()
    np.save(os.path.join(out, "targets_boot.npy"), boot)
    old = {("equal", "sust"): 0.05, ("equal", "all4"): 0.05, ("equal", "gini"): 0.10, ("mixed", "sust"): 0.30,
           ("mixed", "all4"): 0.30, ("mixed", "gini"): 0.10, ("proportional", "sust"): 0.60,
           ("proportional", "all4"): 0.00, ("proportional", "gini"): 0.38}
    for row in tv:
        row["approx_published_target_v1"] = old[(row["mech"], row["stat"])]
    pd.DataFrame(tv).to_csv(os.path.join(out, "calibration_targets.csv"), index=False, float_format="%.6f")
    # ordinal facts with human estimates (difference a - b), bootstrap CI and permutation P
    fr = []
    for fid, lab, s, a, b_, src, cal in FACTS:
        va, vb = by[a][s].to_numpy(float), by[b_][s].to_numpy(float)
        diff = va.mean() - vb.mean()
        bs = np.array([rng.choice(va, len(va)).mean() - rng.choice(vb, len(vb)).mean() for _ in range(B)])
        pool_ = np.concatenate([va, vb])
        perm = np.array([(lambda q: q[:len(va)].mean() - q[len(va):].mean())(rng.permutation(pool_))
                         for _ in range(B)])
        pp = (1 + np.sum(np.abs(perm) >= abs(diff) - 1e-12)) / (B + 1)
        fr.append(dict(id=fid, fact=lab, outcome=s, cond_a=a, cond_b=b_, source=src, calibrated_on=cal,
                       n_a=len(va), n_b=len(vb), mean_a=va.mean(), mean_b=vb.mean(), human_diff=diff,
                       lo=float(np.quantile(bs, 0.025)), hi=float(np.quantile(bs, 0.975)), perm_p=pp,
                       human_sign_supported=bool(np.quantile(bs, 0.025) > 0)))
    pd.DataFrame(fr).to_csv(os.path.join(out, "human_facts.csv"), index=False, float_format="%.6f")
    # individual-level reciprocation analysis (F2 equation): y_it = a_i + lam obar_-i,t-1 + gam d_it
    P.to_csv(os.path.join(out, "human_player_rounds.csv.gz"), index=False, float_format="%.5f")
    res = {}
    for name, sub in [("all_exp1to3", P), ("exp1_baselines", P[P.cond.isin(["E1_equal", "E1_mixed", "E1_proportional"])]),
                      ("exp1_proportional", P[P.cond == "E1_proportional"]), ("exp1_mixed", P[P.cond == "E1_mixed"]),
                      ("exp2_heuristic", P[P.cond == "E2_heuristic"])]:
        b1, s1, n1, g1 = fe_ols(sub, "y", ["obar", "d"], "pid", "game")
        b2, s2, n2, g2 = fe_ols(sub, "y", ["d"], "pid", "game")
        res[name] = dict(lam=b1["obar"], lam_se=s1["obar"], gam=b1["d"], gam_se=s1["d"], n=n1, games=g1,
                         gam_only=b2["d"], gam_only_se=s2["d"])
    # between-player (pooled) association for comparison: pooled OLS of y on d with cluster SE
    sub = P.dropna(subset=["y", "d", "obar"]).copy()
    X = np.c_[np.ones(len(sub)), sub.obar, sub.d]
    bb = np.linalg.lstsq(X, sub.y.to_numpy(), rcond=None)[0]
    res["pooled_ols"] = dict(const=bb[0], lam=bb[1], gam=bb[2], n=len(sub))
    json.dump(res, open(os.path.join(out, "human_reciprocation.json"), "w"), indent=1)
    print(S[S.cond.str.startswith("E1")].pivot(index="cond", columns="stat", values="mean").round(3))
    print(pd.DataFrame(fr)[["id", "human_diff", "lo", "hi", "perm_p"]].round(3))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
