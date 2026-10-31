"""Revision driver: all simulation stages for the revised manuscript (run on the Linux server).

usage: python3 run_rev.py <stage> [nproc]
stages: calib theory grid reeval robust t3 h4 es cross
Outputs go to out/ (raw).  process.py turns them into the tables, numbers and figure inputs.
Seeds are derived from CRC32 hashes of task keys, so every stage is bit-reproducible.
"""
import sys, os, json, time, zlib, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

OUT = os.environ.get("REV_OUT", "out")
HUM = os.environ.get("REV_HUMAN", "out_human")
os.makedirs(OUT, exist_ok=True)
R7 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
M = sim.METRICS


def seed_of(*key):
    return zlib.crc32("|".join(map(str, key)).encode()) & 0x3FFFFFFF


def envs_main():
    E = []
    for r in R7:
        for f in [0.0, 0.1, 0.2, 0.3, 0.4]:
            E.append(dict(slice="S1_r_f", r=r, p=4, T=40, sig=0.05, eps=0.0, f=f))
        for sig in [0.0, 0.1, 0.2]:
            E.append(dict(slice="S2_r_noise", r=r, p=4, T=40, sig=sig, eps=0.0, f=0.2))
        for p in [3, 8, 16]:
            E.append(dict(slice="S3_r_p", r=r, p=p, T=40, sig=0.05, eps=0.0, f=0.2))
        for eps in [0.1, 0.2, 0.3]:
            E.append(dict(slice="S4_r_growthnoise", r=r, p=4, T=40, sig=0.05, eps=eps, f=0.2))
        for T in [20, 80]:
            E.append(dict(slice="S5_r_horizon", r=r, p=4, T=T, sig=0.05, eps=0.0, f=0.2))
    for j, e in enumerate(E):
        e.update(env=j, rho=0.0, gk=0, obs=0.0, delay=0, crec=0.0, integer=0)
    return E


def envs_robust():
    """Structural and interaction tests (not part of the 112-environment phase diagram)."""
    E = []
    base = dict(p=4, T=40, sig=0.05, eps=0.0, f=0.2, rho=0.0, gk=0, obs=0.0, delay=0, crec=0.0, integer=0)
    for r in R7:
        for f in [0.0, 0.2]:
            E.append(dict(base, slice="X1_bevholt", r=r, f=f, gk=1))
            E.append(dict(base, slice="X5_integer", r=r, f=f, integer=1))
        for obs in [0.05, 0.1, 0.2]:
            E.append(dict(base, slice="X2_obsnoise", r=r, obs=obs))
        E.append(dict(base, slice="X2_obsdelay", r=r, delay=1))
        E.append(dict(base, slice="X3_recordnoise", r=r, crec=0.2))
        E.append(dict(base, slice="X4_corrnoise", r=r, sig=0.1, rho=0.5))
    for r in [0.2, 0.4, 0.8]:
        for sig in [0.1, 0.2]:
            for eps in [0.1, 0.3]:
                E.append(dict(base, slice="X6_sig_x_eps", r=r, sig=sig, eps=eps))
            for p in [8, 16]:
                E.append(dict(base, slice="X7_sig_x_p", r=r, sig=sig, p=p))
        for f in [0.1, 0.3]:
            for T in [20, 80]:
                E.append(dict(base, slice="X8_f_x_T", r=r, f=f, T=T))
    for j, e in enumerate(E):
        e["env"] = 1000 + j
    return E


def cfg_of(e):
    return sim.make_cfg(e["r"], e["sig"], e["eps"], e.get("rho", 0.0), e.get("gk", 0), e.get("obs", 0.0),
                        e.get("delay", 0), e.get("crec", 0.0), e.get("integer", 0))


def load_beh(which="best"):
    c = json.load(open(os.path.join(OUT, "calib.json")))
    b = c[which] if which in c else c["best"]
    return np.array([b["mu_b"], b["sd_b"], b["lam"], b["gam"], b["sd_int"]]), c


def met_dict(v, e, prefix=""):
    v = np.asarray(v)
    d = {prefix + k: float(v[i]) for i, k in enumerate(M)}
    d[prefix + "eff"] = float(v[0] / sim.ceiling(e["r"], e["T"], e.get("gk", 0)))
    return d


def evaluate(fam, e, inst, seed, beh, use_pl=0, th=None, G2=1000, f3kw=None, frozen_ret=True):
    """One evaluation of a rule (tuple) or planner (use_pl=1, th) in one environment and family."""
    cfg = cfg_of(e)
    ip = sim.make_inst(*inst) if inst is not None else sim.make_inst(0)
    if th is None:
        th = np.zeros(sim.NPAR)
    if fam == "F1":
        out, _, _ = sim.evolve_f1(cfg, e["p"], e["T"], e["f"], ip, use_pl, th, seed=seed)
        return met_dict(out, e)
    if fam == "F2":
        met = sim.play_f2(G2, e["p"], e["T"], cfg, e["f"], beh, ip, use_pl, th, seed)
        return met_dict(met.mean(0), e)
    kw = dict(f3kw or {})
    late, fro, _, _ = sim.f3_eval(cfg, e["p"], e["T"], e["f"], ip, seed, use_pl, th, **kw)
    d = met_dict(late, e)
    if frozen_ret:
        d.update(met_dict(fro, e, "fz_"))
    return d


# ------------------------------------------------------------------------------------------------
# calibration of F2 to the per-round human data (group-level targets and bootstrap uncertainty)
# ------------------------------------------------------------------------------------------------
TK = [(m, s) for m in ["equal", "mixed", "proportional"] for s in ["sust", "all4", "gini"]]
MECH_CAL = {"equal": sim.EQ, "mixed": sim.MIX, "proportional": sim.PROP}
CAL_GRID = dict(mu_b=list(np.round(np.arange(0.2, 0.801, 0.05), 3)), sd_b=[0.1, 0.2, 0.3, 0.4, 0.5],
                lam=[0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0], gam=[0.0, 0.1, 0.2, 0.3, 0.45, 0.6],
                sd_int=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], f_h=[0.0, 0.1, 0.2, 0.3])


def _calib_task(par):
    mu_b, sd_b, lam, gam, sd_int, f_h = par
    beh = np.array([mu_b, sd_b, lam, gam, sd_int])
    row = dict(mu_b=mu_b, sd_b=sd_b, lam=lam, gam=gam, sd_int=sd_int, f_h=f_h)
    e = dict(r=0.4, T=40, gk=0)
    for m, inst in MECH_CAL.items():
        met = sim.play_f2(1000, 4, 40, sim.make_cfg(0.4, 0.0), f_h, beh, sim.make_inst(*inst), 0,
                          np.zeros(sim.NPAR), seed_of("calib", m))
        d = met_dict(met.mean(0), e)
        for k in ["sust", "all4", "gini", "eff", "act", "phi"]:
            row[f"{m}_{k}"] = d[k]
    return row


def stage_calib(nproc):
    grid = list(itertools.product(*CAL_GRID.values()))
    with Pool(nproc) as pool:
        rows = pool.map(_calib_task, grid, chunksize=16)
    df = pd.DataFrame(rows)
    tg = pd.read_csv(os.path.join(HUM, "calibration_targets.csv"))
    tvec = np.array([tg[(tg.mech == m) & (tg.stat == s)].target.iloc[0] for m, s in TK])
    sevec = np.array([tg[(tg.mech == m) & (tg.stat == s)].se.iloc[0] for m, s in TK])
    S = df[[f"{m}_{s}" for m, s in TK]].to_numpy()
    w = 1.0 / (sevec ** 2 + 2.5e-4)          # human sampling variance plus Monte Carlo floor (1000 games)
    loss = ((S - tvec) ** 2 * w).sum(1)
    df["loss"] = loss
    df["loss_unweighted"] = ((S - tvec) ** 2).sum(1)
    boot = np.load(os.path.join(HUM, "targets_boot.npy"))
    win = np.array([int(np.argmin(((S - bt) ** 2 * w).sum(1))) for bt in boot])
    cnt = pd.Series(win).value_counts()
    df["boot_freq"] = 0.0
    df.loc[cnt.index, "boot_freq"] = cnt.values / len(boot)
    df.to_csv(os.path.join(OUT, "calib_grid.csv.gz"), index=False, float_format="%.5f")
    keys = list(CAL_GRID.keys())
    ib = int(np.argmin(loss))
    best = {k: float(df.iloc[ib][k]) for k in keys + ["loss", "loss_unweighted"]}
    plaus = df[df.boot_freq >= 0.01].sort_values("boot_freq", ascending=False).head(12)
    out = dict(best=best, targets=dict(zip([f"{m}|{s}" for m, s in TK], tvec.tolist())),
               target_se=dict(zip([f"{m}|{s}" for m, s in TK], sevec.tolist())),
               chi2_dof=float(loss[ib] / (9 - 6)),
               plausible=[{k: float(r[k]) for k in keys + ["loss", "boot_freq"]} for _, r in plaus.iterrows()],
               plausible_coverage=float(plaus.boot_freq.sum()))
    for par in ["gam", "lam", "sd_int"]:
        prof = []
        for v in CAL_GRID[par]:
            sub = df[np.isclose(df[par], v)]
            j = sub.loss.idxmin()
            prof.append({k: float(df.loc[j][k]) for k in keys + ["loss"]})
        out[f"profile_{par}"] = prof
    # micro-consistent fit: generosity response and matching restricted to the range supported by the
    # within-player regressions on the per-round human records (gam <= 0.1, lam <= 0.2)
    sub = df[(df.gam <= 0.1 + 1e-9) & (df.lam <= 0.2 + 1e-9)]
    j = sub.loss.idxmin()
    out["micro"] = {k: float(df.loc[j][k]) for k in keys + ["loss"]}
    S_best = S[ib]
    out["best_fit_values"] = dict(zip([f"{m}|{s}" for m, s in TK], S_best.tolist()))
    out["micro_fit_values"] = dict(zip([f"{m}|{s}" for m, s in TK], S[j].tolist()))
    json.dump(out, open(os.path.join(OUT, "calib.json"), "w"), indent=1)
    print("calib best", best, "plausible", len(plaus), round(plaus.boot_freq.sum(), 3))


# ------------------------------------------------------------------------------------------------
# theory: refined best responses, fixed points, one-sided derivatives, gradient-sign maps, free riders
# ------------------------------------------------------------------------------------------------
RULES = {"equal": sim.EQ, "mixed": sim.MIX, "proportional": sim.PROP, "power22": sim.HEUR,
         "step97": sim.STEP, "capstep": sim.CAPSTEP}


def _payoff(ph_f, ph_r, r, T, p, inst):
    phi = np.full((len(ph_f), p), ph_r)
    phi[:, 0] = ph_f
    fr = np.zeros_like(phi)
    pay, met = sim.play_const(phi, fr, sim.make_cfg(r, 0.0), T, sim.make_inst(*inst), 0, np.zeros(sim.NPAR), 1)
    return pay[:, 0], met


def best_response(ph_r, r, T, p, inst):
    g = np.linspace(0, 1, 2001)
    pf, _ = _payoff(g, ph_r, r, T, p, inst)
    j = int(np.argmax(pf))
    lo, hi = g[max(j - 1, 0)], g[min(j + 1, 2000)]
    for _ in range(30):                       # golden-section refinement of the grid maximum
        a = hi - 0.618034 * (hi - lo)
        b = lo + 0.618034 * (hi - lo)
        fa, _ = _payoff(np.array([a, b]), ph_r, r, T, p, inst)
        if fa[0] >= fa[1]:
            hi = b
        else:
            lo = a
    x = 0.5 * (lo + hi)
    fx, _ = _payoff(np.array([x, ph_r]), ph_r, r, T, p, inst)
    return x, float(fx[0]), float(fx[1])


def _br_curve_task(args):
    name, r, T, p = args
    inst = RULES[name]
    res = np.round(np.arange(0.30, 1.0001, 0.0025), 6)
    rows = []
    for ph in res:
        br, pbr, pres = best_response(ph, r, T, p, inst)
        rows.append(dict(rule=name, r=r, T=T, p=p, phi_res=ph, br=br, gain=pbr - pres))
    df = pd.DataFrame(rows)
    # fixed points: sign changes of br - phi, refined by bisection
    fps = []
    gv = (df.br - df.phi_res).to_numpy()
    for k in range(len(res) - 1):
        if gv[k] == 0 or gv[k] * gv[k + 1] < 0:
            lo, hi, glo = res[k], res[k + 1], gv[k]
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                gm = best_response(mid, r, T, p, inst)[0] - mid
                if gm * glo > 0:
                    lo, glo = mid, gm
                else:
                    hi = mid
            x = 0.5 * (lo + hi)
            brx = best_response(x, r, T, p, inst)
            _, met = _payoff(np.array([x]), x, r, T, p, inst)
            fps.append(dict(rule=name, r=r, T=T, p=p, fixed_point=x, residual=brx[0] - x,
                            downcrossing=bool(gv[k] > 0 >= gv[k + 1]),
                            eff_mono=float(met[0, 0] / sim.ceiling(r, T)), boundary=1 / (1 + r)))
    return rows, fps


def _deriv_task(args):
    name, r, T, p = args
    inst = RULES[name]
    a = 1.0 / (1.0 + r)
    h = 1e-5
    pf, _ = _payoff(np.array([a - h, a, a + h]), a, r, T, p, inst)
    Dm = (pf[1] - pf[0]) / h
    Dp = (pf[2] - pf[1]) / h
    if name == "capstep":
        an = (-T + r * T * (T - 1) / 2) / p
    elif name in ("step97", "power22"):
        an = (-T + (r / p) * T * (T - 1) / 2) / p
    else:
        an = np.nan
    return dict(rule=name, r=r, T=T, p=p, D_minus=Dm, D_plus=Dp, D_minus_analytic=an, D_plus_analytic=-T / p,
                last_round_gain=a / p)


def _fr_task(args):
    r, phic, m, name = args
    phi = np.full((1, 20), phic)
    fr = np.zeros((1, 20))
    fr[0, :m] = 1.0
    phi[0, :m] = 0.0
    pay, met = sim.play_const(phi, fr, sim.make_cfg(r, 0.0), 200, sim.make_inst(*RULES[name]), 0,
                              np.zeros(sim.NPAR), 1)
    d = met_dict(met[0], dict(r=r, T=200))
    d.update(r=r, phi_c=phic, f=m / 20, rule=name, recovers=bool(met[0, 8] >= 1 - 1e-9),
             q=(1 + r) * phic)
    return d


def stage_theory(nproc):
    rg = list(np.round(np.arange(0.05, 1.0001, 0.05), 3))
    tasks = [(n, r, 40, 4) for n in RULES for r in rg]
    tasks += [(n, r, T, 4) for n in ["proportional", "step97", "power22", "capstep"] for r in [0.1, 0.4]
              for T in [10, 20, 80, 160]]
    tasks += [("proportional", 0.4, T, p) for T in [5, 320] for p in [4]]
    tasks += [(n, 0.4, 40, p) for n in ["proportional", "step97", "power22", "capstep"] for p in [2, 8]]
    with Pool(nproc) as pool:
        res = pool.map(_br_curve_task, tasks, chunksize=1)
        dtasks = [(n, r, T, p) for n in ["step97", "power22", "capstep", "proportional", "mixed"]
                  for r in rg for T in [5, 10, 20, 40, 80, 160] for p in [2, 4, 8, 16]]
        der = pool.map(_deriv_task, dtasks, chunksize=8)
        ftasks = [(r, phic, m, n) for r in [0.2, 0.4, 0.8] for phic in [1.0, 0.9, 0.8] for m in range(21)
                  for n in ["equal", "mixed", "proportional", "power22", "step97", "capstep"]]
        frr = pool.map(_fr_task, ftasks, chunksize=16)
    pd.DataFrame([x for a, _ in res for x in a]).to_csv(os.path.join(OUT, "theory_br.csv"), index=False,
                                                        float_format="%.6f")
    pd.DataFrame([x for _, b in res for x in b]).to_csv(os.path.join(OUT, "theory_fixedpoints.csv"),
                                                        index=False, float_format="%.6f")
    pd.DataFrame(der).to_csv(os.path.join(OUT, "theory_derivatives.csv"), index=False, float_format="%.6f")
    pd.DataFrame(frr).to_csv(os.path.join(OUT, "theory_freeriders.csv"), index=False, float_format="%.6f")
    # proportional singular strategy: analytic value and its monomorphic efficiency
    rows = []
    for T in [5, 10, 20, 40, 80, 160, 320]:
        for p in [2, 4, 8]:
            for r in [0.2, 0.4, 0.8]:
                ps = 1.0 / (1 + 2.0 / ((T - 1) * (1 - 1.0 / p)))
                _, met = _payoff(np.array([ps]), ps, r, T, p, sim.PROP)
                rows.append(dict(T=T, p=p, r=r, phi_star=ps, in_domain=bool(ps > 1 / (1 + r)),
                                 eff_analytic=(1 - ps) * (1 + r) / r, eff_sim=float(met[0, 0] / sim.ceiling(r, T))))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "theory_contest.csv"), index=False, float_format="%.6f")


# ------------------------------------------------------------------------------------------------
# phase diagram (common fine grid for all three families) and fresh-seed re-evaluation
# ------------------------------------------------------------------------------------------------
def _grid_task(args):
    fam, e, idx, beh, tag = args
    rows = []
    for ii, inst in idx:
        t0 = time.time()
        s = evaluate(fam, e, inst, seed_of(tag, fam, e["env"]), beh)
        s.update(family=fam, env=e["env"], inst=ii, secs=time.time() - t0)
        rows.append(s)
    return rows


def run_grid(nproc, E, fams, insts, tag, outname, beh=None):
    if beh is None:
        beh, _ = load_beh()
    pd.DataFrame([dict(inst=i, kind=x[0], theta=x[1], wlo=x[2], whi=x[3], k=x[4], amnesty=x[5],
                       regime=sim.regime(x)) for i, x in enumerate(insts)]).to_csv(
        os.path.join(OUT, f"insts_{tag}.csv"), index=False)
    for fam in fams:
        chunk = {"F2": 40, "F1": 4, "F3": 2}[fam]
        idx = list(enumerate(insts))
        tasks = [(fam, e, idx[a:a + chunk], beh, tag) for e in E for a in range(0, len(idx), chunk)]
        t0 = time.time()
        with Pool(nproc) as pool:
            res = pool.map(_grid_task, tasks, chunksize=1)
        df = pd.DataFrame([row for rr in res for row in rr])
        df.to_csv(os.path.join(OUT, f"{outname}_{fam}.csv.gz"), index=False, float_format="%.5f")
        print(tag, fam, "done", len(df), round(time.time() - t0, 1), flush=True)


def stage_grid(nproc, fams=("F2", "F1", "F3")):
    E = envs_main()
    pd.DataFrame(E).to_csv(os.path.join(OUT, "envs.csv"), index=False)
    run_grid(nproc, E, fams, sim.inst_grid(True), "fine", "grid")


def stage_robust(nproc, fams=("F2", "F1", "F3")):
    E = envs_robust()
    pd.DataFrame(E).to_csv(os.path.join(OUT, "envs_robust.csv"), index=False)
    insts = sim.inst_grid(True, extra_low_theta=True)
    run_grid(nproc, E, fams, insts, "robust", "robust")


def _reeval_task(args):
    fam, e, ii, inst, rep, beh = args
    s = evaluate(fam, e, inst, seed_of("reeval", fam, e["env"], rep), beh)
    s.update(family=fam, env=e["env"], inst=ii, rep=rep)
    return s


def stage_reeval(nproc, nrep=6, topk=8):
    """Fresh-seed re-evaluation (selection and evaluation randomness separated) of the leading institutions,
    the best member of each regime, the best contingent rule with and without re-inclusion, canonical
    rules, and the common (cross-family) policies."""
    E = envs_main()
    insts = sim.inst_grid(True) + [sim.MIX]      # the 50:50 mixed rule is not on the 0.2-step weight grid
    pd.DataFrame([dict(inst=i, kind=x[0], theta=x[1], wlo=x[2], whi=x[3], k=x[4], amnesty=x[5],
                       regime=sim.regime(x)) for i, x in enumerate(insts)]).to_csv(
        os.path.join(OUT, "insts_fine_ext.csv"), index=False)
    reg = np.array([sim.regime(x) for x in insts])
    am = np.array([x[5] for x in insts])
    beh, _ = load_beh()
    G = {f: pd.read_csv(os.path.join(OUT, f"grid_{f}.csv.gz")) for f in ["F1", "F2", "F3"]}
    canon = [insts.index(x) for x in [sim.STEP, sim.HEUR, sim.PROP, sim.EQ, sim.MIX, sim.STEP_PERM]]
    # common policies: minimax regret across families, per environment and globally
    piv = {}
    for f, df in G.items():
        X = df.pivot(index="env", columns="inst", values="eff")
        piv[f] = X.sub(X.max(1), axis=0)          # regret (<= 0) relative to the family's best
    reg_all = np.minimum(np.minimum(piv["F1"].to_numpy(), piv["F2"].to_numpy()), piv["F3"].to_numpy())
    reg = reg[:reg_all.shape[1]]; am = am[:reg_all.shape[1]]
    glob = int(np.argmax(reg_all.min(0)))
    glob_mean = int(np.argmax(reg_all.mean(0)))
    tasks = []
    chosen = {}
    for j, e in enumerate(E):
        cand = set(canon + [glob, glob_mean, int(np.argmax(reg_all[j]))])
        for f, df in G.items():
            d = df[df.env == e["env"]].sort_values("eff", ascending=False)
            cand |= set(d.inst.head(topk).tolist())
            for rg in "PEITM":
                dd = d[reg[d.inst.to_numpy()] == rg]
                if len(dd):
                    cand.add(int(dd.inst.iloc[0]))
            for a in (0, 1):
                dd = d[(reg[d.inst.to_numpy()] == "T") & (am[d.inst.to_numpy()] == a)]
                if len(dd):
                    cand.add(int(dd.inst.iloc[0]))
        chosen[e["env"]] = sorted(cand)
        for f in ["F1", "F2", "F3"]:
            for ii in sorted(cand):
                for rep in range(nrep):
                    tasks.append((f, e, ii, insts[ii], rep, beh))
    json.dump(dict(global_minimax=glob, global_meanregret=glob_mean, chosen={str(k): v for k, v in chosen.items()}),
              open(os.path.join(OUT, "reeval_candidates.json"), "w"))
    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_reeval_task, tasks, chunksize=2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "reeval.csv.gz"), index=False, float_format="%.5f")
    print("reeval", len(rows), round(time.time() - t0, 1), flush=True)


def _sens_task(args):
    kind, e, ii, inst, rep, beh, extra = args
    if kind == "f3var":
        s = evaluate("F3", e, inst, seed_of("f3var", e["env"], rep, json.dumps(extra)), beh, f3kw=extra)
    else:   # calibration-sensitivity for F2
        s = evaluate("F2", e, inst, seed_of("calsens", e["env"], rep), np.array(extra["beh"]))
    s.update(kind=kind, env=e["env"], inst=ii, rep=rep, variant=json.dumps(extra if kind == "f3var" else extra["name"]))
    return s


def stage_sens(nproc):
    """(i) Q-learner representation/budget sensitivity on representative cells; (ii) F2 phase diagram under
    every plausible calibration and the gamma = 0 calibration (all 112 environments, fine grid)."""
    E = envs_main()
    insts = sim.inst_grid(True) + [sim.MIX]
    beh, c = load_beh()
    canon = {n: insts.index(x) for n, x in [("step", sim.STEP), ("heur", sim.HEUR), ("prop", sim.PROP),
                                            ("eq", sim.EQ), ("mix", sim.MIX), ("stepperm", sim.STEP_PERM)]}
    rep_env = [e for e in E if e["slice"] in ("S1_r_f",) and e["f"] in (0.0, 0.2)] + \
              [e for e in E if e["slice"] == "S2_r_noise" and e["sig"] == 0.2]
    tasks = []
    variants = [dict(qv=1), dict(qv=2), dict(n_ep=5000), dict(alpha=0.05), dict(gamma=0.99)]
    for e in rep_env:
        for v in variants:
            for nm, ii in canon.items():
                for rep in range(3):
                    tasks.append(("f3var", e, ii, insts[ii], rep, beh, v))
    fits = [dict(name=f"plaus{j}", beh=[q["mu_b"], q["sd_b"], q["lam"], q["gam"], q["sd_int"]])
            for j, q in enumerate(c["plausible"][:8])]
    g0 = c["micro"]
    fits.append(dict(name="micro", beh=[g0["mu_b"], g0["sd_b"], g0["lam"], g0["gam"], g0["sd_int"]]))
    for e in E:
        for ft in fits:
            for ii in range(len(insts) - 1):
                tasks.append(("calsens", e, ii, insts[ii], 0, beh, ft))
    with Pool(nproc) as pool:
        rows = pool.map(_sens_task, tasks, chunksize=8)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity.csv.gz"), index=False, float_format="%.5f")


# ------------------------------------------------------------------------------------------------
# retrospective comparison with the human experiments (continuous and integer-token implementations)
# ------------------------------------------------------------------------------------------------
T3MECH = {"equal": sim.EQ, "mixed": sim.MIX, "proportional": sim.PROP, "heuristic": sim.HEUR_R}


def _t3_task(args):
    fam, mech, rep, beh, fh, integer, tag = args
    e = dict(r=0.4, p=4, T=40, sig=0.05 if fam != "F2" else 0.0, eps=0.0, f=fh if fam == "F2" else 0.0,
             integer=integer, env=0)
    s = evaluate(fam, e, T3MECH[mech], seed_of("t3", tag, fam, mech, rep, integer), beh, G2=2000)
    s.update(family=tag, mech=mech, rep=rep, integer=integer)
    return s


def stage_t3(nproc, nrep=10):
    beh, c = load_beh()
    fh = c["best"]["f_h"]
    tasks = [(f, m, rep, beh, fh, integer, f) for f in ["F1", "F2", "F3"] for m in T3MECH
             for rep in range(nrep) for integer in (0, 1)]
    for j, q in enumerate(c["plausible"]):
        b = np.array([q["mu_b"], q["sd_b"], q["lam"], q["gam"], q["sd_int"]])
        for m in T3MECH:
            for rep in range(5):
                tasks.append(("F2", m, rep, b, q["f_h"], 0, f"F2plaus{j:02d}"))
    g0 = c["micro"]
    b = np.array([g0["mu_b"], g0["sd_b"], g0["lam"], g0["gam"], g0["sd_int"]])
    for m in T3MECH:
        for rep in range(5):
            tasks.append(("F2", m, rep, b, g0["f_h"], 0, "F2micro"))
    with Pool(nproc) as pool:
        rows = pool.map(_t3_task, tasks, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "t3.csv"), index=False, float_format="%.6f")


# ------------------------------------------------------------------------------------------------
# simulated scores of the five institution classes used in the field analysis (H4)
# ------------------------------------------------------------------------------------------------
CLASSES = {"EQUAL": sim.EQ, "THRESHOLD-AMNESTY": sim.STEP, "THRESHOLD-PERMANENT": sim.STEP_PERM,
           "MIXED-FORGIVING": sim.MIX, "PROPORTIONAL": sim.PROP}


def _h4_task(args):
    cls, fam, r, f, sig, p, rep, beh = args
    e = dict(r=r, p=p, T=40, sig=sig, eps=0.0, f=f, env=0)
    s = evaluate(fam, e, CLASSES[cls], seed_of("h4", cls, fam, r, f, sig, p, rep), beh)
    s.update(cls=cls, family=fam, r=r, f=f, sig=sig, p=p, rep=rep)
    return s


def stage_h4(nproc):
    beh, _ = load_beh()
    tasks = []
    for cls in CLASSES:
        for fam in ["F1", "F2", "F3"]:
            for (r, f, sig, p) in [(0.2, 0.2, 0.05, 4), (0.4, 0.2, 0.05, 4), (0.8, 0.2, 0.05, 4),
                                   (0.4, 0.0, 0.05, 4), (0.4, 0.2, 0.1, 4), (0.4, 0.2, 0.05, 8)]:
                for rep in range(10):
                    tasks.append((cls, fam, r, f, sig, p, rep, beh))
    with Pool(nproc) as pool:
        rows = pool.map(_h4_task, tasks, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "h4_class_sims.csv"), index=False, float_format="%.6f")


# ------------------------------------------------------------------------------------------------
# neural planners: ES training against F2, distillation diagnostics, cross-model and control experiments
# ------------------------------------------------------------------------------------------------
def train_es(e, beh, iters, seed, npairs=48, sigma=0.05, lr=0.03, G=128, log_every=10):
    cfg = cfg_of(e)
    ip = sim.make_inst(0)
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.1, sim.NPAR)
    m = np.zeros(sim.NPAR); v = np.zeros(sim.NPAR)
    b1, b2 = 0.9, 0.999
    hist = []
    for it in range(iters):
        s = int(rng.integers(1 << 30))
        Ep = rng.normal(0, 1, (npairs, sim.NPAR))
        fit = np.zeros(2 * npairs)
        for j in range(npairs):
            for sgn, idx in ((1.0, 2 * j), (-1.0, 2 * j + 1)):
                met = sim.play_f2(G, e["p"], e["T"], cfg, e["f"], beh, ip, 1, th + sgn * sigma * Ep[j], s)
                fit[idx] = met[:, 0].mean()
        ranks = np.empty(2 * npairs); ranks[np.argsort(fit)] = np.arange(2 * npairs)
        u = ranks / (2 * npairs - 1) - 0.5
        grad = ((u[0::2] - u[1::2])[:, None] * Ep).sum(0) / (2 * npairs * sigma)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** (it + 1)); vh = v / (1 - b2 ** (it + 1))
        th = th + lr * mh / (np.sqrt(vh) + 1e-8) - lr * 0.005 * th
        if it % log_every == 0 or it == iters - 1:
            met = sim.play_f2(512, e["p"], e["T"], cfg, e["f"], beh, ip, 1, th, 12345)
            hist.append((it, met[:, 0].mean() / sim.ceiling(e["r"], e["T"]), met[:, 1].mean(), met[:, 2].mean(),
                         met[:, 3].mean()))
    return th, np.array(hist)


def distil(th, e, beh, G=2000, seed=777):
    cfg = cfg_of(e)
    R_tr, sh, cs_, ex = sim.trace_f2(G, e["p"], e["T"], cfg, e["f"], beh, sim.make_inst(0), 1, th, seed)
    p = e["p"]
    rows = []
    nall = 0
    for g in range(G):
        for t in range(1, e["T"]):
            R = R_tr[g, t]
            if R <= 0:
                continue
            nall += 1
            x = 1.0 / p - cs_[g, t]
            y = sh[g, t] - cs_[g, t]
            den = float((x * x).sum())
            if den < 1e-4:
                continue
            w = float((y * x).sum() / den)
            res = float(((y - w * x) ** 2).sum())
            tot = float((y * y).sum())
            proj = w / p + (1 - w) * cs_[g, t]
            rows.append((R, w, res, tot, float((proj < -1e-9).any()), float((sh[g, t] == 0).sum())))
    return np.array(rows), nall


def planner_summary(rows, nall):
    R, w, res, tot, neg, nz = rows.T
    out = dict(n_rounds=int(nall), n_used=len(rows), frac_discarded=1 - len(rows) / max(nall, 1),
               R2=1 - res.sum() / max(tot.sum(), 1e-12),
               med_resid_frac=float(np.median(res[tot > 1e-8] / tot[tot > 1e-8])) if (tot > 1e-8).any() else np.nan,
               frac_negative_projection=float(neg.mean()))
    for nm, lo, hi in [("w_lt05", 0, 0.5), ("w_05_09", 0.5, 0.9), ("w_09_096", 0.9, 0.96), ("w_ge096", 0.96, 1.01)]:
        sel = (R >= lo) & (R < hi)
        out[nm] = float(w[sel].mean()) if sel.sum() >= 20 else np.nan        # round-weighted (occupancy) mean
        out[nm + "_n"] = int(sel.sum())
    bins = np.linspace(0, 1, 21)
    bi = np.clip(np.digitize(R, bins) - 1, 0, 19)
    tab = np.full(20, np.nan)
    for b in range(20):
        if (bi == b).sum() >= 20:
            tab[b] = float(np.median(w[bi == b]))
    # fill unvisited bins from the nearest visited bin (executable distilled table)
    vis = np.where(np.isfinite(tab))[0]
    if len(vis):
        for b in range(20):
            if not np.isfinite(tab[b]):
                tab[b] = tab[vis[np.argmin(np.abs(vis - b))]]
    else:
        tab[:] = 0.0
    out["wtab"] = tab.tolist()
    return out


def _es_task(args):
    r, f, seed, beh, iters, tag = args
    r = round(float(r), 3)
    e = dict(r=r, p=4, T=40, sig=0.05, eps=0.0, f=f, env=0)
    t0 = time.time()
    th, hist = train_es(e, beh, iters, seed_of("es", tag, r, f, seed))
    rows, nall = distil(th, e, beh, seed=seed_of("distil", tag, r, f, seed))
    summ = planner_summary(rows, nall)
    bins = np.linspace(0, 1, 21)
    bi = np.clip(np.digitize(rows[:, 0], bins) - 1, 0, 19)
    curve = [dict(tag=tag, r=r, f=f, seed=seed, Rmid=(bins[b] + bins[b + 1]) / 2, w_mean=rows[bi == b, 1].mean(),
                  w_med=float(np.median(rows[bi == b, 1])), n=int((bi == b).sum()))
             for b in range(20) if (bi == b).sum() >= 20]
    np.save(os.path.join(OUT, f"es_theta_{tag}_r{r}_f{f}_s{seed}.npy"), th)
    summ.update(tag=tag, r=r, f=f, seed=seed, secs=time.time() - t0, iters=iters)
    hist_rows = [dict(tag=tag, r=r, f=f, seed=seed, it=h[0], eff=h[1], gini=h[2], sust=h[3], act=h[4]) for h in hist]
    return summ, curve, hist_rows


def stage_es(nproc, iters=800):
    beh, c = load_beh()
    g0 = c["micro"]
    beh0 = np.array([g0["mu_b"], g0["sd_b"], g0["lam"], g0["gam"], g0["sd_int"]])
    tasks = [(r, f, s, beh, iters, "best") for r in [0.2, 0.3, 0.4, 0.6, 0.8] for f in [0.0, 0.2] for s in range(5)]
    tasks += [(r, f, s, beh0, iters, "micro") for r in [0.2, 0.3, 0.4, 0.6, 0.8] for f in [0.0, 0.2] for s in range(2)]
    with Pool(min(nproc, len(tasks))) as pool:
        res = pool.map(_es_task, tasks, chunksize=1)
    pd.DataFrame([a for a, _, _ in res]).to_json(os.path.join(OUT, "es_summary.json"), orient="records", indent=0)
    pd.DataFrame([row for _, b, _ in res for row in b]).to_csv(os.path.join(OUT, "es_curves.csv"), index=False,
                                                               float_format="%.5f")
    pd.DataFrame([row for _, _, h in res for row in h]).to_csv(os.path.join(OUT, "es_hist.csv"), index=False,
                                                               float_format="%.5f")


def _cross_task(args):
    kind, key, tag, r, f, fam, obj, rep, beh, mode = args
    e = dict(r=r, p=4, T=40, sig=0.05, eps=0.0, f=f, env=0)
    cfg = cfg_of(e)
    sd = seed_of("cross", kind, key, tag, r, f, fam, rep, mode)
    if kind == "planner":
        use_pl, th, inst = 1, obj, None
    elif kind == "table":
        use_pl, th, inst = 0, None, (2, 0.0, 0.0, 0.0, 0.0, 1)
    else:
        use_pl, th, inst = 0, None, obj
    ip = sim.make_inst(2, wtab=obj) if kind == "table" else (sim.make_inst(*inst) if inst is not None else sim.make_inst(0))
    thv = th if th is not None else np.zeros(sim.NPAR)
    out = dict(kind=kind, key=key, tag=tag, r=r, f=f, family=fam, rep=rep, mode=mode)
    if mode == "adapt":
        if fam == "F2":
            met = sim.play_f2(2000, 4, 40, cfg, f, beh, ip, use_pl, thv, sd)
            out.update(met_dict(met.mean(0), e))
        elif fam == "F1":
            o, _, _ = sim.evolve_f1(cfg, 4, 40, f, ip, use_pl, thv, seed=sd)
            out.update(met_dict(o, e))
        else:
            late, fro, _, _ = sim.f3_eval(cfg, 4, 40, f, ip, sd, use_pl, thv)
            out.update(met_dict(late, e)); out.update(met_dict(fro, e, "fz_"))
        return out
    # control: recipients first adapted to the collective step rule, then exposed to the policy with
    # (mode == "pre_adapt") or without (mode == "pre_frozen") further adaptation
    ips = sim.make_inst(*sim.STEP)
    if fam == "F1":
        _, phi, fixed = sim.evolve_f1(cfg, 4, 40, f, ips, 0, None, seed=sd)
        o, _, _ = sim.evolve_f1(cfg, 4, 40, f, ip, use_pl, thv, seed=sd + 1, phi0=phi,
                                evolve=(mode == "pre_adapt"))
        out.update(met_dict(o, e))
    else:
        Q, frm = sim.f3_init(64, 4, f, sd)
        sim.run_f3(Q, frm, 4, 40, cfg, ips, 0, np.zeros(sim.NPAR), sd, 2500, 1, 0, 0.1, 0.95, 0.2, 0.01, 1, 0)
        if mode == "pre_frozen":
            late, fro = sim.run_f3(Q, frm, 4, 40, cfg, ip, use_pl, thv, sd + 1, 0, 0, 300, 0.1, 0.95, 0.01, 0.01, 0, 0)
            out.update(met_dict(fro, e))
        else:
            late, fro = sim.run_f3(Q, frm, 4, 40, cfg, ip, use_pl, thv, sd + 1, 2500, 300, 0, 0.1, 0.95, 0.01, 0.01, 1, 0)
            out.update(met_dict(late, e))
    return out


def stage_cross(nproc, nrep=3):
    beh, c = load_beh()
    summ = pd.read_json(os.path.join(OUT, "es_summary.json"))
    fits = {"best": beh}
    for j, q in enumerate(c["plausible"][:8]):
        fits[f"plaus{j}"] = np.array([q["mu_b"], q["sd_b"], q["lam"], q["gam"], q["sd_int"]])
    g0 = c["micro"]
    fits["micro"] = np.array([g0["mu_b"], g0["sd_b"], g0["lam"], g0["gam"], g0["sd_int"]])
    fixed = [("step97", sim.STEP), ("heuristic", sim.HEUR), ("proportional", sim.PROP), ("mixed", sim.MIX),
             ("equal", sim.EQ), ("step97_half", (0, 0.97, 0.0, 0.5, 0.0, 1)), ("step97_perm", sim.STEP_PERM)]
    tasks = []
    for r in [0.2, 0.3, 0.4, 0.6, 0.8]:
        for f in [0.0, 0.2]:
            objs = []
            for _, s in summ[(summ.r == r) & (summ.f == f)].iterrows():
                th = np.load(os.path.join(OUT, f"es_theta_{s.tag}_r{r}_f{f}_s{int(s.seed)}.npy"))
                objs.append(("planner", f"es_s{int(s.seed)}", s.tag, th))
                objs.append(("table", f"tab_s{int(s.seed)}", s.tag, np.array(s.wtab)))
            objs += [("inst", nm, "fixed", inst) for nm, inst in fixed]
            objs.append(("table", "compensatory133", "fixed", np.full(20, 1.33)))
            for kind, key, tag, obj in objs:
                for fam in ["F1", "F2", "F3"]:
                    for rep in range(nrep):
                        tasks.append((kind, key, tag, r, f, fam, obj, rep, beh, "adapt"))
                for fname, b in fits.items():
                    if fname != "best":
                        tasks.append((kind, key, tag + "|" + fname, r, f, "F2", obj, 0, b, "adapt"))
                if kind != "table" or key.startswith("tab"):
                    for fam in ["F1", "F3"]:
                        for mode in ["pre_frozen", "pre_adapt"]:
                            tasks.append((kind, key, tag, r, f, fam, obj, 0, beh, mode))
    with Pool(nproc) as pool:
        rows = pool.map(_cross_task, tasks, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_eval.csv.gz"), index=False, float_format="%.5f")


def stage_es_lam1(nproc, iters=800):
    """Planners trained against the best calibration constrained to lam = 1 (strong matching, the regime of the
    original submission's fit), to test how the optimiser's discovery depends on a weakly identified parameter."""
    beh, c = load_beh()
    q = [x for x in c["profile_lam"] if abs(x["lam"] - 1.0) < 1e-9][0]
    b1 = np.array([q["mu_b"], q["sd_b"], q["lam"], q["gam"], q["sd_int"]])
    tasks = [(r, f, s, b1, iters, "lam1") for r in [0.2, 0.3, 0.4, 0.6, 0.8] for f in [0.0, 0.2] for s in range(2)]
    with Pool(min(nproc, len(tasks))) as pool:
        res = pool.map(_es_task, tasks, chunksize=1)
    pd.DataFrame([a for a, _, _ in res]).to_json(os.path.join(OUT, "es_summary_lam1.json"), orient="records", indent=0)
    pd.DataFrame([row for _, b, _ in res for row in b]).to_csv(os.path.join(OUT, "es_curves_lam1.csv"), index=False)
    pd.DataFrame([row for _, _, h in res for row in h]).to_csv(os.path.join(OUT, "es_hist_lam1.csv"), index=False)
    tasks = []
    fits = {"lam1": b1, "best": beh}
    for r in [0.2, 0.3, 0.4, 0.6, 0.8]:
        for f in [0.0, 0.2]:
            for s in range(2):
                th = np.load(os.path.join(OUT, f"es_theta_lam1_r{r}_f{f}_s{s}.npy"))
                for fam in ["F1", "F3"]:
                    for rep in range(3):
                        tasks.append(("planner", f"es_s{s}", "lam1", r, f, fam, th, rep, beh, "adapt"))
                for fname, b in fits.items():
                    for rep in range(3):
                        tasks.append(("planner", f"es_s{s}", "lam1|" + fname, r, f, "F2", th, rep, b, "adapt"))
                for fam in ["F1", "F3"]:
                    for mode in ["pre_frozen", "pre_adapt"]:
                        tasks.append(("planner", f"es_s{s}", "lam1", r, f, fam, th, 0, beh, mode))
            for nm, inst in [("step97", sim.STEP), ("proportional", sim.PROP), ("equal", sim.EQ)]:
                for rep in range(3):
                    tasks.append(("inst", nm, "fixed|lam1", r, f, "F2", inst, rep, b1, "adapt"))
    with Pool(nproc) as pool:
        rows = pool.map(_cross_task, tasks, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_eval_lam1.csv.gz"), index=False, float_format="%.5f")


def stage_grid_alt(nproc):
    """F2 phase diagram under the strong-matching calibration (lam = 1) and under the submitted version's fit."""
    _, c = load_beh()
    q = [x for x in c["profile_lam"] if abs(x["lam"] - 1.0) < 1e-9][0]
    E = envs_main()
    run_grid(nproc, E, ["F2"], sim.inst_grid(True), "fine", "gridlam1",
             beh=np.array([q["mu_b"], q["sd_b"], q["lam"], q["gam"], q["sd_int"]]))
    run_grid(nproc, E, ["F2"], sim.inst_grid(True), "fine", "gridv1", beh=np.array([0.6, 0.2, 1.0, 0.3, 0.3]))


if __name__ == "__main__":
    sim.warmup()
    stage = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    fams = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("F2", "F1", "F3")
    {"calib": lambda: stage_calib(nproc), "theory": lambda: stage_theory(nproc),
     "grid": lambda: stage_grid(nproc, fams), "reeval": lambda: stage_reeval(nproc),
     "robust": lambda: stage_robust(nproc, fams), "sens": lambda: stage_sens(nproc),
     "t3": lambda: stage_t3(nproc), "h4": lambda: stage_h4(nproc), "es": lambda: stage_es(nproc),
     "cross": lambda: stage_cross(nproc), "es_lam1": lambda: stage_es_lam1(nproc), "grid_alt": lambda: stage_grid_alt(nproc)}[stage]()
    print("stage", stage, "finished in", round(time.time() - t0, 1), "s", flush=True)
