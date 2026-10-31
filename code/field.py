"""Revised field analysis (Baggio et al. 2016 truth tables): reconstruction audit, construct-level codebook,
exhaustive class mapping, pre-specified H1-H4 with corrected reporting, and the sensitivity analyses requested
in review (comparator scores, out-of-sample discrimination, separation handling, small-sample LR calibration,
alternative mappings and score definitions, outcome-dependent completions, misclassification, clustering).

usage: python3 field.py <S1_extracted_text> <h4_class_sims.csv> <outdir>
"""
import re, sys, os, json, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
DPS = ['1A', '1B', '2A', '2B', '3', '4A', '4B', '5', '6', '7', '8']
COLS = [f'DP{d}' for d in DPS]
# Labels as defined in the source coding (Baggio et al. 2016, Table A2, which follows Cox et al. 2010, Table 3)
LABELS = {'1A': 'User boundaries', '1B': 'Resource boundaries', '2A': 'Congruence with local conditions',
          '2B': 'Appropriation-provision congruence', '3': 'Collective-choice arrangements', '4A': 'Monitoring',
          '4B': 'Monitoring the monitors', '5': 'Graduated sanctions', '6': 'Conflict-resolution mechanisms',
          '7': 'Minimal recognition of rights', '8': 'Nested enterprises'}
CODINGS = [('complete', 'complete cases'), ('miss0', 'missing = absent'), ('miss1', 'missing = present')]
B_BOOT = 4000
B_PERM = 20000


def parse_tables(txt):
    recs, table, block = [], None, None
    for line in txt.replace('\r', '').split('\n'):
        m = re.match(r'Table (A[4-7]): Truth table, (\w+)', line)
        if m:
            table = m.group(1); continue
        if line.startswith('Table A8'):
            table = None; continue
        if table is None or line.startswith('cont. Truth table'):
            continue
        s = line.strip()
        for pat, b in [(r'^Complete\s+', 'complete'), (r'^Missing\s*=\s*0\s+', 'miss0'), (r'^Missing\s*=\s*1\s+', 'miss1')]:
            mm = re.match(pat, s)
            if mm:
                block = b; s = s[mm.end():]; break
        toks = s.split()
        if len(toks) == 14 and all(re.match(r'^\d+(\.\d+)?$', t) for t in toks):
            v = [float(t) for t in toks]
            dec = len(toks[13].split('.')[1]) if '.' in toks[13] else 0
            recs.append(dict(table=table, block=block, **{c: int(v[i]) for i, c in enumerate(COLS)},
                             out=int(v[11]), n=int(v[12]), cons=v[13], cons_txt=toks[13], cons_dec=dec))
    df = pd.DataFrame(recs)
    df['n_succ'] = (df.n * df.cons).round().astype(int)
    return df


def audit(tt):
    rows = []
    for _, r in tt.iterrows():
        tol = 0.5 * 10 ** (-r.cons_dec) + 1e-9
        ks = [k for k in range(int(r.n) + 1) if abs(k / r.n - r.cons) <= tol]
        rows.append(dict(table=r.table, block=r.block, config=''.join(str(int(r[c])) for c in COLS), n=int(r.n),
                         consistency_printed=r.cons_txt, implied=r.n * r.cons, assigned_successes=int(r.n_succ),
                         compatible_counts=';'.join(map(str, ks)), unique=len(ks) == 1,
                         outcome_column=int(r.out), mixed=0 < r.n_succ < r.n))
    return pd.DataFrame(rows)


def expand(df):
    rows = []
    for q, (_, r) in enumerate(df.iterrows()):
        for j in range(int(r.n)):
            rows.append({**{c: int(r[c]) for c in COLS}, 'success': int(j < r.n_succ), 'config_id': q})
    return pd.DataFrame(rows)


def mcc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def auc_two(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


MAPPINGS = {
    # pre-specified mapping (analysis plan): precedence 2A&5 -> amnesty; 2A -> permanent; 5 -> mixed; 2B -> P; else E
    'prespecified': lambda a, b, s: ('EQUAL' if not (b or s) else 'THRESHOLD-AMNESTY' if (a and s) else
                                     'THRESHOLD-PERMANENT' if a else 'MIXED-FORGIVING' if s else 'PROPORTIONAL'),
    # contingent classes require the accountability code 2B
    'strict_2B': lambda a, b, s: ('EQUAL' if not (b or s) else 'THRESHOLD-AMNESTY' if (a and b and s) else
                                  'THRESHOLD-PERMANENT' if (a and b) else 'MIXED-FORGIVING' if s else 'PROPORTIONAL'),
    # graduated sanctions not interpreted as re-inclusion (no amnesty classes): contingency from 2A&2B only
    'no_amnesty_reading': lambda a, b, s: ('THRESHOLD-PERMANENT' if (a and b) else 'PROPORTIONAL' if b else
                                           'MIXED-FORGIVING' if s else 'EQUAL'),
    # absence of 2B read as unconditional sharing even when 5 is present
    'equal_without_2B': lambda a, b, s: ('EQUAL' if not b else 'THRESHOLD-AMNESTY' if (a and s) else
                                         'THRESHOLD-PERMANENT' if a else 'MIXED-FORGIVING' if s else 'PROPORTIONAL'),
}


def classes(d, mapping='prespecified'):
    f = MAPPINGS[mapping]
    return np.array([f(a == 1, b == 1, s == 1) for a, b, s in zip(d.DP2A, d.DP2B, d.DP5)])


def class_scores(h4):
    """Class scores under alternative estimands, families and reference environments."""
    out = {}
    ref = (0.4, 0.2, 0.05, 4)
    envs = h4[['r', 'f', 'sig', 'p']].drop_duplicates().itertuples(index=False)
    for (r, f, sig, p) in envs:
        sub = h4[(h4.r == r) & (h4.f == f) & (h4.sig == sig) & (h4.p == p)]
        tag = 'ref' if (r, f, sig, p) == ref else f'r{r}_f{f}_s{sig}_p{p}'
        g = sub.groupby(['cls', 'family'])
        prod = (g.sust.mean() * g.act.mean()).unstack()
        joint = g.sust_act.mean().unstack()
        for nm, tab in [('prod', prod), ('joint', joint), ('sust', g.sust.mean().unstack()),
                        ('act', g.act.mean().unstack()), ('eff', g.eff.mean().unstack())]:
            out[f'{tag}|{nm}|mean'] = tab.mean(1).to_dict()
            for fam in tab.columns:
                out[f'{tag}|{nm}|{fam}'] = tab[fam].to_dict()
    return out


def logit_fit(X, y):
    import statsmodels.api as sm
    Xc = sm.add_constant(np.asarray(X, float), has_constant='add')
    try:
        m = sm.Logit(y, Xc).fit(disp=0, method='newton', maxiter=200)
        sep = bool(np.max(np.abs(m.params)) > 15 or not np.all(np.isfinite(m.bse)) or np.max(m.bse) > 1e3)
        return m, sep
    except Exception:
        return None, True


def lr_test(base, aug, y, nboot=2000, seed=0):
    m1, s1 = logit_fit(base, y)
    m2, s2 = logit_fit(aug, y)
    if m1 is None or m2 is None or s1 or s2:
        return dict(LR=np.nan, p_asym=np.nan, p_boot=np.nan, separated=True)
    LR = 2 * (m2.llf - m1.llf)
    rng = np.random.default_rng(seed)
    pr = m1.predict()
    null = []
    for _ in range(nboot):
        yb = (rng.random(len(y)) < pr).astype(int)
        a, sa = logit_fit(base, yb)
        b, sb = logit_fit(aug, yb)
        if a is None or b is None or sa or sb:
            continue
        null.append(2 * (b.llf - a.llf))
    null = np.array(null)
    ci = m2.conf_int()[-1]
    return dict(LR=float(LR), p_asym=float(stats.chi2.sf(LR, 1)),
                p_boot=float((1 + (null >= LR - 1e-12).sum()) / (len(null) + 1)), nboot_ok=len(null),
                coef=float(m2.params[-1]), coef_lo=float(ci[0]), coef_hi=float(ci[1]), separated=False)


def loo_ridge_auc(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        if len(np.unique(y[tr])) < 2:
            pred[i] = y[tr].mean(); continue
        m = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=2000).fit(X[tr], y[tr])
        pred[i] = m.predict_proba(X[i:i + 1])[0, 1]
    brier = float(np.mean((pred - y) ** 2))
    return auc(pred, y), brier, pred


def analyse(d, name, scores, rng):
    y = d.success.values.astype(int)
    out = dict(coding=name, n=len(d), n_success=int(y.sum()), n_configs=int(d.config_id.nunique()))
    core = (d.DP2A & d.DP2B & d.DP5).values.astype(int)
    a, b = int(((core == 1) & (y == 1)).sum()), int(((core == 1) & (y == 0)).sum())
    c_, dd = int(((core == 0) & (y == 1)).sum()), int(((core == 0) & (y == 0)).sum())
    out.update(core_n=a + b, core_succ=a, rest_n=c_ + dd, rest_succ=c_, H1_rd=a / max(a + b, 1) - c_ / max(c_ + dd, 1))
    out['H1_fisher_p'] = float(stats.fisher_exact([[a, b], [c_, dd]], alternative='greater')[1])
    pc = rng.beta(1 + a, 1 + b, 200000); pr = rng.beta(1 + c_, 1 + dd, 200000)
    out['H1_post_diff_mean'] = float((pc - pr).mean())
    out['H1_post_diff_lo'], out['H1_post_diff_hi'] = [float(q) for q in np.quantile(pc - pr, [0.025, 0.975])]
    # configuration-cluster bootstrap of the risk difference
    cfg_ids = d.config_id.unique()
    rds = []
    for _ in range(B_BOOT):
        pick = rng.choice(cfg_ids, len(cfg_ids))
        dd_ = pd.concat([d[d.config_id == k] for k in pick])
        cc = (dd_.DP2A & dd_.DP2B & dd_.DP5).values
        yy = dd_.success.values
        if cc.sum() == 0 or (1 - cc).sum() == 0:
            continue
        rds.append(yy[cc == 1].mean() - yy[cc == 0].mean())
    out['H1_rd_cluster_lo'], out['H1_rd_cluster_hi'] = [float(q) for q in np.quantile(rds, [0.025, 0.975])]
    # H2: descriptive 80% consistency criterion with denominators and Wilson intervals
    for X in ['2A', '2B', '5', '4A', '4B']:
        k = int(d.loc[y == 1, f'DP{X}'].sum()); n = int((y == 1).sum())
        lo, hi = wilson(k, n)
        out[f'H2_{X}_k'], out[f'H2_{X}_n'], out[f'H2_{X}_prop'], out[f'H2_{X}_lo'], out[f'H2_{X}_hi'] = k, n, k / n, lo, hi
    # H3: subset rank, ties, inclusive tail fraction, and a label-permutation P for the core's MCC
    for k, core_set, lab in [(3, ('2A', '2B', '5'), 'triple'), (5, ('2A', '2B', '4A', '4B', '5'), 'five')]:
        vals = {}
        for S in itertools.combinations(DPS, k):
            ind = np.ones(len(d), int)
            for s in S:
                ind &= d[f'DP{s}'].values
            vals[S] = mcc(ind, y)
        v0 = vals[core_set]
        allv = np.array(list(vals.values()))
        out[f'H3_{lab}_mcc'] = v0
        out[f'H3_{lab}_rank'] = int((allv > v0 + 1e-12).sum() + 1)
        out[f'H3_{lab}_ties'] = int((np.abs(allv - v0) <= 1e-12).sum())
        out[f'H3_{lab}_N'] = len(allv)
        out[f'H3_{lab}_tailfrac'] = float((allv >= v0 - 1e-12).sum() / len(allv))
        out[f'H3_{lab}_best'] = '+'.join(max(vals, key=vals.get))
        out[f'H3_{lab}_bestmcc'] = float(max(allv))
        ind0 = np.ones(len(d), int)
        for s in core_set:
            ind0 &= d[f'DP{s}'].values
        null = np.array([mcc(ind0, rng.permutation(y)) for _ in range(5000)])
        out[f'H3_{lab}_perm_p'] = float((1 + (null >= v0 - 1e-12).sum()) / (len(null) + 1))
    # H4 and comparators
    cls = classes(d)
    out['class_counts'] = json.dumps(pd.Series(cls).value_counts().to_dict())
    out['class_success'] = json.dumps({k: [int(y[cls == k].sum()), int((cls == k).sum())] for k in sorted(set(cls))})
    s = np.array([scores[c] for c in cls])
    cnt = d[COLS].sum(1).values.astype(float)
    cnt3 = (d.DP2A + d.DP2B + d.DP5).values.astype(float)
    comps = {'H4_score': s, 'count11': cnt, 'count_core3': cnt3, 'core_triple': core.astype(float)}
    for X in ['2A', '2B', '5']:
        comps[f'DP{X}'] = d[f'DP{X}'].values.astype(float)
    # out-of-sample discrimination of fitted models (leave-one-out)
    if len(np.unique(y)) == 2:
        Xall = d[COLS].values.astype(float)
        a_r, b_r, pr_r = loo_ridge_auc(Xall, y, C=1.0)
        out['ridge11_loo_auc'], out['ridge11_loo_brier'] = a_r, b_r
        a_q, b_q, _ = loo_ridge_auc(np.c_[cnt, cnt ** 2] / 11, y, C=1e6)
        out['quadcount_loo_auc'], out['quadcount_loo_brier'] = a_q, b_q
        a_s, b_s, _ = loo_ridge_auc(np.c_[cnt / 11, s], y, C=1e6)
        out['count_plus_score_loo_auc'], out['count_plus_score_loo_brier'] = a_s, b_s
        a_c, b_c, _ = loo_ridge_auc(np.c_[cnt / 11], y, C=1e6)
        out['count_loo_auc'], out['count_loo_brier'] = a_c, b_c
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = {k: [] for k in comps}
    diffs = {k: [] for k in comps if k != 'H4_score'}
    for _ in range(B_BOOT):
        ip_ = rng.choice(idx_pos, len(idx_pos)); in_ = rng.choice(idx_neg, len(idx_neg))
        aa = {k: auc_two(v[ip_], v[in_]) for k, v in comps.items()}
        for k in comps:
            boots[k].append(aa[k])
        for k in diffs:
            diffs[k].append(aa['H4_score'] - aa[k])
    for k, v in comps.items():
        out[f'auc_{k}'] = auc(v, y)
        out[f'auc_{k}_lo'], out[f'auc_{k}_hi'] = [float(q) for q in np.quantile(boots[k], [0.025, 0.975])]
    for k in diffs:
        out[f'dauc_H4_minus_{k}'] = out['auc_H4_score'] - out[f'auc_{k}']
        out[f'dauc_H4_minus_{k}_lo'], out[f'dauc_H4_minus_{k}_hi'] = [float(q) for q in np.quantile(diffs[k], [0.025, 0.975])]
    null = np.array([auc(s, rng.permutation(y)) for _ in range(B_PERM)])
    b_ = int((null >= out['auc_H4_score'] - 1e-12).sum())
    out['H4_perm_b'], out['H4_perm_B'], out['H4_perm_p'] = b_, B_PERM, (b_ + 1) / (B_PERM + 1)
    # incremental likelihood-ratio tests with separation handling and parametric-bootstrap calibration
    lr1 = lr_test(cnt[:, None], np.c_[cnt, s], y, seed=1)
    lr2 = lr_test(np.c_[cnt, cnt ** 2], np.c_[cnt, cnt ** 2, s], y, seed=2)
    lr3 = lr_test(cnt3[:, None], np.c_[cnt3, s], y, seed=3)
    for nm, lr in [('lin', lr1), ('quad', lr2), ('core3count', lr3)]:
        for kk, vv in lr.items():
            out[f'H4_LR_{nm}_{kk}'] = vv
    # calibration of the score as a probability (descriptive): Brier score of the raw class score
    out['H4_brier_raw_score'] = float(np.mean((s - y) ** 2))
    return out, cls, s


def main(txt_path, h4_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20261019)
    txt = open(txt_path, encoding='utf-8', errors='replace').read()
    tt = parse_tables(txt)
    tt.to_csv(os.path.join(outdir, 'baggio2016_truth_tables_parsed.csv'), index=False)
    au = audit(tt[tt.table == 'A4'])
    au.to_csv(os.path.join(outdir, 'reconstruction_audit_A4.csv'), index=False)
    # coding agreement statistics in Table A2 (Fleiss' kappa between the Cox et al. and CBIE codings; a = missing
    # values included, b = missing values excluded)
    kap = []
    for X, lab in LABELS.items():
        m = re.search(re.escape(X) + r':\s*[^\d]*?\s(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)%\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)%', txt.replace('\n', ' '))
        if m:
            kap.append(dict(principle=X, label=lab, kappa_a=float(m.group(2)), agree_a=float(m.group(3)),
                            kappa_b=float(m.group(5)), agree_b=float(m.group(6))))
    pd.DataFrame(kap).to_csv(os.path.join(outdir, 'coding_agreement_A2.csv'), index=False)
    h4 = pd.read_csv(h4_path)
    SC = class_scores(h4)
    json.dump(SC, open(os.path.join(outdir, 'class_scores_all.json'), 'w'), indent=1)
    scores = SC['ref|prod|mean']
    pd.DataFrame([dict(cls=k, score=v) for k, v in scores.items()]).to_csv(os.path.join(outdir, 'class_scores.csv'), index=False)
    res, cases, dec = [], [], []
    D = {}
    for block, name in CODINGS:
        d = expand(tt[(tt.table == 'A4') & (tt.block == block)]).reset_index(drop=True)
        D[name] = d
        o, cls, s = analyse(d, name, scores, rng)
        res.append(o)
        cases.append(d.assign(coding=name, inst_class=cls, score=s))
        for pat in itertools.product([0, 1], repeat=3):
            sel = (d.DP2A == pat[0]) & (d.DP2B == pat[1]) & (d.DP5 == pat[2])
            row = dict(coding=name, DP2A=pat[0], DP2B=pat[1], DP5=pat[2], n=int(sel.sum()), successes=int(d.success[sel].sum()))
            for mp, f in MAPPINGS.items():
                row[mp] = f(*map(bool, pat))
            dec.append(row)
    R = pd.DataFrame(res)
    # Holm adjustment over the confirmatory family (H1 in the three codings, H4 permutation in the two full codings)
    fam = [('H1', i, R.loc[i, 'H1_fisher_p']) for i in range(3)] + [('H4', i, R.loc[i, 'H4_perm_p']) for i in (1, 2)]
    order = sorted(range(len(fam)), key=lambda j: fam[j][2])
    adj = {}
    run = 0
    for rank, j in enumerate(order):
        run = max(run, min(1, (len(fam) - rank) * fam[j][2]))
        adj[j] = run
    for j, (h, i, pv) in enumerate(fam):
        R.loc[i, f'{h}_p_holm'] = adj[j]
    R.to_csv(os.path.join(outdir, 'field_results.csv'), index=False)
    pd.concat(cases).to_csv(os.path.join(outdir, 'baggio2016_cases_reconstructed.csv'), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(outdir, 'class_decision_table.csv'), index=False)
    # sensitivity: alternative mappings x score definitions x families x reference environments (full codings)
    sens = []
    for name in ['missing = absent', 'missing = present', 'complete cases']:
        d = D[name]; y = d.success.values
        for mp in MAPPINGS:
            cls = classes(d, mp)
            for key, sc in SC.items():
                s = np.array([sc[c] for c in cls])
                sens.append(dict(coding=name, mapping=mp, score=key, auc=auc(s, y),
                                 amnesty_gt_permanent=bool(sc['THRESHOLD-AMNESTY'] > sc['THRESHOLD-PERMANENT'])))
    pd.DataFrame(sens).to_csv(os.path.join(outdir, 'field_sensitivity_mapping_score.csv'), index=False)
    # outcome-dependent extreme completions (successes coded under one full coding, failures under the other)
    comp = []
    for sc_name in ['missing = absent', 'missing = present']:
        for fc_name in ['missing = absent', 'missing = present']:
            ds, df_ = D[sc_name], D[fc_name]
            S_ = ds[ds.success == 1]; F_ = df_[df_.success == 0]
            cs_ = (S_.DP2A & S_.DP2B & S_.DP5).values; cf_ = (F_.DP2A & F_.DP2B & F_.DP5).values
            a, b = cs_.sum(), cf_.sum(); c2, d2 = len(cs_) - a, len(cf_) - b
            ss = np.array([scores[c] for c in classes(S_)]); sf = np.array([scores[c] for c in classes(F_)])
            comp.append(dict(success_coding=sc_name, failure_coding=fc_name, core_succ=int(a), core_fail=int(b),
                             rest_succ=int(c2), rest_fail=int(d2), rd=a / (a + b) - c2 / (c2 + d2),
                             fisher_p=float(stats.fisher_exact([[a, b], [c2, d2]], alternative='greater')[1]),
                             H4_auc=auc_two(ss, sf)))
    pd.DataFrame(comp).to_csv(os.path.join(outdir, 'field_outcome_dependent_completions.csv'), index=False)
    # non-differential misclassification of single codes
    mis = []
    for name in ['missing = absent', 'missing = present', 'complete cases']:
        d0 = D[name]; y = d0.success.values
        for X in ['2A', '2B', '5']:
            for q in [0.05, 0.1, 0.2, 0.3]:
                rd, au_ = [], []
                for _ in range(1000):
                    d = d0.copy()
                    flip = rng.random(len(d)) < q
                    d[f'DP{X}'] = np.where(flip, 1 - d[f'DP{X}'], d[f'DP{X}'])
                    core = (d.DP2A & d.DP2B & d.DP5).values
                    if core.sum() and (1 - core).sum():
                        rd.append(y[core == 1].mean() - y[core == 0].mean())
                    au_.append(auc(np.array([scores[c] for c in classes(d)]), y))
                mis.append(dict(coding=name, principle=X, flip_prob=q, rd_mean=np.mean(rd), rd_lo=np.quantile(rd, 0.025),
                                rd_hi=np.quantile(rd, 0.975), rd_pos_frac=float(np.mean(np.array(rd) > 0)),
                                auc_mean=np.mean(au_), auc_lo=np.quantile(au_, 0.025), auc_hi=np.quantile(au_, 0.975)))
    pd.DataFrame(mis).to_csv(os.path.join(outdir, 'field_misclassification.csv'), index=False)
    # complete versus incomplete cases (selection into complete-case analysis)
    full = D['missing = absent']
    sel = dict(complete_success=int(D['complete cases'].success.sum()), complete_n=len(D['complete cases']),
               incomplete_success=int(full.success.sum() - D['complete cases'].success.sum()),
               incomplete_n=len(full) - len(D['complete cases']))
    json.dump(sel, open(os.path.join(outdir, 'field_complete_vs_incomplete.json'), 'w'))
    print(R[['coding', 'core_succ', 'core_n', 'rest_succ', 'rest_n', 'H1_rd', 'H1_fisher_p', 'auc_H4_score',
             'auc_count11', 'auc_count_core3', 'H4_perm_p']].to_string())
    print(pd.DataFrame(comp).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
