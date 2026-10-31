"""Summary tables quoted in the manuscript and Extended Data (run after process.py, on the server).
usage: python3 ms_tables.py <proc_dir>"""
import os, sys, json
import numpy as np
import pandas as pd

P = sys.argv[1]
# Extended Data Table 4: reconstruction audit by coding block
au = pd.read_csv(os.path.join(P, "reconstruction_audit_A4.csv"))
rows = []
for b, d in au.groupby("block", sort=False):
    rows.append(dict(block=b, rows=len(d), unique=int(d.unique.sum()), mixed=int(d.mixed.sum()),
                     cases=int(d.n.sum()), successes=int(d.assigned_successes.sum()),
                     ambiguous_cases=int(d[~d.unique].n.sum())))
pd.DataFrame(rows).to_csv(os.path.join(P, "ms_audit_summary.csv"), index=False)
# Extended Data Fig. 5 / SI: median outcome by family and policy
mo = pd.read_csv(os.path.join(P, "multiobjective.csv"))
cols = ["eff", "sust", "act", "gini", "minR", "worst", "ever_excl"]
mo.groupby(["family", "policy"])[cols].median().reset_index().to_csv(os.path.join(P, "ms_multiobjective_medians.csv"),
                                                                      index=False, float_format="%.3f")
# Table 2: eleven contrasts, human estimates and family classifications (continuous implementation)
hf = pd.read_csv(os.path.join(P, "human_facts.csv"))
tc = pd.read_csv(os.path.join(P, "t3_contrasts.csv"))
t = tc[(tc.integer == 0) & tc.family.isin(["F1", "F2", "F3"])].pivot(index="id", columns="family", values="cls")
ti = tc[(tc.integer == 1) & (tc.family == "F2")].set_index("id").cls.rename("F2_integer")
pl = tc[tc.family.str.startswith("F2plaus")].groupby("id").cls.apply(lambda s: int((s == "yes").sum())).rename("F2_plaus_yes")
out = hf.set_index("id")[["fact", "source", "calibrated_on", "mean_a", "mean_b", "human_diff", "lo", "hi", "perm_p",
                          "human_sign_supported"]].join(t).join(ti).join(pl)
out["n_plaus"] = tc[tc.family.str.startswith("F2plaus")].family.nunique()
out.reset_index().to_csv(os.path.join(P, "ms_table2_contrasts.csv"), index=False, float_format="%.4f")
# robustness summary by slice (if available)
if os.path.exists(os.path.join(P, "robust_summary.csv")):
    RB = pd.read_csv(os.path.join(P, "robust_summary.csv"))
    g = RB.groupby(["slice", "family"]).apply(lambda q: pd.Series(dict(n=len(q), T=int((q.best_regime == "T").sum()),
                                                                     dT_med=float(q.dT.median()))))
    g.reset_index().to_csv(os.path.join(P, "ms_robust_by_slice.csv"), index=False, float_format="%.4f")
fs = pd.read_csv(os.path.join(P, "field_sensitivity_mapping_score.csv"))
fs.groupby(["coding", "mapping"]).auc.agg(["min", "median", "max"]).reset_index().to_csv(
    os.path.join(P, "ms_field_sens_summary.csv"), index=False, float_format="%.4f")
print("ok")
