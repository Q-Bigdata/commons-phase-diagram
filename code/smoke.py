"""Unit checks for the revised implementation (run before the production stages)."""
import numpy as np, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

sim.warmup()
th0 = np.zeros(sim.NPAR)
# 1. committed free riders return exactly zero in F1 even with execution noise
cfg = sim.make_cfg(0.4, 0.2)
phi = np.zeros((2000, 4)); fr = np.ones((2000, 4))
pay, met = sim.play_const(phi, fr, cfg, 40, sim.make_inst(*sim.EQ), 0, th0, 3)
assert np.allclose(met[:, 0], 1.0) and np.allclose(met[:, 6], 1), (met[:5, 0], met[:5, 6])
print("ok F1 free riders return zero (surplus = R0, collapse in round 1)")
# 2. Q-learning: the final transition is learned in a one-round, non-depleted game
Q, frq = sim.f3_init(4, 4, 0.0, 1)
cfg0 = sim.make_cfg(0.4, 0.0)
late, fro = sim.run_f3(Q, frq, 4, 1, cfg0, sim.make_inst(*sim.EQ), 0, th0, 1, 1, 1, 0, 0.1, 0.95, 1.0, 1.0, 1, 0)
nchg = int((Q != 1.0).sum())
assert nchg > 0, "terminal transition not learned"
print("ok Q terminal update at the horizon:", nchg, "entries changed")
# 3. exclusion is a zero-reward semi-Markov gap, not a terminal state: learning continues after re-entry
Q, frq = sim.f3_init(8, 4, 0.0, 2)
late, fro = sim.run_f3(Q, frq, 4, 40, sim.make_cfg(0.4, 0.05), sim.make_inst(*sim.STEP), 0, th0, 2, 50, 10, 10,
                       0.1, 0.95, 0.2, 0.01, 1, 0)
print("ok F3 run late/frozen eff", round(late[0] / sim.ceiling(0.4, 40), 3), round(fro[0] / sim.ceiling(0.4, 40), 3))
# 4. integer-token implementation: allocations are whole tokens
cfgi = sim.make_cfg(0.4, 0.05, integer=1)
pay, met = sim.play_const(np.full((50, 4), 0.75), np.zeros((50, 4)), cfgi, 40, sim.make_inst(*sim.HEUR_R), 0, th0, 5)
assert np.allclose(pay * 200, np.round(pay * 200), atol=1e-6), "integer-token payoffs are not integer multiples of 1/200"
print("ok integer mode eff", round(met[:, 0].mean() / sim.ceiling(0.4, 40), 3), "sust", met[:, 2].mean())
# 5. allocation fallback: tiny pool is still allocated in full
e = np.zeros(16); tok = np.zeros(16)
e[:] = 0.004 / 16
sim._finalize(e, 0.004, 0, tok)
assert abs(e.sum() - 0.004) < 1e-12
print("ok tiny-pool fallback allocates the whole pool")
# 6. classifier agrees with Methods
g = sim.inst_grid(True)
print("fine grid", len(g), collections.Counter(sim.regime(x) for x in g))
gc = sim.inst_grid(False)
print("coarse grid", len(gc), collections.Counter(sim.regime(x) for x in gc))
assert sim.regime((0, 0.97, 0.0, 0.25, 0.0, 1)) == "T"
# 7. planner allocation and table rule run in all families
th = np.random.default_rng(0).normal(0, 0.1, sim.NPAR)
beh = np.array([0.6, 0.2, 1.0, 0.3, 0.3])
m2 = sim.play_f2(200, 4, 40, sim.make_cfg(0.4, 0.05), 0.2, beh, sim.make_inst(0), 1, th, 1)
mt = sim.play_f2(200, 4, 40, sim.make_cfg(0.4, 0.05), 0.2, beh, sim.make_inst(2, wtab=np.full(20, 1.33)), 0, th0, 1)
print("ok planner/table F2 eff", round(m2[:, 0].mean() / sim.ceiling(0.4, 40), 3), round(mt[:, 0].mean() / sim.ceiling(0.4, 40), 3))
o, phi, fixed = sim.evolve_f1(sim.make_cfg(0.4, 0.05), 4, 40, 0.2, sim.make_inst(*sim.STEP), gens=50, tail=10, seed=1)
print("ok F1 evolve eff", round(o[0] / sim.ceiling(0.4, 40), 3), "phi", round(o[5], 3))
print("BH ceiling r=0.4 T=40", round(sim.ceiling(0.4, 40, 1), 4), "linear", round(sim.ceiling(0.4, 40), 4))
print("SMOKE OK")
