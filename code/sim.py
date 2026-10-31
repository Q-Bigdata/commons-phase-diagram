"""Revised simulation core (revision 1) for the dynamic common-pool trust game of Koster, Pislar et al.
(2025, Nat. Commun. 16:2824) and three behavioural model families.

Changes relative to the submitted version (all prompted by the referee audit):
  * one authoritative regime classifier (P: both weights <= 0.1; E: both >= 0.75; T: w_full - w_scarce >= 0.2;
    I: w_scarce - w_full >= 0.2; M otherwise), identical to the Methods;
  * committed free riders return exactly zero in every family (no execution noise on a committed zero);
  * Q-learners: the pending transition is closed at the normal horizon (terminal update), and rounds of
    exclusion are zero-reward, non-decision rounds of a semi-Markov transition that bootstraps on re-entry
    (gamma ** gap), instead of terminating credit assignment at exclusion;
  * Q-learners return both late-training metrics and the metrics of the frozen greedy policy;
  * an explicit fallback when the proportional denominator is zero (equal shares) and when every offer is
    below the half-token granularity (the pool goes to the largest offer), so that the whole pool is always
    allocated while R >= 1e-3;
  * optional structural variants: integer-token implementation (200-token pool, largest-remainder offers,
    integer returns, depletion below one token), density-dependent (Beverton-Holt) regeneration, noisy and/or
    one-round-delayed stock observation by the institution, noisy contribution records, correlated execution
    noise, rounded deployment version of the distilled rule, tabulated w(R) (executable distilled planners);
  * the neural planner is part of the same allocation interface, so that every family can be played against
    either a rule or a planner, and learners can be pre-adapted to one allocation policy and then frozen;
  * extended per-game metrics (stock risk, concentration of allocations, aggregate-return variability,
    exclusion spells, worst-off share).

Units: pool capacity R0 = 1 (200 tokens). Institution vector ip (length 26):
  ip[0] kind (0 step, 1 power, 2 table, 3 power rounded to 0.1 as deployed by Koster et al.), ip[1] theta,
  ip[2] w_lo, ip[3] w_hi, ip[4] k, ip[5] amnesty (1 re-inclusion; 0 permanent exclusion), ip[6:26] w table (20 bins)
Configuration vector cfg (length 9):
  r, sig (execution s.d.), eps (log s.d. of mean-one lognormal regeneration shock), rho (pairwise correlation of
  execution noise), gk (0 capped linear, 1 Beverton-Holt), obs (log s.d. of stock observation), delay (0/1),
  crec (log s.d. of contribution-record noise), integer (0/1).
"""
import numpy as np
from numba import njit

EMIN = 0.0025          # half a token
NM = 17                # per-game metrics
NF, H1, H2 = 6, 16, 16
NPAR = NF * H1 + H1 + H1 * H2 + H2 + H2 + 1   # 401 planner parameters
DEAD_C, DEAD_I = 1e-3, 0.005                  # depletion thresholds (continuous / one token)
METRICS = ["surplus", "gini", "sust", "act", "all4", "phi", "tdead", "minR", "termR", "below_half",
           "conc", "sdPhi", "worst", "spell", "ever_excl", "sust_act", "act_alive"]


def make_cfg(r, sig=0.05, eps=0.0, rho=0.0, gk=0, obs=0.0, delay=0, crec=0.0, integer=0):
    return np.array([r, sig, eps, rho, gk, obs, delay, crec, integer], dtype=np.float64)


def make_inst(kind, theta=0.0, wlo=0.0, whi=0.0, k=0.0, amnesty=1, wtab=None):
    ip = np.zeros(26)
    ip[0], ip[1], ip[2], ip[3], ip[4], ip[5] = kind, theta, wlo, whi, k, amnesty
    if wtab is not None:
        ip[6:26] = np.asarray(wtab, dtype=np.float64)
    return ip


def inst_from_tuple(t):
    return make_inst(*t)


@njit(cache=True)
def w_of(Rn, ip):
    kind = int(ip[0])
    if kind == 0:
        return ip[3] if Rn >= ip[1] - 1e-12 else ip[2]
    if kind == 1:
        return min(Rn, 1.0) ** ip[4]
    if kind == 3:
        return np.floor(min(Rn, 1.0) ** ip[4] * 10.0 + 0.5) / 10.0
    j = int(Rn * 20.0)
    if j > 19:
        j = 19
    if j < 0:
        j = 0
    return ip[6 + j]


@njit(cache=True)
def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True)
def _gini_sub(x, mask):
    n = 0
    for i in range(x.shape[0]):
        if mask[i] < 0.5:
            n += 1
    if n == 0:
        return 0.0
    y = np.empty(n)
    q = 0
    s = 0.0
    for i in range(x.shape[0]):
        if mask[i] < 0.5:
            y[q] = x[i]
            s += x[i]
            q += 1
    if s <= 1e-12:
        return 0.0
    y = np.sort(y)
    acc = 0.0
    for i in range(n):
        acc += (i + 1) * y[i]
    return 2.0 * acc / (n * s) - (n + 1.0) / n


# ------------------------------------------------------------------------------------------------
# allocation: rule or neural planner, followed by token granularity
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def _unpack(th):
    o = 0
    W1 = th[o:o + NF * H1].reshape(NF, H1); o += NF * H1
    b1 = th[o:o + H1]; o += H1
    W2 = th[o:o + H1 * H2].reshape(H1, H2); o += H1 * H2
    b2 = th[o:o + H2]; o += H2
    W3 = th[o:o + H2]; o += H2
    b3 = th[o]
    return W1, b1, W2, b2, W3, b3


@njit(cache=True)
def _raw_alloc(use_pl, th, ip, Rtrue, Robs, crec, e_prev, excl_run, t, e_out):
    """Unrounded allocation (sums to Rtrue).  Rule: e_i = w eq_i + (1-w) prop_i.  Planner: softmax scores."""
    p = crec.shape[0]
    if t == 0:
        for i in range(p):
            e_out[i] = Rtrue / p
        return
    sc = 0.0
    nel = 0
    for i in range(p):
        sc += crec[i]
        if crec[i] > 1e-12:
            nel += 1
    if use_pl == 0:
        w = w_of(Robs, ip)
        am = int(ip[5])
        tot = 0.0
        for i in range(p):
            prop = Rtrue * crec[i] / sc if sc > 1e-12 else Rtrue / p   # zero-denominator fallback: equal
            if am == 1 or nel == 0:
                eq = Rtrue / p
            else:
                eq = Rtrue / nel if crec[i] > 1e-12 else 0.0
            v = w * eq + (1.0 - w) * prop
            if v < 0.0:
                v = 0.0                    # tabulated w > 1 (compensatory) can imply negative offers
            e_out[i] = v
            tot += v
        if tot > 1e-12:
            for i in range(p):
                e_out[i] *= Rtrue / tot
        else:
            for i in range(p):
                e_out[i] = Rtrue / p
        return
    W1, b1, W2, b2, W3, b3 = _unpack(th)
    se = 0.0
    for i in range(p):
        se += e_prev[i]
    gret = sc / se if se > 1e-12 else 0.0
    scores = np.zeros(p)
    feat = np.zeros(NF)
    h1 = np.zeros(H1)
    h2 = np.zeros(H2)
    for i in range(p):
        csh = crec[i] / sc if sc > 1e-12 else 1.0 / p
        feat[0] = Robs
        feat[1] = np.log(p * csh + 0.01)
        feat[2] = crec[i] / e_prev[i] if e_prev[i] > 0.0 else 0.0
        feat[3] = gret
        feat[4] = 1.0 if e_prev[i] <= 0.0 else 0.0
        feat[5] = min(excl_run[i], 5.0) / 5.0
        for a in range(H1):
            s = b1[a]
            for q in range(NF):
                s += feat[q] * W1[q, a]
            h1[a] = np.tanh(s)
        for a in range(H2):
            s = b2[a]
            for q in range(H1):
                s += h1[q] * W2[q, a]
            h2[a] = np.tanh(s)
        s = b3
        for q in range(H2):
            s += h2[q] * W3[q]
        scores[i] = s
    m = scores.max()
    z = 0.0
    for i in range(p):
        scores[i] = np.exp(scores[i] - m)
        z += scores[i]
    for i in range(p):
        e_out[i] = Rtrue * scores[i] / z


@njit(cache=True)
def _finalize(e, Rtrue, integer, tok):
    """Token granularity.  Continuous: offers below half a token are zeroed and the rest renormalised; if all
    offers are below half a token the whole pool goes to the largest offer.  Integer: largest-remainder
    rounding of shares to the integer number of tokens in the pool.  tok receives integer token counts."""
    p = e.shape[0]
    if integer == 0:
        tot = 0.0
        imax = 0
        for i in range(p):
            if e[i] > e[imax]:
                imax = i
        for i in range(p):
            tok[i] = 0.0
            if e[i] < EMIN:
                e[i] = 0.0
            tot += e[i]
        if tot > 1e-12:
            for i in range(p):
                e[i] *= Rtrue / tot
        else:
            for i in range(p):
                e[i] = 0.0
            e[imax] = Rtrue
        return
    N = np.floor(Rtrue * 200.0 + 1e-9)
    s = 0.0
    for i in range(p):
        s += e[i]
    fr = np.zeros(p)
    used = 0.0
    for i in range(p):
        q = e[i] / s if s > 1e-12 else 1.0 / p
        x = q * N
        tok[i] = np.floor(x + 1e-9)
        fr[i] = x - tok[i]
        used += tok[i]
    rem = int(N - used + 0.5)
    for _ in range(rem):
        j = 0
        for i in range(1, p):
            if fr[i] > fr[j]:
                j = i
        tok[j] += 1.0
        fr[j] = -1.0
    for i in range(p):
        e[i] = tok[i] / 200.0


@njit(cache=True)
def _regrow(S, cfg, xi):
    r = cfg[0]
    if cfg[4] < 0.5:
        return min(1.0, (1.0 + r) * S * xi)
    return min(1.0, (1.0 + r) * S / (1.0 + r * S) * xi)


@njit(cache=True)
def _xi(cfg):
    eps = cfg[2]
    if eps > 0.0:
        return np.exp(eps * np.random.randn() - 0.5 * eps * eps)
    return 1.0


@njit(cache=True)
def _observe(R, Rlag, t, cfg):
    src = Rlag if (cfg[6] > 0.5 and t > 0) else R
    if cfg[5] > 0.0:
        return src * np.exp(cfg[5] * np.random.randn() - 0.5 * cfg[5] * cfg[5])
    return src


@njit(cache=True)
def _record(c, crec, cfg):
    for i in range(c.shape[0]):
        if cfg[7] > 0.0 and c[i] > 0.0:
            crec[i] = c[i] * np.exp(cfg[7] * np.random.randn() - 0.5 * cfg[7] * cfg[7])
        else:
            crec[i] = c[i]


@njit(cache=True)
def _noisy(ph, cfg, z0):
    sig = cfg[1]
    if sig <= 0.0:
        return ph
    rho = cfg[3]
    if rho > 0.0:
        z = np.sqrt(rho) * z0 + np.sqrt(1.0 - rho) * np.random.randn()
    else:
        z = np.random.randn()
    return _clip01(ph + sig * z)


@njit(cache=True)
def _ret(ph, e_i, tok_i, integer):
    if integer == 0:
        return ph * e_i
    return np.floor(ph * tok_i + 0.5) / 200.0


# ------------------------------------------------------------------------------------------------
# per-game bookkeeping shared by all families
# acc: 0 act, 1 last_all, 2 phisum, 3 phin, 4 minR, 5 nbelow, 6 conc, 7 nround, 8 Phisum, 9 Phisq,
#      10 spells, 11 spell_len, 12 tdead, 13 alive
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def _acc_reset(acc, cur, ever, T):
    for q in range(acc.shape[0]):
        acc[q] = 0.0
    acc[4] = 1.0
    acc[12] = T
    acc[13] = 1.0
    for i in range(cur.shape[0]):
        cur[i] = 0.0
        ever[i] = 0.0


@njit(cache=True)
def _acc_round(acc, e, c, R, fr, cur, ever, t):
    p = e.shape[0]
    na = 0
    se = 0.0
    sc = 0.0
    conc = 0.0
    for i in range(p):
        se += e[i]
        sc += c[i]
        if e[i] > 0.0:
            na += 1
    for i in range(p):
        a = e[i] / se if se > 1e-12 else 0.0
        conc += a * a
        if fr[i] < 0.5 and t > 0:
            if e[i] <= 0.0:
                cur[i] += 1.0
                ever[i] = 1.0
            elif cur[i] > 0.0:
                acc[10] += 1.0
                acc[11] += cur[i]
                cur[i] = 0.0
    acc[0] += na / p
    acc[1] = 1.0 if na == p else 0.0
    if R < acc[4]:
        acc[4] = R
    if R < 0.5:
        acc[5] += 1.0
    acc[6] += conc
    acc[7] += 1.0
    Phi = sc / se if se > 1e-12 else 0.0
    acc[8] += Phi
    acc[9] += Phi * Phi
    return sc


@njit(cache=True)
def _acc_finish(row, acc, pay, fr, cur, R_end, T, r):
    p = pay.shape[0]
    alive = acc[13] > 0.5
    for i in range(p):
        if fr[i] < 0.5 and cur[i] > 0.0:
            acc[10] += 1.0
            acc[11] += cur[i]
    tot = 0.0
    for i in range(p):
        tot += pay[i]
    row[0] = tot
    row[1] = _gini_sub(pay, fr)
    row[2] = 1.0 if alive else 0.0
    row[3] = acc[0] / T
    row[4] = acc[1] if alive else 0.0
    row[5] = acc[2] / acc[3] if acc[3] > 0 else 0.0
    row[6] = acc[12]
    row[7] = acc[4] if alive else 0.0
    row[8] = R_end if alive else 0.0
    row[9] = (acc[5] + (T - acc[7])) / T
    row[10] = acc[6] / acc[7] if acc[7] > 0 else 1.0
    n = acc[7]
    if n > 1:
        mu = acc[8] / n
        v = acc[9] / n - mu * mu
        row[11] = np.sqrt(v * n / (n - 1)) if v > 0 else 0.0
    else:
        row[11] = 0.0
    ns = 0
    sm = 0.0
    mn = 1e18
    ne = 0.0
    for i in range(p):
        if fr[i] < 0.5:
            ns += 1
            sm += pay[i]
            if pay[i] < mn:
                mn = pay[i]
            ne += cur[i] * 0.0
    row[12] = (mn / (sm / ns)) if (ns > 0 and sm > 1e-12) else 0.0
    row[13] = acc[11] / acc[10] if acc[10] > 0 else 0.0
    ev = 0.0
    return ns


@njit(cache=True)
def _finish_ever(row, ever, fr):
    ns = 0
    ev = 0.0
    for i in range(ever.shape[0]):
        if fr[i] < 0.5:
            ns += 1
            ev += ever[i]
    row[14] = ev / ns if ns > 0 else 0.0
    row[15] = row[2] * row[3]
    row[16] = row[3] * 0.0


# ------------------------------------------------------------------------------------------------
# F1: fixed per-player return fractions (one game per row), used inside the evolutionary loop
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def play_const(phi, fr, cfg, T, ip, use_pl, th, seed):
    np.random.seed(seed)
    G, p = phi.shape
    integer = int(cfg[8])
    dead = DEAD_I if integer == 1 else DEAD_C
    pay = np.zeros((G, p))
    met = np.zeros((G, NM))
    e = np.zeros(p); c = np.zeros(p); crec = np.zeros(p); tok = np.zeros(p)
    e_prev = np.zeros(p); excl = np.zeros(p)
    acc = np.zeros(14); cur = np.zeros(p); ever = np.zeros(p)
    pg = np.zeros(p)
    for g in range(G):
        R = 1.0
        Rlag = 1.0
        for i in range(p):
            c[i] = 0.0; crec[i] = 0.0; e_prev[i] = 0.0; excl[i] = 0.0
        _acc_reset(acc, cur, ever, T)
        nrounds = 0
        for t in range(T):
            Robs = _observe(R, Rlag, t, cfg)
            _raw_alloc(use_pl, th, ip, R, Robs, crec, e_prev, excl, t, e)
            _finalize(e, R, integer, tok)
            z0 = np.random.randn() if cfg[3] > 0.0 else 0.0
            for i in range(p):
                if fr[g, i] > 0.5:
                    ph = 0.0
                else:
                    ph = _noisy(phi[g, i], cfg, z0)
                    if e[i] > 0.0:
                        acc[2] += ph; acc[3] += 1.0
                c[i] = _ret(ph, e[i], tok[i], integer)
                pay[g, i] += e[i] - c[i]
            sc = _acc_round(acc, e, c, R, fr[g], cur, ever, t)
            _record(c, crec, cfg)
            for i in range(p):
                if e[i] > 0.0:
                    excl[i] = 0.0
                else:
                    excl[i] += 1.0
                e_prev[i] = e[i]
            Rlag = R
            R = _regrow(sc, cfg, _xi(cfg))
            nrounds += 1
            if R < dead:
                acc[13] = 0.0
                acc[12] = t + 1
                break
        for i in range(p):
            pg[i] = pay[g, i]
        _acc_finish(met[g], acc, pg, fr[g], cur, R, T, cfg[0])
        _finish_ever(met[g], ever, fr[g])
        met[g, 16] = acc[0] / nrounds if nrounds > 0 else 0.0
    return pay, met


# ------------------------------------------------------------------------------------------------
# F2: heterogeneous conditional reciprocators (non-strategic)
#   phi_it = b_i + lam (obar_-i,t-1 - b_i) + gam clip(rho_it - 1, -1, 1) + N(0, sd_int^2 + sig^2)
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def play_f2(G, p, T, cfg, f, beh, ip, use_pl, th, seed):
    np.random.seed(seed)
    mu_b, sd_b, lam, gam, sd_int = beh[0], beh[1], beh[2], beh[3], beh[4]
    integer = int(cfg[8])
    dead = DEAD_I if integer == 1 else DEAD_C
    met = np.zeros((G, NM))
    e = np.zeros(p); c = np.zeros(p); crec = np.zeros(p); tok = np.zeros(p)
    e_prev = np.zeros(p); excl = np.zeros(p)
    phi_prev = np.zeros(p); got_prev = np.zeros(p); b = np.zeros(p); fr = np.zeros(p); pay = np.zeros(p)
    acc = np.zeros(14); cur = np.zeros(p); ever = np.zeros(p)
    sd_beh = np.sqrt(sd_int * sd_int)
    cfg2 = cfg.copy()
    sdt = np.sqrt(sd_int * sd_int + cfg[1] * cfg[1])
    for g in range(G):
        for i in range(p):
            b[i] = _clip01(mu_b + sd_b * np.random.randn())
            fr[i] = 1.0 if np.random.rand() < f else 0.0
            pay[i] = 0.0; c[i] = 0.0; crec[i] = 0.0; e_prev[i] = 0.0; excl[i] = 0.0
            phi_prev[i] = b[i]; got_prev[i] = 1.0
        R = 1.0
        Rlag = 1.0
        _acc_reset(acc, cur, ever, T)
        nrounds = 0
        for t in range(T):
            Robs = _observe(R, Rlag, t, cfg)
            _raw_alloc(use_pl, th, ip, R, Robs, crec, e_prev, excl, t, e)
            _finalize(e, R, integer, tok)
            z0 = np.random.randn() if cfg[3] > 0.0 else 0.0
            for i in range(p):
                s_o = 0.0
                n_o = 0
                for j in range(p):
                    if j != i and got_prev[j] > 0.5:
                        s_o += phi_prev[j]
                        n_o += 1
                ob = s_o / n_o if n_o > 0 else b[i]
                if e[i] > 0.0 and fr[i] < 0.5:
                    rho = e[i] / (R / p)
                    d = rho - 1.0
                    d = -1.0 if d < -1.0 else (1.0 if d > 1.0 else d)
                    base = b[i] + lam * (ob - b[i]) + gam * d
                    # intrinsic and execution noise combined (correlation applies to the execution part)
                    if cfg[3] > 0.0 and cfg[1] > 0.0:
                        ph = _clip01(base + sd_int * np.random.randn()
                                     + cfg[1] * (np.sqrt(cfg[3]) * z0 + np.sqrt(1 - cfg[3]) * np.random.randn()))
                    else:
                        ph = _clip01(base + sdt * np.random.randn())
                    acc[2] += ph; acc[3] += 1.0
                else:
                    ph = 0.0
                c[i] = _ret(ph, e[i], tok[i], integer)
                pay[i] += e[i] - c[i]
            sc = _acc_round(acc, e, c, R, fr, cur, ever, t)
            for i in range(p):
                if e[i] > 0.0:
                    phi_prev[i] = c[i] / e[i]
                    got_prev[i] = 1.0
                    excl[i] = 0.0
                else:
                    got_prev[i] = 0.0
                    excl[i] += 1.0
                e_prev[i] = e[i]
            _record(c, crec, cfg)
            Rlag = R
            R = _regrow(sc, cfg, _xi(cfg))
            nrounds += 1
            if R < dead:
                acc[13] = 0.0
                acc[12] = t + 1
                break
        _acc_finish(met[g], acc, pay, fr, cur, R, T, cfg[0])
        _finish_ever(met[g], ever, fr)
        met[g, 16] = acc[0] / nrounds if nrounds > 0 else 0.0
    return met


@njit(cache=True)
def trace_f2(G, p, T, cfg, f, beh, ip, use_pl, th, seed):
    """F2 rollouts recording pool, allocation shares, contribution shares and exclusion runs (distillation)."""
    np.random.seed(seed)
    mu_b, sd_b, lam, gam, sd_int = beh[0], beh[1], beh[2], beh[3], beh[4]
    R_tr = np.full((G, T), -1.0)
    sh_tr = np.zeros((G, T, p))
    cs_tr = np.zeros((G, T, p))
    ex_tr = np.zeros((G, T, p))
    e = np.zeros(p); c = np.zeros(p); crec = np.zeros(p); tok = np.zeros(p)
    e_prev = np.zeros(p); excl = np.zeros(p)
    phi_prev = np.zeros(p); got_prev = np.zeros(p); b = np.zeros(p); fr = np.zeros(p)
    sdt = np.sqrt(sd_int * sd_int + cfg[1] * cfg[1])
    for g in range(G):
        for i in range(p):
            b[i] = _clip01(mu_b + sd_b * np.random.randn())
            fr[i] = 1.0 if np.random.rand() < f else 0.0
            c[i] = 0.0; crec[i] = 0.0; e_prev[i] = 0.0; excl[i] = 0.0
            phi_prev[i] = b[i]; got_prev[i] = 1.0
        R = 1.0
        for t in range(T):
            _raw_alloc(use_pl, th, ip, R, R, crec, e_prev, excl, t, e)
            _finalize(e, R, 0, tok)
            R_tr[g, t] = R
            scp = 0.0
            for i in range(p):
                scp += crec[i]
            for i in range(p):
                sh_tr[g, t, i] = e[i] / R
                cs_tr[g, t, i] = crec[i] / scp if scp > 1e-12 else 1.0 / p
                ex_tr[g, t, i] = excl[i]
            sc = 0.0
            for i in range(p):
                s_o = 0.0
                n_o = 0
                for j in range(p):
                    if j != i and got_prev[j] > 0.5:
                        s_o += phi_prev[j]
                        n_o += 1
                ob = s_o / n_o if n_o > 0 else b[i]
                if e[i] > 0.0 and fr[i] < 0.5:
                    d = e[i] / (R / p) - 1.0
                    d = -1.0 if d < -1.0 else (1.0 if d > 1.0 else d)
                    ph = _clip01(b[i] + lam * (ob - b[i]) + gam * d + sdt * np.random.randn())
                else:
                    ph = 0.0
                c[i] = ph * e[i]
                sc += c[i]
            for i in range(p):
                if e[i] > 0.0:
                    phi_prev[i] = c[i] / e[i]; got_prev[i] = 1.0; excl[i] = 0.0
                else:
                    got_prev[i] = 0.0; excl[i] += 1.0
                e_prev[i] = e[i]
                crec[i] = c[i]
            R = _regrow(sc, cfg, _xi(cfg))
            if R < DEAD_C:
                break
    return R_tr, sh_tr, cs_tr, ex_tr


# ------------------------------------------------------------------------------------------------
# F3: independent tabular Q-learners
#   qv = 0: pool bins (0.5, 0.8, 0.97), 11 actions (default);  qv = 1: shifted pool bins (0.4, 0.7, 0.9);
#   qv = 2: finer pool bins (0.3, 0.5, 0.7, 0.85, 0.95) and 21 actions
# ------------------------------------------------------------------------------------------------
@njit(cache=True)
def _state(Rn, rho, gret, qv):
    if qv == 0:
        a = 0 if Rn < 0.5 else (1 if Rn < 0.8 else (2 if Rn < 0.97 else 3))
    elif qv == 1:
        a = 0 if Rn < 0.4 else (1 if Rn < 0.7 else (2 if Rn < 0.9 else 3))
    else:
        a = 0 if Rn < 0.3 else (1 if Rn < 0.5 else (2 if Rn < 0.7 else (3 if Rn < 0.85 else (4 if Rn < 0.95 else 5))))
    bb = 0 if rho < 0.75 else (1 if rho < 1.25 else 2)
    cc = 0 if gret < 0.5 else (1 if gret < 0.75 else 2)
    return a * 9 + bb * 3 + cc


@njit(cache=True)
def run_f3(Q, fr, p, T, cfg, ip, use_pl, th, seed, n_ep, n_eval, n_frozen, alpha, gamma, eps0, eps1, learn, qv):
    """Q: (G, p, 54, 21) initial tables (modified in place); fr: (G, p) committed free riders.
    Trains for n_ep games (learning if learn == 1, epsilon annealed from eps0 to eps1 over the first
    n_ep - n_eval games), recording metrics over the last n_eval games (late-training estimand), then plays
    n_frozen games with the frozen greedy policy (frozen estimand).  Returns (met_late, met_frozen)."""
    np.random.seed(seed)
    G = Q.shape[0]
    NA = 21 if qv == 2 else 11
    acts = np.linspace(0.0, 1.0, NA)
    integer = int(cfg[8])
    dead = DEAD_I if integer == 1 else DEAD_C
    out_late = np.zeros(NM)
    out_fro = np.zeros(NM)
    nl = 0.0
    nf = 0.0
    e = np.zeros(p); c = np.zeros(p); crec = np.zeros(p); tok = np.zeros(p)
    e_prev = np.zeros(p); excl = np.zeros(p)
    s_prev = np.zeros(p, np.int64); a_prev = np.zeros(p, np.int64); r_prev = np.zeros(p)
    has_prev = np.zeros(p); gap = np.zeros(p)
    pay = np.zeros(p)
    row = np.zeros(NM)
    acc = np.zeros(14); cur = np.zeros(p); ever = np.zeros(p)
    for ep in range(n_ep + n_frozen):
        frozen = ep >= n_ep
        if frozen:
            epsl = 0.0
        else:
            frac = ep / max(1, n_ep - n_eval)
            if frac > 1.0:
                frac = 1.0
            epsl = eps0 + (eps1 - eps0) * frac
        upd = (learn == 1) and (not frozen)
        rec_late = (not frozen) and ep >= n_ep - n_eval
        for g in range(G):
            R = 1.0
            Rlag = 1.0
            gret = 0.75
            for i in range(p):
                c[i] = 0.0; crec[i] = 0.0; has_prev[i] = 0.0; pay[i] = 0.0; e_prev[i] = 0.0; excl[i] = 0.0
                gap[i] = 0.0
            _acc_reset(acc, cur, ever, T)
            nrounds = 0
            for t in range(T):
                Robs = _observe(R, Rlag, t, cfg)
                _raw_alloc(use_pl, th, ip, R, Robs, crec, e_prev, excl, t, e)
                _finalize(e, R, integer, tok)
                z0 = np.random.randn() if cfg[3] > 0.0 else 0.0
                se = 0.0
                for i in range(p):
                    if e[i] > 0.0:
                        if fr[g, i] > 0.5:
                            c[i] = 0.0
                        else:
                            s = _state(R, e[i] / (R / p), gret, qv)
                            if has_prev[i] > 0.5 and upd:
                                mq = Q[g, i, s, 0]
                                for aa in range(1, NA):
                                    if Q[g, i, s, aa] > mq:
                                        mq = Q[g, i, s, aa]
                                sp = s_prev[i]; ap = a_prev[i]
                                Q[g, i, sp, ap] += alpha * (r_prev[i] + (gamma ** gap[i]) * mq - Q[g, i, sp, ap])
                            if np.random.rand() < epsl:
                                a = np.random.randint(NA)
                            else:
                                a = 0
                                mq = Q[g, i, s, 0]
                                for aa in range(1, NA):
                                    if Q[g, i, s, aa] > mq:
                                        mq = Q[g, i, s, aa]
                                        a = aa
                            ph = _noisy(acts[a], cfg, z0)
                            acc[2] += ph; acc[3] += 1.0
                            c[i] = _ret(ph, e[i], tok[i], integer)
                            s_prev[i] = s; a_prev[i] = a; r_prev[i] = (e[i] - c[i]) * p
                            has_prev[i] = 1.0; gap[i] = 1.0
                    else:
                        c[i] = 0.0
                        if has_prev[i] > 0.5:
                            gap[i] += 1.0          # zero-reward round inside a semi-Markov transition
                    pay[i] += e[i] - c[i]
                    se += e[i]
                sc = _acc_round(acc, e, c, R, fr[g], cur, ever, t)
                gret = sc / se if se > 1e-12 else 0.0
                _record(c, crec, cfg)
                for i in range(p):
                    if e[i] > 0.0:
                        excl[i] = 0.0
                    else:
                        excl[i] += 1.0
                    e_prev[i] = e[i]
                Rlag = R
                R = _regrow(sc, cfg, _xi(cfg))
                nrounds += 1
                if R < dead:
                    acc[13] = 0.0
                    acc[12] = t + 1
                    break
            # terminal update at depletion or at the normal horizon
            if upd:
                for i in range(p):
                    if has_prev[i] > 0.5 and fr[g, i] < 0.5:
                        sp = s_prev[i]; ap = a_prev[i]
                        Q[g, i, sp, ap] += alpha * (r_prev[i] - Q[g, i, sp, ap])
            for i in range(p):
                has_prev[i] = 0.0
            if rec_late or frozen:
                _acc_finish(row, acc, pay, fr[g], cur, R, T, cfg[0])
                _finish_ever(row, ever, fr[g])
                row[16] = acc[0] / nrounds if nrounds > 0 else 0.0
                if frozen:
                    for q in range(NM):
                        out_fro[q] += row[q]
                    nf += 1.0
                else:
                    for q in range(NM):
                        out_late[q] += row[q]
                    nl += 1.0
    if nl > 0:
        out_late /= nl
    if nf > 0:
        out_fro /= nf
    return out_late, out_fro


def f3_init(G, p, f, seed):
    rng = np.random.default_rng(seed + 7919)
    Q = np.full((G, p, 54, 21), 1.0)
    fr = (rng.random((G, p)) < f).astype(np.float64)
    return Q, fr


def f3_eval(cfg, p, T, f, ip, seed, use_pl=0, th=None, G=64, n_ep=2500, n_eval=300, n_frozen=300,
            alpha=0.1, gamma=0.95, eps0=0.2, eps1=0.01, qv=0):
    if th is None:
        th = np.zeros(NPAR)
    Q, fr = f3_init(G, p, f, seed)
    late, fro = run_f3(Q, fr, p, T, cfg, ip, use_pl, th, seed, n_ep, n_eval, n_frozen, alpha, gamma, eps0, eps1, 1, qv)
    return late, fro, Q, fr


# ------------------------------------------------------------------------------------------------
# F1: evolutionary payoff-maximisers with constant return fractions (Fermi imitation)
# ------------------------------------------------------------------------------------------------
def evolve_f1(cfg, p, T, f, ip, use_pl=0, th=None, N=400, gens=800, K=0.1, mu=0.01, seed=0, tail=200,
              phi0=None, evolve=True):
    """Returns (mean metrics over the last `tail` generations (NM,), final strategy vector, fixed mask).
    phi0: optional initial strategies (e.g. a population pre-adapted to another policy); evolve=False plays
    `tail` generations of regrouping without imitation or mutation (frozen strategies)."""
    if th is None:
        th = np.zeros(NPAR)
    rng = np.random.default_rng(seed)
    N = int(np.ceil(N / p) * p)
    nz = int(round(f * N))
    fixed = np.zeros(N, bool)
    fixed[:nz] = True
    phi = rng.uniform(0, 1, N) if phi0 is None else np.array(phi0, dtype=np.float64).copy()
    phi[fixed] = 0.0
    acc = np.zeros(NM)
    na = 0
    G = gens if evolve else tail
    for gen in range(G):
        perm = rng.permutation(N)
        ph = phi[perm].reshape(-1, p)
        frm = fixed[perm].reshape(-1, p).astype(np.float64)
        pay, met = play_const(ph, frm, cfg, T, ip, use_pl, th, int(rng.integers(1 << 30)))
        if gen >= G - tail:
            acc += met.mean(0)
            na += 1
        if not evolve:
            continue
        payoff = np.empty(N)
        payoff[perm] = pay.ravel()
        model = rng.integers(0, N, N)
        prob = 1.0 / (1.0 + np.exp(-(payoff[model] - payoff) / K))
        adopt = rng.random(N) < prob
        phi = np.where(adopt, phi[model], phi)
        phi = np.clip(phi + rng.normal(0, mu, N), 0, 1)
        phi[fixed] = 0.0
    out = acc / na
    out[5] = phi[~fixed].mean() if (~fixed).any() else 0.0
    return out, phi, fixed


# ------------------------------------------------------------------------------------------------
# benchmarks, grids and the authoritative regime classifier
# ------------------------------------------------------------------------------------------------
def ceiling(r, T, gk=0):
    """Deterministic stationary-surplus reference: T times the maximal sustainable per-round surplus."""
    if gk == 0:
        return T * r / (1.0 + r)
    # Beverton-Holt: stationary stock R*(Phi) = ((1+r)Phi - 1)/(r Phi); per-round surplus R*(1-Phi)
    ph = np.linspace(1.0 / (1.0 + r) + 1e-9, 1.0, 200001)
    Rs = np.minimum(1.0, ((1 + r) * ph - 1) / (r * ph))
    return T * float(np.max(Rs * (1 - ph)))


def inst_grid(fine=True, extra_low_theta=False):
    if fine:
        thetas = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0, 1.01]
        ws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    else:
        thetas = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 1.0, 1.01]
        ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    if extra_low_theta:
        thetas = sorted(set(thetas + [0.2, 0.4, 0.45, 0.55, 0.65, 0.75]))
    out = []
    seen = set()
    for am in (1, 0):
        for th in thetas:
            for wl in ws:
                for wh in ws:
                    if th <= 0.0:
                        key = ("c", wh, am)
                    elif th > 1.0:
                        key = ("c", wl, am)
                    elif wl == wh:
                        key = ("c", wl, am)
                    else:
                        key = (th, wl, wh, am)
                    if key in seen:
                        continue
                    seen.add(key)
                    if key[0] == "c":
                        out.append((0, 0.0, key[1], key[1], 0.0, am))
                    else:
                        out.append((0, th, wl, wh, 0.0, am))
    for kk in [1.0, 2.0, 4.0, 8.0, 16.0, 22.0, 32.0, 64.0]:
        out.append((1, 0.0, 0.0, 0.0, kk, 1))
    return out


def weights(inst, R_scarce=0.6):
    kind, theta, wlo, whi, k, am = inst[:6]
    if kind in (1, 3):
        wf, ws = 1.0, R_scarce ** k
        if kind == 3:
            ws = np.floor(ws * 10 + 0.5) / 10
    else:
        wf = whi if 1.0 >= theta else wlo
        ws = whi if R_scarce >= theta else wlo
    return wf, ws


def regime(inst, p_cut=0.1, e_cut=0.75, d_cut=0.2, R_scarce=0.6):
    """Authoritative classifier (Methods): P if both weights <= 0.1, E if both >= 0.75, T if w_full - w_scarce
    >= 0.2, I if w_scarce - w_full >= 0.2, M otherwise.  Arguments allow the cut-off sensitivity analysis."""
    wf, ws = weights(inst, R_scarce)
    if max(wf, ws) <= p_cut + 1e-12:
        return "P"
    if min(wf, ws) >= e_cut - 1e-12:
        return "E"
    if wf - ws >= d_cut - 1e-12:
        return "T"
    if ws - wf >= d_cut - 1e-12:
        return "I"
    return "M"


EQ = (0, 0.0, 1.0, 1.0, 0.0, 1)
MIX = (0, 0.0, 0.5, 0.5, 0.0, 1)
PROP = (0, 0.0, 0.0, 0.0, 0.0, 1)
HEUR = (1, 0.0, 0.0, 0.0, 22.0, 1)
HEUR_R = (3, 0.0, 0.0, 0.0, 22.0, 1)          # deployed version, w rounded to 0.1
STEP = (0, 0.97, 0.0, 1.0, 0.0, 1)
STEP_PERM = (0, 0.97, 0.0, 1.0, 0.0, 0)
CAPSTEP = (0, 1.0, 0.0, 1.0, 0.0, 1)


def warmup():
    cfg = make_cfg(0.4, 0.05, 0.1, 0.3, 0, 0.1, 1, 0.1, 0)
    ip = make_inst(*STEP)
    th = np.zeros(NPAR)
    fr = np.zeros((2, 4))
    play_const(np.full((2, 4), 0.7), fr, cfg, 5, ip, 0, th, 1)
    play_const(np.full((2, 4), 0.7), fr, cfg, 5, ip, 1, th, 1)
    beh = np.array([0.6, 0.2, 1.0, 0.3, 0.3])
    play_f2(2, 4, 5, cfg, 0.1, beh, ip, 0, th, 1)
    play_f2(2, 4, 5, cfg, 0.1, beh, ip, 1, th, 1)
    trace_f2(2, 4, 5, cfg, 0.1, beh, ip, 1, th, 1)
    Q, frq = f3_init(2, 4, 0.1, 1)
    run_f3(Q, frq, 4, 5, cfg, ip, 0, th, 1, 3, 1, 1, 0.1, 0.95, 0.2, 0.01, 1, 0)
    run_f3(Q, frq, 4, 5, cfg, ip, 1, th, 1, 3, 1, 1, 0.1, 0.95, 0.2, 0.01, 1, 2)
    cfgi = make_cfg(0.4, 0.05, 0.0, integer=1)
    play_const(np.full((2, 4), 0.7), fr, cfgi, 5, ip, 0, th, 1)
