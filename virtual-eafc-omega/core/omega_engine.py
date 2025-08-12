# core/omega_engine.py
# UltraGOD OMEGA v4 — Single-Market Robust Analyzer (Android/Windows-safe)
# numpy-only; no SciPy. Supports 1X2 or OU (+ optional second OU line).
# Adds: CVaR alpha, median EV, safety margin, Kelly on p_alpha.

import math, random, json, os
import numpy as np
from datetime import datetime

# ============ Utils ============
def clamp(x, lo, hi): return lo if x<lo else (hi if x>hi else x)
def implied_probs(odds): return np.array([1.0/o for o in odds], dtype=float)
def normalize(p): 
    s = float(np.sum(p)); 
    return p/s if s>0 else np.ones_like(p)/len(p)
def entropy(p): 
    q=np.clip(p,1e-12,1.0); 
    return float(-np.sum(q*np.log(q)))
def ev(o,p): return o*p - 1.0

def kelly_fraction(o,p):
    den = max(1e-12, o-1.0)
    f = (o*p - 1.0)/den
    return max(0.0, f)

def qtile(x, q):
    # x: list/np.array, q in [0,1]
    a = np.sort(np.array(x))
    if len(a)==0: return float('nan')
    idx = clamp(int(q*(len(a)-1)), 0, len(a)-1)
    return float(a[idx])

def cvar(x, alpha):
    # CVaR_alpha: میانگین بدترین α%
    a = np.sort(np.array(x))
    if len(a)==0: return float('nan')
    k = max(1, int(math.ceil(alpha*len(a))))
    return float(np.mean(a[:k]))

# ============ Margin Models ============
def devig_proportional(odds):
    p = normalize(implied_probs(odds))
    return p, {"model":"proportional","alpha":1.0}

def devig_power(odds, alpha):
    z = np.array([(1.0/o)**alpha for o in odds], dtype=float)
    return normalize(z)

def fit_alpha(odds):
    H0 = math.log(len(odds))
    best=(1e9,1.0)
    for a in np.linspace(0.70,1.40,29):
        p=devig_power(odds,a)
        sc=abs(entropy(p)-H0)
        if sc<best[0]: best=(sc,a)
    return best[1]

def devig_shin(odds, tol=1e-9, it=200):
    z = implied_probs(odds)
    s_lo, s_hi = 0.0, 0.25
    def psum(s):
        den = 2.0*(1.0-s)
        if den<=1e-12: return 10.0
        p = (np.sqrt(s*s + 4.0*(1.0-s)*z) - s)/den
        return float(np.sum(p))
    while psum(s_hi)>1.0 and s_hi<0.95:
        s_hi=(s_hi+1.0)*0.5
    s=0.0
    for _ in range(it):
        m=0.5*(s_lo+s_hi); v=psum(m)
        if abs(v-1.0)<tol: s=m; break
        if v>1.0: s_lo=m
        else: s_hi=m
    den=2.0*(1.0-s)
    p=(np.sqrt(s*s + 4.0*(1.0-s)*z) - s)/den
    p=np.maximum(p,1e-12); p/=np.sum(p)
    return p, {"model":"shin","s":float(s)}

def margin_candidates(odds):
    c=[]
    p,m=devig_proportional(odds); m["alpha"]=1.0; c.append((p,m))
    a=fit_alpha(odds); p=devig_power(odds,a); c.append((p,{"model":"power","alpha":float(a)}))
    p,m=devig_shin(odds); c.append((p,m))
    return c

# ============ Adaptive Tick + Shading ============
def _dec_places(x):
    s=f"{x:.10f}".rstrip('0').rstrip('.')
    return len(s.split('.')[1]) if '.' in s else 0

def _tick_list(o):
    dp=_dec_places(o)
    base = [0.985, 0.9925, 1.0, 1.0075, 1.015] if dp>=3 else \
           [0.985, 0.995, 1.0, 1.005, 1.015] if dp==2 else \
           [0.98, 0.99, 1.0, 1.01, 1.02]
    if o>=3.0: base = sorted(set(base+[1.03]))
    # افزودن همسایه‌های گردشده با قدم تقریبی 0.01 یا 0.02
    steps = [0.01,0.02] if dp>=2 else [0.05,0.1]
    ext=set([o])
    for st in steps:
        for k in (-2,-1,0,1,2):
            r = round((o+k*st)/st)*st
            ext.add(float(f"{max(1.01,r):.3f}"))
    for m in base: ext.add(float(f"{max(1.01,o*m):.3f}"))
    return sorted(ext)

def tick_scenarios(odds):
    grids=[_tick_list(o) for o in odds]
    idxs=[list(range(len(g))) for g in grids]
    combos=[]
    def bt(k,acc):
        if k==len(grids): combos.append(tuple(acc)); return
        for i in idxs[k]:
            acc.append(grids[k][i]); bt(k+1,acc); acc.pop()
    bt(0,[])
    # sort by closeness to original to keep top-K
    def closeness(t): return sum(abs(t[i]-odds[i]) for i in range(len(odds)))
    combos.sort(key=closeness)
    return combos[:240]  # cap to keep performance on Android

# ============ Core: Robust Evaluation ============
class OmegaSingleMarket:
    """
    kind: "1X2" ⇒ odds=(H,D,A)
          "OU"  ⇒ odds=(Over,Under), line=float
    aux_ou (optional): {"line2": float, "odds2": (O2,U2)}
    params:
      alpha_cvar: مثلاً 0.1 یعنی بدترین 10% سناریوها را میانگین می‌گیریم
      ri_threshold: حداقل RI برای پذیرش
      safety_margin: حداقل EV_low که می‌خواهی (مثلاً +0.01)
      kelly_cap: درصد Kelly (مثلاً 0.25)
      stake_cap_abs: سقف درصد بانک (مثلاً 0.05)
    """
    def __init__(self, seed=123, alpha_cvar=0.10, ri_threshold=0.55, safety_margin=0.0,
                 kelly_cap=0.25, stake_cap_abs=0.05):
        random.seed(seed); np.random.seed(seed)
        self.alpha = float(alpha_cvar)
        self.ri_thr = float(ri_threshold)
        self.s_margin = float(safety_margin)
        self.kelly_cap = float(kelly_cap)
        self.stake_cap_abs = float(stake_cap_abs)

    # scenario prob set
    def _Pset(self, odds):
        Ps=[]
        for od in tick_scenarios(odds):
            for p,_ in margin_candidates(od):
                Ps.append(p)
        return Ps

    def _rows_from_Pset(self, label_fn, odds, Pset):
        n=len(odds); rows=[]
        EV_mat = np.zeros((len(Pset), n))
        for s, p in enumerate(Pset):
            for i in range(n):
                EV_mat[s,i] = ev(odds[i], p[i])
        for i in range(n):
            evs = EV_mat[:,i]
            pvals = [p[i] for p in Pset]
            rows.append({
                "name": label_fn(i),
                "odds": float(odds[i]),
                "p_low": float(np.min(pvals)),
                "p_high": float(np.max(pvals)),
                "EV_low": float(np.min(evs)),
                "EV_high": float(np.max(evs)),
                "EV_med": float(qtile(evs, 0.5)),
                "EV_cvar": float(cvar(evs, self.alpha)),  # ≤ median
                "RI": float(np.mean(evs>0.0))
            })
        return rows

    def _apply_ou_consistency(self, rows, line1, rows2, line2):
        # enforce monotonic hint: if line2>line1 ⇒ P_over(line2) ≤ P_over(line1)
        def find(rows, name):
            for r in rows:
                if r["name"]==name: return r
            return None
        o1 = find(rows,  f"OU {line1} Over");  u1 = find(rows,  f"OU {line1} Under")
        o2 = find(rows2, f"OU {line2} Over");  u2 = find(rows2, f"OU {line2} Under")
        if not (o1 and u1 and o2 and u2): return rows + rows2
        if line2>line1:
            o2["p_high"]=min(o2["p_high"], o1["p_high"])
            o2["p_low"] =min(o2["p_low"],  o1["p_low"])
            u2["p_low"] =max(u2["p_low"], 1.0-o2["p_high"])
            u2["p_high"]=max(u2["p_high"],1.0-o2["p_low"])
        elif line2<line1:
            o1["p_high"]=min(o1["p_high"], o2["p_high"])
            o1["p_low"] =min(o1["p_low"],  o2["p_low"])
            u1["p_low"] =max(u1["p_low"], 1.0-o1["p_high"])
            u1["p_high"]=max(u1["p_high"],1.0-o1["p_low"])
        return rows + rows2

    def analyze(self, kind, odds, line=None, aux_ou=None, bank=None):
        assert kind in ("1X2","OU")
        n=len(odds); assert (kind=="1X2" and n==3) or (kind=="OU" and n==2)

        names = ["Home","Draw","Away"] if kind=="1X2" else ["Over","Under"]
        label = (lambda i: f"1X2:{names[i]}") if kind=="1X2" else (lambda i: f"OU {line} {names[i]}")

        Pset = self._Pset(odds)
        rows = self._rows_from_Pset(label, odds, Pset)

        # optional second OU
        if kind=="OU" and aux_ou and "line2" in aux_ou and "odds2" in aux_ou:
            line2=float(aux_ou["line2"]); odds2=aux_ou["odds2"]
            Pset2=self._Pset(odds2)
            label2=lambda i: f"OU {line2} {names[i]}"
            rows2=self._rows_from_Pset(label2, odds2, Pset2)
            rows=self._apply_ou_consistency(rows, float(line), rows2, line2)

        # choose final pick with three guards: EV_low>0, EV_cvar>0, RI≥thr, and EV_low≥s_margin
        cands=[r for r in rows if r["EV_low"]>self.s_margin and r["EV_cvar"]>0.0 and r["RI"]>=self.ri_thr]
        cands.sort(key=lambda r: (r["EV_low"], r["EV_cvar"], r["RI"]), reverse=True)
        final = cands[0] if cands else None

        stake=None
        if final and bank not in (None,"",0,0.0):
            try:
                bank=float(bank)
                # conservative prob = quantile alpha of scenario p_i
                # تقریبی: از EV_cvar و odds نتیجه‌گیری مستقیم سخت است؛ بهتر: p_alpha ≈ max(p_low, (1+EV_cvar)/odds)
                p_alpha = max(final["p_low"], (1.0 + final["EV_cvar"]) / final["odds"])
                f_k = kelly_fraction(final["odds"], p_alpha)
                f_bet = min(self.kelly_cap * f_k, self.stake_cap_abs)
                stake={"fraction":float(f_bet), "amount":float(bank*f_bet), "p_alpha":float(p_alpha)}
            except Exception:
                pass

        return {"rows":rows, "final":final, "stake":stake}

# ============ Logging ============
def default_log_path():
    android_path="/storage/emulated/0/Download/ultragod_omega_log.json"
    if os.path.isdir("/storage/emulated/0/Download"): return android_path
    return os.path.join(os.getcwd(),"ultragod_omega_log.json")

def append_log(entry, path=None, keep_last=10000):
    try:
        path = path or default_log_path()
        data=[]
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                try: data=json.load(f); 
                except: data=[]
        entry["_ts"]=datetime.utcnow().isoformat()+"Z"
        data.append(entry)
        if len(data)>keep_last: data=data[-keep_last:]
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        return True, path
    except Exception as e:
        return False, str(e)
