# -*- coding: utf-8 -*-
"""
Nested Logit (4 alts) with {1=轻伤, 2=无伤} in one nest, and {3=住院}, {4=死亡} alone.
原始 grav: 0=死亡, 1=轻伤, 2=无伤, 3=住院
内部映射: alt = {1=轻伤, 2=无伤, 3=住院, 4=死亡}
"""

import os
import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme import expressions as ex
from biogeme.expressions import Beta, Variable, Derive
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit

# -------------------------------------------------
FILE_PATH = "cleaned_data_recoded_final.csv"
OUT_DIR   = "nested_logit_4alts_with_group12"
SAMPLE_N  = None

NUM_COLS  = ["age", "nbv"]
CAT_COLS  = ["sexe", "lum", "secu", "obs", "prof", "plan", "atm", "obsm", "surf"]
# -------------------------------------------------


def make_var(database, name: str, expr):
    if hasattr(database, "define_variable"):
        database.define_variable(name, expr)
    elif hasattr(database, "DefineVariable"):
        database.DefineVariable(name, expr)
    else:
        from biogeme.expressions import DefineVariable
        DefineVariable(name, expr, database)


def get_params_df(results):
    try:
        from biogeme.results_processing import get_pandas_estimated_parameters
        return get_pandas_estimated_parameters(estimation_results=results)
    except Exception:
        pass
    for attr in ("get_estimated_parameters", "getEstimatedParameters"):
        if hasattr(results, attr):
            try:
                return getattr(results, attr)()
            except Exception:
                pass
    raise RuntimeError("无法从当前 Biogeme 版本提取参数表。")


def get_general_df(results):
    for attr in ("get_general_statistics", "getGeneralStatistics"):
        if hasattr(results, attr):
            try:
                gen = getattr(results, attr)()
                if isinstance(gen, pd.DataFrame):
                    return gen
                else:
                    return pd.DataFrame(list(gen.items()), columns=["stat", "value"])
            except Exception:
                pass
    return pd.DataFrame({"stat": ["info"], "value": ["unavailable"]})


# ================== 读取 & 清洗 ==================
df = pd.read_csv(FILE_PATH)

df = df[~pd.isna(df["grav"])].copy()
df["grav"] = pd.to_numeric(df["grav"], errors="coerce")
df = df[df["grav"].isin([0, 1, 2, 3])].copy()

map_to_alt = {1: 1, 2: 2, 3: 3, 0: 4}
df["alt"] = df["grav"].map(map_to_alt).astype(int)

use_cols = ["grav", "alt"] + [c for c in NUM_COLS if c in df.columns] + [c for c in CAT_COLS if c in df.columns]
df = df[use_cols].dropna(axis=0).reset_index(drop=True)

if SAMPLE_N is not None and len(df) > SAMPLE_N:
    df = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)

database = db.Database("nested_4alts", df)

# ================== Biogeme 变量 ==================
CHOICE = Variable("alt")

# 可用性：用 Numeric(1) 创建“表达式常数”
Av1 = ex.Numeric(1)
Av2 = ex.Numeric(1)
Av3 = ex.Numeric(1)
Av4 = ex.Numeric(1)

# ---- 连续变量：标准化并注册 ----
std_info = {}
for c in NUM_COLS:
    if c in df.columns:
        m, s = float(df[c].mean()), float(df[c].std(ddof=0))
        if s > 0 and np.isfinite(s):
            make_var(database, f"{c}_std", (Variable(c) - m) / s)
            std_info[c] = (m, s)

# ---- 分类变量：K-1 哑变量，基类=众数 ----
cat_base = {}
dummy_names = []
for col in CAT_COLS:
    if col not in df.columns:
        continue
    vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    if vals.empty:
        continue
    base = int(vals.mode().iloc[0])
    cat_base[col] = base
    levels = sorted(int(v) for v in pd.unique(vals))
    for v in levels:
        if v == base:
            continue
        name = f"{col}=={v}"
        make_var(database, name, (Variable(col) == v))
        dummy_names.append(name)

# ================== 参数（系数） ==================
ASC1 = Beta("ASC_1_minor", 0, None, None, 0)
ASC2 = Beta("ASC_2_none",  0, None, None, 0)
ASC3 = Beta("ASC_3_hosp",  0, None, None, 0)
ASC4 = Beta("ASC_4_fatal(baseline)", 0, None, None, 1)  # 固定0

MU_12 = Beta("MU_12(light_nest)", 0.7, 1e-3, 5.0, 0)

beta_cont_shared_12 = {c: Beta(f"B_{c}_shared_in_{{1,2}}", 0, None, None, 0) for c in std_info}
beta_cont_alt3      = {c: Beta(f"B_{c}_alt3",               0, None, None, 0) for c in std_info}

beta_dummy_shared_12 = {d: Beta(f"B_{d}_shared_in_{{1,2}}", 0, None, None, 0) for d in dummy_names}
beta_dummy_alt3      = {d: Beta(f"B_{d}_alt3",               0, None, None, 0) for d in dummy_names}

# ================== 构建效用 ==================
V1, V2, V3, V4 = ASC1, ASC2, ASC3, ASC4

for c in std_info:
    xc = Variable(f"{c}_std")
    V1 += beta_cont_shared_12[c] * xc
    V2 += beta_cont_shared_12[c] * xc
    V3 += beta_cont_alt3[c]      * xc

for d in dummy_names:
    xd = Variable(d)
    V1 += beta_dummy_shared_12[d] * xd
    V2 += beta_dummy_shared_12[d] * xd
    V3 += beta_dummy_alt3[d]      * xd

V = {1: V1, 2: V2, 3: V3, 4: V4}
av = {1: Av1, 2: Av2, 3: Av3, 4: Av4}

nest12 = OneNestForNestedLogit(MU_12, [1, 2])
nest3  = OneNestForNestedLogit(1.0,   [3])
nest4  = OneNestForNestedLogit(1.0,   [4])
nests = [(MU_12, [1, 2]), (1.0, [3]), (1.0, [4])]


# ================== 拟合 ==================
loglike = models.lognested(V, av, nests, CHOICE)
bg = bio.BIOGEME(database, loglike)
bg.model_name = "nested_logit_4alts_with_group12"
results = bg.estimate()

# ================== 导出参数 / 总体统计 ==================
os.makedirs(OUT_DIR, exist_ok=True)

est = get_params_df(results)
est.to_csv(os.path.join(OUT_DIR, "params.csv"), index=False)

gen = get_general_df(results)
gen.to_csv(os.path.join(OUT_DIR, "general_stats.csv"), index=False)

# ================== 概率与合并概率 ==================
P1  = models.nested(V, av, nests, 1)
P2  = models.nested(V, av, nests, 2)
P3  = models.nested(V, av, nests, 3)
P4  = models.nested(V, av, nests, 4)
P12 = P1 + P2

expr = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P12": P12}

# ---- 连续变量解析边际效应 ∂P/∂x
for c in std_info:
    xc = Variable(f"{c}_std")
    for label, Prob in [("P1", P1), ("P2", P2), ("P3", P3), ("P4", P4), ("P12", P12)]:
        try:
            expr[f"d{label}_d{c}"] = Derive(Prob, xc)
        except Exception as e:
            print(f"[WARN] Derivative failed for {label} wrt {c}: {e}")

# ---- Dummy 的有限差分 APE：覆盖时也用 Numeric(0/1)
def simulate_with_overrides(overrides: dict):
    forced = {**expr}
    forced.update(overrides)  # value 必须是 Expression
    sim_bio = bio.BIOGEME(database, forced)
    betas = results.get_beta_values() if hasattr(results, "get_beta_values") else results.getBetaValues()
    return sim_bio.simulate(betas)

ape_rows = []
for dname in dummy_names:
    try:
        sim1 = simulate_with_overrides({dname: ex.Numeric(1)})
        sim0 = simulate_with_overrides({dname: ex.Numeric(0)})
        for label in ["P12", "P3", "P4"]:
            diff = (sim1[label] - sim0[label]).mean()
            base_val = cat_base.get(dname.split("==")[0], None)
            ape_rows.append({
                "quantity": f"Δ{label}_when_{dname}=1_vs_base({base_val})",
                "APE": float(diff)
            })
    except Exception as e:
        print(f"[WARN] APE (finite diff) failed for {dname}: {e}")

biosim = bio.BIOGEME(database, expr)
betas = results.get_beta_values() if hasattr(results, "get_beta_values") else results.getBetaValues()
sim = biosim.simulate(betas)
ape_deriv = sim.mean().to_frame(name="APE").reset_index().rename(columns={"index": "quantity"})

ape_fd = pd.DataFrame(ape_rows) if ape_rows else pd.DataFrame(columns=["quantity", "APE"])
ape_all = pd.concat([ape_deriv, ape_fd], ignore_index=True)
ape_all.to_csv(os.path.join(OUT_DIR, "marginal_effects_APE.csv"), index=False)

print("✅ 完成，结果保存在：", os.path.abspath(OUT_DIR))
print("\n[分类变量基类（众数）记录]")
for k, v in cat_base.items():
    print(f"  {k}: base={v}")
print("\n[参数预览]\n", est.head())
print("\n[总体统计预览]\n", gen.head() if isinstance(gen, pd.DataFrame) else gen)
print("\n[边际效应（含 P12 合并后的概率 & dummy 的有限差分）预览]\n", ape_all.head())
