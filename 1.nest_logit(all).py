
"""
Nested Logit for 4 categories, grav coded as:
0 = 死亡, 1 = 轻伤, 2 = 无伤, 3 = 住院

内部重编码为 1..4（便于建模）：
1 = 无伤(原2), 2 = 死亡(原0), 3 = 住院(原3), 4 = 轻伤(原1)

候选嵌套：
S1: {无伤, 轻伤} / {住院, 死亡}
S2: {无伤, 轻伤, 住院} / {死亡}
S3: {无伤} / {轻伤, 住院} / {死亡}
S4: {无伤} / {轻伤} / {住院, 死亡}
"""

import os
import pandas as pd
import numpy as np

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable


FILE_PATH = "cleaned_data_final.csv"   
OUT_DIR   = "nested_logit_4alts_batch"
SAMPLE_N  = None        
DEFAULT_AV = True   
# -------------------------------------


def load_and_prepare():
    df = pd.read_csv(FILE_PATH)
    df = df[~pd.isna(df["grav"])].copy()

    # 原始 grav: 0=死亡, 1=轻伤, 2=无伤, 3=住院
    df["grav"] = pd.to_numeric(df["grav"], errors="coerce").astype(int)
    df = df[df["grav"].isin([0, 1, 2, 3])].copy()

    # 重编码到 1..4（Biogeme 更稳）：
    # 1=无伤(原2), 2=死亡(原0), 3=住院(原3), 4=轻伤(原1)
    recode_map = {2: 1, 0: 2, 3: 3, 1: 4}
    df["grav_int"] = df["grav"].map(recode_map).astype(int)

    # 抽样
    if SAMPLE_N is not None and len(df) > SAMPLE_N:
        df = df.sample(n=SAMPLE_N, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

 
    if DEFAULT_AV:
        df["Av1"] = 1  # 无伤
        df["Av2"] = 1  # 死亡
        df["Av3"] = 1  # 住院
        df["Av4"] = 1  # 轻伤

    return df


def fit_nested(df, model_name, nests_spec):
    os.makedirs(OUT_DIR, exist_ok=True)
    database = db.Database(model_name, df)

    CHOICE = Variable("grav_int")
    Av1, Av2, Av3, Av4 = Variable("Av1"), Variable("Av2"), Variable("Av3"), Variable("Av4")

    # 截距（ASC_2=0 基准：死亡）
    ASC_1 = Beta("ASC_NoInjury",    0, None, None, 0)  # 1=无伤
    ASC_2 = Beta("ASC_Fatal",       0, None, None, 1)  # 2=死亡（固定为0）
    ASC_3 = Beta("ASC_Hospital",    0, None, None, 0)  # 3=住院
    ASC_4 = Beta("ASC_MinorInjury", 0, None, None, 0)  # 4=轻伤

    V  = {1: ASC_1, 2: ASC_2, 3: ASC_3, 4: ASC_4}
    av = {1: Av1,   2: Av2,   3: Av3,   4: Av4}

    loglike = models.lognested(V, av, nests_spec, CHOICE)
    bg = bio.BIOGEME(database, loglike)
    bg.model_name = model_name
    results = bg.estimate()

    # 3.3.x 优先
    try:
        est = results.get_pandas_estimated_parameters()
    except Exception:
        est = results.get_estimated_parameters() if hasattr(results, "get_estimated_parameters") else results.getEstimatedParameters()

    try:
        gen = results.get_general_statistics()
    except Exception:
        gen = results.getGeneralStatistics()

    est_path = os.path.join(OUT_DIR, f"{model_name}_params.csv")
    gen_path = os.path.join(OUT_DIR, f"{model_name}_general_stats.csv")
    est.to_csv(est_path, index=False)

    if isinstance(gen, pd.DataFrame):
        gen.to_csv(gen_path, index=False)
        aic = gen.loc[gen["stat"].str.contains("Akaike", case=False, na=False), "value"]
        bic = gen.loc[gen["stat"].str.contains("Bayesian", case=False, na=False), "value"]
        aic = float(aic.values[0]) if len(aic) else np.nan
        bic = float(bic.values[0]) if len(bic) else np.nan
    else:
        pd.DataFrame(list(gen.items()), columns=["stat", "value"]).to_csv(gen_path, index=False)
        aic = float(gen.get("Akaike Information Criterion", np.nan))
        bic = float(gen.get("Bayesian Information Criterion", np.nan))

    mu_rows = est[est["Name"].str.contains("MU_", case=False, na=False)] if "Name" in est.columns else est

    return {
        "model": model_name,
        "mu_table": mu_rows,
        "AIC": aic,
        "BIC": bic,
        "params_csv": est_path,
        "stats_csv": gen_path,
    }


def main():
    df = load_and_prepare()
    print("[INFO] grav 原始值分布（0=死亡,1=轻伤,2=无伤,3=住院）：")
    print(df["grav"].value_counts().sort_index())
    print("\n[INFO] grav_int(1=无伤,2=死亡,3=住院,4=轻伤) 分布：")
    print(df["grav_int"].value_counts().sort_index())


    # S1：{无伤, 轻伤} / {住院, 死亡}
    MU_Light  = Beta("MU_Light",  0.8, 1e-3, 0.999, 0)
    MU_Severe = Beta("MU_Severe", 0.8, 1e-3, 0.999, 0)
    nests_S1 = [(MU_Light, [1, 4]), (MU_Severe, [3, 2])]

    # S2：{无伤, 轻伤, 住院} / {死亡}
    MU_NonFatal = Beta("MU_NonFatal", 0.8, 1e-3, 0.999, 0)
    nests_S2 = [(MU_NonFatal, [1, 4, 3]), (1.0, [2])]

    # S3：{无伤} / {轻伤, 住院} / {死亡}
    MU_Mid = Beta("MU_Mid", 0.8, 1e-3, 0.999, 0)
    nests_S3 = [(1.0, [1]), (MU_Mid, [4, 3]), (1.0, [2])]

    # S4：{无伤} / {轻伤} / {住院, 死亡}
    MU_Severe2 = Beta("MU_Severe", 0.8, 1e-3, 0.999, 0)
    nests_S4 = [(1.0, [1]), (1.0, [4]), (MU_Severe2, [3, 2])]

    results = []
    results.append(fit_nested(df, "S1_nested_light_vs_severe", nests_S1))
    results.append(fit_nested(df, "S2_nested_nonfatal_vs_fatal", nests_S2))
    results.append(fit_nested(df, "S3_nested_mid_cluster",      nests_S3))
    results.append(fit_nested(df, "S4_nested_severe_pair",      nests_S4))

    print("\n================= SUMMARY (μ & AIC/BIC) =================")
    for r in results:
        print(f"\nModel: {r['model']}")
        print(f"AIC={r['AIC']:.3f}, BIC={r['BIC']:.3f}")
        if isinstance(r["mu_table"], pd.DataFrame) and not r["mu_table"].empty:
            cols = [c for c in ["Name", "Value", "Robust t-stat.", "Robust p-value"] if c in r["mu_table"].columns]
            print(r["mu_table"][cols])
        else:
            print("(no MU rows found — check params CSV)")
        print(f"Params CSV: {r['params_csv']}")
        print(f"Stats  CSV: {r['stats_csv']}")
    print("==========================================================\n")

    print(f"全部结果已保存到：{os.path.abspath(OUT_DIR)}")
    print("解读：μ 在 (0,1) 且显著 → 该 nest 内部确有相关性；对比 AIC/BIC 选更优结构。")


if __name__ == "__main__":
    main()
