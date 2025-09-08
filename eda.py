# eda.py
# -*- coding: utf-8 -*-

import os
import math
import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# === Config ===
DATA_DIR = "Datos"        # carpeta donde están train_data.txt y test_data.txt
OUTPUT_DIR = "outputs"    # carpeta donde se guardan los CSVs
SEP = "\t"                # separador de los .txt
TOP_K = 10                # top-k valores más frecuentes por columna
NUM_BINS = 20             # bins para comparar distribuciones numéricas

# === Utils ===

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def mem_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024**2))

def describe_numeric(s: pd.Series) -> dict:
    s_num = pd.to_numeric(s, errors="coerce")
    desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
    out = {
        "count_non_null": int(desc["count"]) if pd.notnull(desc["count"]) else 0,
        "mean": float(desc["mean"]) if pd.notnull(desc["mean"]) else np.nan,
        "std": float(desc["std"]) if pd.notnull(desc["std"]) else np.nan,
        "min": float(desc["min"]) if pd.notnull(desc["min"]) else np.nan,
        "q25": float(desc["25%"]) if pd.notnull(desc["25%"]) else np.nan,
        "median": float(desc["50%"]) if pd.notnull(desc["50%"]) else np.nan,
        "q75": float(desc["75%"]) if pd.notnull(desc["75%"]) else np.nan,
        "max": float(desc["max"]) if pd.notnull(desc["max"]) else np.nan,
    }
    return out

def try_parse_datetime(series: pd.Series) -> pd.Series:
    """
    Intenta parsear a datetime si el nombre sugiere fecha.
    Evita el FutureWarning de errors='ignore' usando try/except explícito.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    name = (series.name or "").lower()
    if ("time" in name) or ("date" in name) or (name in {"ts", "offline_timestamp"}):
        try:
            return pd.to_datetime(series, utc=False)
        except Exception:
            return series
    return series

def topk_freq(s: pd.Series, k: int = 10) -> pd.DataFrame:
    """
    Top-k de frecuencias robusto a nombres de índice tras reset_index.
    """
    vc = s.value_counts(dropna=False).head(k)
    total = len(s)

    df = (
        vc.rename("count")
        .to_frame()
        .assign(pct=lambda d: d["count"] / total if total else 0.0)
        .reset_index()
    )

    # La primera columna puede llamarse 'index' o el nombre del Series original.
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "value"})

    # Representar NaN como string para exportar a CSV sin perder filas
    df["value"] = df["value"].astype(object).where(pd.notnull(df["value"]), "NaN")
    return df[["value", "count", "pct"]]

def jensen_shannon_divergence(p, q) -> float:
    """JS divergence entre dos distribuciones discretas (arrays que suman 1)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log2(a / b))
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(js)

def categorical_js(train_s: pd.Series, test_s: pd.Series) -> float:
    """JS para categóricas, con unión de categorías."""
    train_vc = train_s.value_counts(dropna=False)
    test_vc  = test_s.value_counts(dropna=False)
    cats = list(set(train_vc.index).union(set(test_vc.index)))
    train_p = np.array([train_vc.get(c, 0) for c in cats], dtype=float)
    test_p  = np.array([test_vc.get(c, 0) for c in cats], dtype=float)
    if train_p.sum() == 0 or test_p.sum() == 0:
        return np.nan
    return jensen_shannon_divergence(train_p, test_p)

def numeric_js(train_s: pd.Series, test_s: pd.Series, bins: int = 20) -> float:
    """JS para numéricas (binning por cuantiles de train)."""
    x = pd.to_numeric(train_s, errors="coerce").dropna()
    y = pd.to_numeric(test_s, errors="coerce").dropna()
    if len(x) == 0 or len(y) == 0:
        return np.nan

    # bins por cuantiles de train
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(x, qs))

    # Si no hay variación suficiente o edges degenerados, construir por rango combinado
    if len(edges) < 3 or np.allclose(edges, edges[0]):
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        if np.isclose(lo, hi):
            # distribución degenerada en ambos -> sin divergencia
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)

    # Asegurar edges estrictamente crecientes
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    train_hist, _ = np.histogram(x, bins=edges)
    test_hist, _  = np.histogram(y, bins=edges)
    return jensen_shannon_divergence(train_hist, test_hist)

def ood_rate_categorical(train_s: pd.Series, test_s: pd.Series) -> float:
    """Porcentaje de valores de test que no aparecen en train (fuera de diccionario)."""
    train_set = set(train_s.dropna().unique().tolist())
    test_vals = test_s.dropna()
    if len(test_vals) == 0:
        return 0.0
    ood = (~test_vals.isin(train_set)).mean()
    return float(ood)

# === Carga de datos ===

def load_competition_datasets(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train_data.txt")
    test_path  = os.path.join(data_dir, "test_data.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontró: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró: {test_path}")

    print(f">> Cargando train: {train_path}")
    train_df = pd.read_csv(train_path, sep=SEP, low_memory=False)

    print(f">> Cargando test:  {test_path}")
    test_df  = pd.read_csv(test_path,  sep=SEP, low_memory=False)

    # Intento de parseo de fechas típicas
    for col in train_df.columns:
        train_df[col] = try_parse_datetime(train_df[col])
    for col in test_df.columns:
        test_df[col] = try_parse_datetime(test_df[col])

    return train_df, test_df

# === Resumen por dataset y por columna ===

def dataframe_overview(df: pd.DataFrame, name: str) -> dict:
    info = {
        "dataset": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": round(mem_usage_mb(df), 2),
    }
    return info

def summarize_columns(df: pd.DataFrame, name: str, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    top_rows: list[dict] = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_null = int(s.isna().sum())
        pct_null = float(n_null / n) if n > 0 else np.nan
        n_unique = int(s.nunique(dropna=True))

        row = {
            "dataset": name,
            "column": col,
            "dtype": dtype,
            "rows": n,
            "n_null": n_null,
            "pct_null": pct_null,
            "n_unique": n_unique,
        }

        # Métricas adicionales según tipo
        if pd.api.types.is_numeric_dtype(s):
            row.update(describe_numeric(s))
        elif pd.api.types.is_datetime64_any_dtype(s):
            s_dt = pd.to_datetime(s, errors="coerce")
            row["min_datetime"] = s_dt.min()
            row["max_datetime"] = s_dt.max()

        # Top-k (para todos)
        top = topk_freq(s, k=top_k)
        for _, r in top.iterrows():
            top_rows.append({
                "dataset": name,
                "column": col,
                "value": r["value"],
                "count": int(r["count"]),
                "pct": float(r["pct"]),
            })

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    topk_df = pd.DataFrame(top_rows)
    return summary_df, topk_df

# === Comparativa train vs test por columna ===

def compare_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, num_bins: int = NUM_BINS) -> pd.DataFrame:
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    rows = []

    for col in common_cols:
        s_tr = train_df[col]
        s_te = test_df[col]
        dtype_tr = str(s_tr.dtype)
        dtype_te = str(s_te.dtype)

        comp = {
            "column": col,
            "dtype_train": dtype_tr,
            "dtype_test": dtype_te,
            "n_train": len(s_tr),
            "n_test": len(s_te),
            "n_unique_train": int(s_tr.nunique(dropna=True)),
            "n_unique_test": int(s_te.nunique(dropna=True)),
        }

        # Elegimos rama por tipo "principal"
        if pd.api.types.is_numeric_dtype(s_tr) and pd.api.types.is_numeric_dtype(s_te):
            comp["js_divergence"] = numeric_js(s_tr, s_te, bins=num_bins)
            comp["ood_rate"] = np.nan  # No aplica a numérico
        else:
            # Tratar todo lo no-numérico como categórico para comparación
            comp["js_divergence"] = categorical_js(s_tr.astype("object"), s_te.astype("object"))
            comp["ood_rate"] = ood_rate_categorical(s_tr.astype("object"), s_te.astype("object"))

        rows.append(comp)

    return pd.DataFrame(rows)

# === Extra específico: distribución de reason_end en train ===

def reason_end_distribution(train_df: pd.DataFrame) -> pd.DataFrame:
    if "reason_end" not in train_df.columns:
        return pd.DataFrame(columns=["reason_end", "count", "pct"])
    s = train_df["reason_end"]
    vc = s.value_counts(dropna=False)
    df = (
        vc.rename("count")
        .to_frame()
        .assign(pct=lambda d: d["count"] / len(s) if len(s) else 0.0)
        .reset_index()
    )
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "reason_end"})
    df["reason_end"] = df["reason_end"].astype(object).where(pd.notnull(df["reason_end"]), "NaN")
    return df[["reason_end", "count", "pct"]]

# === Main ===

def main() -> None:
    print("=== EDA competencia: train vs test ===")
    ensure_dir(OUTPUT_DIR)

    train_df, test_df = load_competition_datasets(DATA_DIR)

    # Overviews
    train_over = dataframe_overview(train_df, "train")
    test_over  = dataframe_overview(test_df,  "test")
    print("\n--- Dataset overview ---")
    print(train_over)
    print(test_over)

    # Summaries
    print("\n--- Resumen por columnas (train) ---")
    summary_train, top_train = summarize_columns(train_df, "train", TOP_K)
    print(summary_train.head(12))

    print("\n--- Resumen por columnas (test) ---")
    summary_test, top_test = summarize_columns(test_df, "test", TOP_K)
    print(summary_test.head(12))

    # reason_end (solo train)
    re_dist = reason_end_distribution(train_df)
    if not re_dist.empty:
        print("\n--- Distribución reason_end (train) ---")
        print(re_dist)

    # Comparativa
    print("\n--- Comparativa train vs test por columna ---")
    comp = compare_train_test(train_df, test_df, NUM_BINS)
    print(comp.head(20))

    # Guardar a CSV
    summary_train.to_csv(os.path.join(OUTPUT_DIR, "summary_train.csv"), index=False)
    summary_test.to_csv(os.path.join(OUTPUT_DIR, "summary_test.csv"), index=False)
    top_train.to_csv(os.path.join(OUTPUT_DIR, "topfreq_train.csv"), index=False)
    top_test.to_csv(os.path.join(OUTPUT_DIR, "topfreq_test.csv"), index=False)
    comp.to_csv(os.path.join(OUTPUT_DIR, "compare_train_test.csv"), index=False)
    re_dist.to_csv(os.path.join(OUTPUT_DIR, "reason_end_train_dist.csv"), index=False)

    print(f"\nReportes guardados en: {os.path.abspath(OUTPUT_DIR)}")
    print("Archivos:")
    print(" - summary_train.csv / summary_test.csv")
    print(" - topfreq_train.csv / topfreq_test.csv")
    print(" - compare_train_test.csv")
    print(" - reason_end_train_dist.csv (si aplica)")
    print("\nListo [OK]")

if __name__ == "__main__":
    main()
