"""
Data Storytelling Tools

LangChain tools for the DataStorytellerAgent that enable:
- KPI computation from datasets
- Pattern and anomaly detection
- Data story spine construction
- Analysis integrity validation
- Group comparison analysis
"""

from typing import Annotated, Dict, List, Optional, Tuple, Any

from langchain.tools import tool
from langgraph.prebuilt import InjectedState


# ---------------------------------------------------------------------------
# Tool 1: Profile data for storytelling
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def profile_data_for_story(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    n_sample: int = 30,
) -> Tuple[str, Dict]:
    """
    Tool: profile_data_for_story
    Description:
        Performs a comprehensive data profile optimized for storytelling.
        Returns shape, column types, missing rates, cardinality, numeric
        distributions, top categorical values, and date range (if applicable).
        Use this as your FIRST tool to deeply understand the dataset before
        crafting any narrative.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        n_sample (int, default=30): Number of sample rows to show.

    LLM Guidance:
        Always call this tool first when analyzing a new dataset. It provides
        the foundational context needed to identify what stories the data can tell.

    Returns:
        Tuple[str, Dict]: Narrative summary (content) and profile artifact (dict).
    """
    print("    * Tool: profile_data_for_story")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    profile = {}
    lines = []

    # --- Basic shape ---
    n_rows, n_cols = df.shape
    profile["shape"] = {"rows": n_rows, "columns": n_cols}
    lines.append("## Dataset Profile\n")
    lines.append("- **Shape**: %d rows x %d columns" % (n_rows, n_cols))
    lines.append("- **Memory**: %.1f KB" % (df.memory_usage(deep=True).sum() / 1024))

    # --- Column types ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    parsed_datetime_cols = []
    for col in categorical_cols:
        if any(kw in col.lower() for kw in ["date", "time", "dt", "created", "updated", "timestamp"]):
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                parsed_datetime_cols.append(col)
            except Exception:
                pass

    profile["column_types"] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "bool": bool_cols,
        "likely_datetime_strings": parsed_datetime_cols,
    }
    lines.append("\n### Column Types")
    lines.append("- Numeric (%d): %s" % (len(numeric_cols), ", ".join(numeric_cols[:10]) or "None"))
    lines.append("- Categorical (%d): %s" % (len(categorical_cols), ", ".join(categorical_cols[:10]) or "None"))
    lines.append("- Datetime (%d): %s" % (len(datetime_cols), ", ".join(datetime_cols) or "None"))
    if parsed_datetime_cols:
        lines.append("- Likely datetime strings (%d): %s" % (len(parsed_datetime_cols), ", ".join(parsed_datetime_cols)))

    # --- Missing values ---
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_pct", ascending=False)

    profile["missing"] = missing_df.to_dict()
    lines.append("\n### Missing Values")
    if missing_df.empty:
        lines.append("- No missing values detected. (Complete dataset)")
    else:
        lines.append("- %d columns have missing values:" % len(missing_df))
        for col, row in missing_df.head(10).iterrows():
            severity = "[HIGH]" if row["missing_pct"] > 20 else ("[MOD]" if row["missing_pct"] > 5 else "[LOW]")
            lines.append("  - `%s`: %d (%.1f%%) %s" % (col, row["missing_count"], row["missing_pct"], severity))

    # --- Numeric summary ---
    if numeric_cols:
        num_summary = df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        num_summary.columns = ["mean", "std", "min", "median", "max"]
        profile["numeric_summary"] = num_summary.round(4).to_dict()
        lines.append("\n### Numeric Column Summaries (top %d)" % min(8, len(numeric_cols)))
        for col in numeric_cols[:8]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                lines.append(
                    "- `%s`: mean=%.3g, median=%.3g, std=%.3g, range=[%.3g, %.3g]" % (
                        col, col_data.mean(), col_data.median(),
                        col_data.std(), col_data.min(), col_data.max()
                    )
                )

    # --- Categorical summary ---
    if categorical_cols:
        cat_profile = {}
        lines.append("\n### Categorical Column Summaries (top %d)" % min(8, len(categorical_cols)))
        for col in categorical_cols[:8]:
            vc = df[col].value_counts(dropna=False)
            cardinality = df[col].nunique()
            top_val = vc.index[0] if len(vc) > 0 else None
            top_pct = (vc.iloc[0] / n_rows * 100) if len(vc) > 0 else 0
            cat_profile[col] = {
                "cardinality": cardinality,
                "top_value": str(top_val),
                "top_pct": round(top_pct, 2),
            }
            dominant_flag = " [DOMINANT]" if top_pct > 80 else ""
            lines.append(
                "- `%s`: %d unique values, top='%s' (%.1f%%)%s" % (
                    col, cardinality, str(top_val)[:40], top_pct, dominant_flag
                )
            )
        profile["categorical_summary"] = cat_profile

    # --- Datetime range ---
    all_dt_cols = datetime_cols + parsed_datetime_cols
    if all_dt_cols:
        lines.append("\n### Date/Time Coverage")
        for col in all_dt_cols[:3]:
            try:
                dt_series = pd.to_datetime(df[col], infer_datetime_format=True)
                lines.append(
                    "- `%s`: %s to %s (%d unique dates)" % (
                        col, str(dt_series.min()), str(dt_series.max()), dt_series.nunique()
                    )
                )
            except Exception:
                pass

    # --- Duplicate check ---
    n_dupes = df.duplicated().sum()
    dupe_pct = (n_dupes / n_rows * 100) if n_rows > 0 else 0
    profile["duplicates"] = {"count": int(n_dupes), "pct": round(dupe_pct, 2)}
    lines.append("\n### Data Integrity Quick Check")
    if n_dupes > 0:
        lines.append("[WARN] %d duplicate rows (%.1f%%) detected" % (n_dupes, dupe_pct))
    else:
        lines.append("[OK] No exact duplicate rows")

    # --- Sample rows ---
    lines.append("\n### Sample Data (%d rows)" % min(n_sample, n_rows))
    lines.append("```\n%s\n```" % df.head(min(n_sample, 5)).to_string())

    content = "\n".join(lines)
    return content, profile


# ---------------------------------------------------------------------------
# Tool 2: Compute KPIs
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def compute_dataset_kpis(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    kpi_instructions: str = "Compute standard KPIs including counts, totals, averages, and key rates.",
    target_column: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Tool: compute_dataset_kpis
    Description:
        Computes decision-grade KPIs and key metrics from the dataset.
        Returns: row/column counts, numeric summaries (sum, mean, median, std,
        percentiles), categorical distributions, optional target variable analysis,
        and date range metrics if a datetime column is present.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        kpi_instructions (str): Guidance on which KPIs to prioritize.
        target_column (str, optional): If provided, compute target-specific metrics
            (distribution, class balance, etc.).

    LLM Guidance:
        Use when asked to define or compute KPIs, metrics, or performance indicators.
        Specify target_column when working on a classification/regression problem.

    Returns:
        Tuple[str, Dict]: KPI narrative (content) and KPI artifact (dict).
    """
    print("    * Tool: compute_dataset_kpis")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    kpis = {}
    lines = ["## Dataset KPIs\n"]

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Structural KPIs ---
    complete_rows = int(df.dropna().shape[0])
    dup_rows = int(df.duplicated().sum())
    missing_cells = int(df.isnull().sum().sum())
    completeness = round((1 - missing_cells / (n_rows * n_cols)) * 100, 2) if n_rows * n_cols > 0 else 100.0

    kpis["structural"] = {
        "row_count": int(n_rows),
        "column_count": int(n_cols),
        "complete_rows": complete_rows,
        "complete_rows_pct": round(complete_rows / n_rows * 100, 2) if n_rows > 0 else 0,
        "duplicate_rows": dup_rows,
        "total_missing_cells": missing_cells,
        "overall_completeness_pct": completeness,
    }
    lines.append("### Structural KPIs")
    lines.append("| KPI | Value |")
    lines.append("|-----|-------|")
    for k, v in kpis["structural"].items():
        display_k = k.replace("_", " ").title()
        if isinstance(v, int):
            lines.append("| %s | %s |" % (display_k, "{:,}".format(v)))
        else:
            lines.append("| %s | %s%% |" % (display_k, v))

    # --- Numeric KPIs ---
    if numeric_cols:
        numeric_kpis = {}
        lines.append("\n### Numeric KPIs")
        lines.append("| Column | Count | Sum | Mean | Median | Std | Min | Max | P95 |")
        lines.append("|--------|-------|-----|------|--------|-----|-----|-----|-----|")

        for col in numeric_cols[:15]:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            col_kpi = {
                "count": int(len(col_data)),
                "null_count": int(df[col].isnull().sum()),
                "sum": round(float(col_data.sum()), 4),
                "mean": round(float(col_data.mean()), 4),
                "median": round(float(col_data.median()), 4),
                "std": round(float(col_data.std()), 4),
                "min": round(float(col_data.min()), 4),
                "max": round(float(col_data.max()), 4),
                "p25": round(float(col_data.quantile(0.25)), 4),
                "p75": round(float(col_data.quantile(0.75)), 4),
                "p95": round(float(col_data.quantile(0.95)), 4),
            }
            numeric_kpis[col] = col_kpi
            lines.append(
                "| `%s` | %s | %.2f | %.3g | %.3g | %.3g | %.3g | %.3g | %.3g |" % (
                    col, "{:,}".format(col_kpi["count"]),
                    col_kpi["sum"], col_kpi["mean"], col_kpi["median"],
                    col_kpi["std"], col_kpi["min"], col_kpi["max"], col_kpi["p95"]
                )
            )
        kpis["numeric"] = numeric_kpis

    # --- Categorical KPIs ---
    if categorical_cols:
        cat_kpis = {}
        lines.append("\n### Categorical KPIs")
        lines.append("| Column | Cardinality | Top Value | Top Value % | Null % |")
        lines.append("|--------|-------------|-----------|-------------|--------|")

        for col in categorical_cols[:10]:
            vc = df[col].value_counts(dropna=True)
            cardinality = df[col].nunique()
            null_pct = round(df[col].isnull().sum() / n_rows * 100, 2)
            top_val = str(vc.index[0])[:30] if len(vc) > 0 else "N/A"
            top_pct = round(vc.iloc[0] / n_rows * 100, 2) if len(vc) > 0 else 0
            top_5 = {str(k): int(v) for k, v in vc.head(5).items()}

            cat_kpis[col] = {
                "cardinality": int(cardinality),
                "null_pct": null_pct,
                "top_value": top_val,
                "top_value_pct": top_pct,
                "top_5_values": top_5,
            }
            lines.append(
                "| `%s` | %s | %s | %.1f%% | %.1f%% |" % (
                    col, "{:,}".format(cardinality), top_val, top_pct, null_pct
                )
            )
        kpis["categorical"] = cat_kpis

    # --- Target variable KPIs ---
    if target_column and target_column in df.columns:
        target_data = df[target_column]
        target_kpi = {}
        lines.append("\n### Target Variable KPIs: `%s`" % target_column)

        if target_data.dtype in [np.float64, np.int64, np.float32, np.int32]:
            target_kpi = {
                "type": "numeric",
                "mean": round(float(target_data.mean()), 4),
                "median": round(float(target_data.median()), 4),
                "std": round(float(target_data.std()), 4),
                "min": round(float(target_data.min()), 4),
                "max": round(float(target_data.max()), 4),
                "null_count": int(target_data.isnull().sum()),
            }
            lines.append(
                "- Type: Numeric | Mean: %.4g | Std: %.4g" % (target_kpi["mean"], target_kpi["std"])
            )
        else:
            vc = target_data.value_counts(normalize=True)
            imbalance = round(float(vc.max() / vc.min()), 2) if len(vc) > 1 and vc.min() > 0 else None
            target_kpi = {
                "type": "categorical",
                "cardinality": int(target_data.nunique()),
                "class_distribution": {str(k): round(v * 100, 2) for k, v in vc.items()},
                "imbalance_ratio": imbalance,
            }
            lines.append("- Type: Categorical | Classes: %d" % target_kpi["cardinality"])
            for cls, pct in list(target_kpi["class_distribution"].items())[:8]:
                lines.append("  - `%s`: %.1f%%" % (cls, pct))
            if imbalance:
                flag = " [WARN: IMBALANCED]" if imbalance > 5 else ""
                lines.append("- Imbalance ratio: %.2f%s" % (imbalance, flag))

        kpis["target"] = target_kpi

    # --- Date KPIs ---
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in df.columns:
        if col not in datetime_cols and any(kw in col.lower() for kw in ["date", "time", "created", "updated", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                datetime_cols.append(col)
            except Exception:
                pass

    if datetime_cols:
        date_kpis = {}
        lines.append("\n### Date/Time KPIs")
        for col in datetime_cols[:3]:
            try:
                dt = df[col].dropna()
                date_kpis[col] = {
                    "min": str(dt.min()),
                    "max": str(dt.max()),
                    "range_days": int((dt.max() - dt.min()).days),
                    "unique_dates": int(dt.nunique()),
                    "null_count": int(df[col].isnull().sum()),
                }
                lines.append(
                    "- `%s`: %s to %s (%d days, %d unique)" % (
                        col, date_kpis[col]["min"], date_kpis[col]["max"],
                        date_kpis[col]["range_days"], date_kpis[col]["unique_dates"]
                    )
                )
            except Exception as e:
                lines.append("- `%s`: Could not parse as datetime -- %s" % (col, str(e)))
        kpis["dates"] = date_kpis

    content = "\n".join(lines)
    return content, kpis


# ---------------------------------------------------------------------------
# Tool 3: Detect data patterns
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def detect_data_patterns(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    n_sample: int = 2000,
) -> Tuple[str, Dict]:
    """
    Tool: detect_data_patterns
    Description:
        Detects patterns, trends, anomalies, and structural signals in the dataset.
        Covers: outlier detection (IQR method), distribution shape (skewness/kurtosis),
        top correlations, categorical concentration, and time trends (if datetime present).

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        n_sample (int, default=2000): Max rows to sample for performance on large datasets.

    LLM Guidance:
        Use when looking for hidden patterns, trends, outliers, or anomalies.
        Essential for the Tension and Insight parts of the story spine.

    Returns:
        Tuple[str, Dict]: Pattern narrative (content) and patterns artifact (dict).
    """
    print("    * Tool: detect_data_patterns")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    if len(df) > n_sample:
        df_sample = df.sample(n=n_sample, random_state=42)
    else:
        df_sample = df.copy()

    patterns = {}
    lines = ["## Data Pattern Detection\n"]

    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_sample.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Distribution shape ---
    if numeric_cols:
        dist_patterns = {}
        lines.append("### Distribution Shape (Numeric Columns)")
        lines.append("| Column | Skewness | Kurtosis | Shape Signal |")
        lines.append("|--------|----------|----------|--------------|")

        for col in numeric_cols[:12]:
            col_data = df_sample[col].dropna()
            if len(col_data) < 4:
                continue
            skew = round(float(col_data.skew()), 3)
            kurt = round(float(col_data.kurtosis()), 3)

            if abs(skew) > 1:
                shape = "Heavily skewed"
            elif abs(skew) > 0.5:
                shape = "Moderately skewed"
            else:
                shape = "Approx normal"

            if kurt > 3:
                shape += " + heavy tails"
            elif kurt < -1:
                shape += " + light tails"

            dist_patterns[col] = {"skewness": skew, "kurtosis": kurt, "shape": shape}
            lines.append("| `%s` | %.3f | %.3f | %s |" % (col, skew, kurt, shape))

        patterns["distributions"] = dist_patterns

    # --- Outlier detection (IQR) ---
    if numeric_cols:
        outlier_info = {}
        lines.append("\n### Outlier Detection (IQR Method, 1.5x IQR fence)")
        lines.append("| Column | Outlier Count | Outlier % | Lower Fence | Upper Fence |")
        lines.append("|--------|---------------|-----------|-------------|-------------|")

        for col in numeric_cols[:12]:
            col_data = df_sample[col].dropna()
            if len(col_data) < 4:
                continue
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = int(((col_data < lower) | (col_data > upper)).sum())
            pct = round(n_outliers / len(col_data) * 100, 2)
            flag = " [HIGH]" if pct > 5 else ""

            outlier_info[col] = {
                "n_outliers": n_outliers,
                "outlier_pct": pct,
                "lower_fence": round(float(lower), 4),
                "upper_fence": round(float(upper), 4),
            }
            lines.append(
                "| `%s` | %s | %.1f%%%s | %.3g | %.3g |" % (
                    col, "{:,}".format(n_outliers), pct, flag, lower, upper
                )
            )

        patterns["outliers"] = outlier_info

    # --- Correlations ---
    if len(numeric_cols) >= 2:
        corr_matrix = df_sample[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(
            pd.DataFrame(
                [[i < j for j in range(len(numeric_cols))] for i in range(len(numeric_cols))],
                columns=numeric_cols,
                index=numeric_cols,
            )
        )
        top_corrs = (
            upper_tri.stack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "abs_corr"})
            .sort_values("abs_corr", ascending=False)
            .head(10)
        )

        lines.append("\n### Top Correlations")
        lines.append("| Column A | Column B | |r| | Signal |")
        lines.append("|----------|----------|------|--------|")

        corr_records = []
        for _, row in top_corrs.iterrows():
            r = round(float(row["abs_corr"]), 3)
            signal = "Very strong" if r > 0.8 else ("Strong" if r > 0.6 else ("Moderate" if r > 0.4 else "Weak"))
            corr_records.append({
                "col_a": row["col_a"], "col_b": row["col_b"],
                "abs_corr": r, "signal": signal
            })
            lines.append("| `%s` | `%s` | %.3f | %s |" % (row["col_a"], row["col_b"], r, signal))

        patterns["correlations"] = corr_records

    # --- Categorical concentration ---
    if categorical_cols:
        cat_patterns = {}
        lines.append("\n### Categorical Concentration")
        lines.append("| Column | Top Value | Concentration | Signal |")
        lines.append("|--------|-----------|---------------|--------|")

        for col in categorical_cols[:8]:
            vc = df_sample[col].value_counts(normalize=True)
            if len(vc) == 0:
                continue
            top_val = str(vc.index[0])
            top_pct = round(float(vc.iloc[0]) * 100, 2)

            if top_pct > 90:
                signal = "Near-constant (low info)"
            elif top_pct > 70:
                signal = "High concentration"
            elif top_pct > 50:
                signal = "Moderate concentration"
            else:
                signal = "Well distributed"

            cat_patterns[col] = {"top_value": top_val, "top_pct": top_pct, "signal": signal}
            lines.append("| `%s` | %s | %.1f%% | %s |" % (col, top_val[:25], top_pct, signal))

        patterns["categorical_concentration"] = cat_patterns

    # --- Time trends (if datetime) ---
    datetime_cols = df_sample.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in df_sample.columns:
        if col not in datetime_cols and any(kw in col.lower() for kw in ["date", "time", "created", "timestamp"]):
            try:
                df_sample = df_sample.copy()
                df_sample[col] = pd.to_datetime(df_sample[col], infer_datetime_format=True)
                datetime_cols.append(col)
            except Exception:
                pass

    if datetime_cols and numeric_cols:
        lines.append("\n### Time Trend Signals")
        for dt_col in datetime_cols[:1]:
            try:
                df_sorted = df_sample.sort_values(dt_col)
                for num_col in numeric_cols[:3]:
                    half = len(df_sorted) // 2
                    first_half_mean = df_sorted[num_col].iloc[:half].mean()
                    second_half_mean = df_sorted[num_col].iloc[half:].mean()
                    if first_half_mean != 0 and not pd.isna(first_half_mean):
                        pct_change = round(
                            (second_half_mean - first_half_mean) / abs(first_half_mean) * 100, 1
                        )
                        direction = "[UP]" if pct_change > 5 else ("[DOWN]" if pct_change < -5 else "[STABLE]")
                        lines.append(
                            "- `%s` over `%s`: %s (%+.1f%% first vs second half)" % (
                                num_col, dt_col, direction, pct_change
                            )
                        )
                        patterns.setdefault("time_trends", {})[num_col] = {
                            "direction": direction, "pct_change": pct_change
                        }
            except Exception as e:
                lines.append("- Time trend analysis failed for `%s`: %s" % (dt_col, str(e)))

    content = "\n".join(lines)
    return content, patterns


# ---------------------------------------------------------------------------
# Tool 4: Validate analysis integrity
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def validate_analysis_integrity(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    checks: str = "all",
) -> Tuple[str, Dict]:
    """
    Tool: validate_analysis_integrity
    Description:
        Audits the dataset for common data quality and analysis integrity issues.
        Checks: duplicate rows, null patterns, type mismatches, cardinality anomalies,
        negative values in positive-expected columns, near-constant columns, and
        potential ID columns masquerading as numeric features.
        Returns a PASS / PASS_WITH_WARNINGS / FAIL verdict.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        checks (str, default='all'): Comma-separated subset of checks to run:
            'duplicates', 'nulls', 'types', 'cardinality', 'negatives', 'constants'.
            Use 'all' to run all checks.

    LLM Guidance:
        Use before finalizing any analysis to catch silent failure modes.
        Always include this check in the INSIGHTS_REPORT integrity section.

    Returns:
        Tuple[str, Dict]: Verdict narrative (content) and detailed check results (dict).
    """
    print("    * Tool: validate_analysis_integrity")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    n_rows, n_cols = df.shape
    results = {}
    warnings = []
    failures = []
    check_lines = []

    run_all = checks.strip().lower() == "all"
    check_list = [c.strip() for c in checks.split(",")] if not run_all else []

    def should_run(name):
        return run_all or name in check_list

    # --- Duplicate rows ---
    if should_run("duplicates"):
        n_dupes = df.duplicated().sum()
        dupe_pct = round(n_dupes / n_rows * 100, 2) if n_rows > 0 else 0
        results["duplicates"] = {"count": int(n_dupes), "pct": dupe_pct}
        if n_dupes > 0:
            msg = "[WARN] %s duplicate rows (%.1f%%) -- may inflate aggregates" % ("{:,}".format(int(n_dupes)), dupe_pct)
            warnings.append(msg)
            check_lines.append(msg)
        else:
            check_lines.append("[OK] No duplicate rows")

    # --- Null patterns ---
    if should_run("nulls"):
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts / n_rows > 0.20]
        all_null_cols = null_counts[null_counts == n_rows]
        results["nulls"] = {
            "high_null_columns": {k: int(v) for k, v in high_null_cols.items()},
            "all_null_columns": list(all_null_cols.index),
        }
        if len(all_null_cols) > 0:
            msg = "[FAIL] Columns with 100%% nulls: %s" % list(all_null_cols.index)
            failures.append(msg)
            check_lines.append(msg)
        if len(high_null_cols) > 0:
            msg = "[WARN] Columns with >20%% nulls: %s" % list(high_null_cols.index.tolist())[:8]
            warnings.append(msg)
            check_lines.append(msg)
        if len(all_null_cols) == 0 and len(high_null_cols) == 0:
            check_lines.append("[OK] Null rates acceptable (all columns <20%%)")

    # --- Type anomalies ---
    if should_run("types"):
        type_issues = []
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                success_rate = converted.notna().sum() / n_rows
                if success_rate > 0.8:
                    type_issues.append(
                        "`%s` stored as object but %.0f%% parseable as numeric" % (col, success_rate * 100)
                    )
            except Exception:
                pass

        results["type_issues"] = type_issues
        if type_issues:
            for issue in type_issues[:5]:
                msg = "[WARN] Type mismatch: %s" % issue
                warnings.append(msg)
                check_lines.append(msg)
        else:
            check_lines.append("[OK] No obvious type mismatches detected")

    # --- Cardinality anomalies ---
    if should_run("cardinality"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cardinality_issues = []
        for col in numeric_cols:
            cardinality = df[col].nunique()
            if cardinality > n_rows * 0.9 and cardinality > 100:
                cardinality_issues.append(
                    "`%s` may be an ID column (cardinality=%d, %.0f%% unique)" % (
                        col, cardinality, cardinality / n_rows * 100
                    )
                )
        results["cardinality_issues"] = cardinality_issues
        if cardinality_issues:
            for issue in cardinality_issues[:5]:
                msg = "[WARN] Likely ID column treated as feature: %s" % issue
                warnings.append(msg)
                check_lines.append(msg)
        else:
            check_lines.append("[OK] No suspicious ID-like numeric columns detected")

    # --- Negative values in positive-expected columns ---
    if should_run("negatives"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        neg_issues = []
        positive_keywords = [
            "price", "amount", "cost", "revenue", "sales", "age", "count",
            "qty", "quantity", "salary", "income", "weight", "height", "duration"
        ]
        for col in numeric_cols:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            if any(kw in col_lower for kw in positive_keywords):
                n_neg = int((df[col] < 0).sum())
                if n_neg > 0:
                    neg_issues.append(
                        "`%s`: %s negative values (unexpected for this column type)" % (
                            col, "{:,}".format(n_neg)
                        )
                    )

        results["negative_value_issues"] = neg_issues
        if neg_issues:
            for issue in neg_issues:
                msg = "[WARN] Negative values in positive-expected column: %s" % issue
                warnings.append(msg)
                check_lines.append(msg)
        else:
            check_lines.append("[OK] No unexpected negative values in positive-expected columns")

    # --- Near-constant columns ---
    if should_run("constants"):
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append("`%s` (0 variance -- all values identical)" % col)
            elif df[col].nunique() == 2:
                vc = df[col].value_counts(normalize=True)
                if vc.iloc[0] > 0.98:
                    constant_cols.append(
                        "`%s` (%.1f%% is '%s' -- near-constant)" % (col, vc.iloc[0] * 100, vc.index[0])
                    )

        results["constant_columns"] = constant_cols
        if constant_cols:
            for col_msg in constant_cols[:5]:
                msg = "[WARN] Near-constant column (low analytical value): %s" % col_msg
                warnings.append(msg)
                check_lines.append(msg)
        else:
            check_lines.append("[OK] No near-constant columns detected")

    # --- Verdict ---
    if failures:
        verdict = "FAIL"
        verdict_label = "[FAIL]"
    elif warnings:
        verdict = "PASS_WITH_WARNINGS"
        verdict_label = "[WARN]"
    else:
        verdict = "PASS"
        verdict_label = "[PASS]"

    results["verdict"] = verdict
    results["warning_count"] = len(warnings)
    results["failure_count"] = len(failures)

    summary = [
        "## Analysis Integrity Check\n",
        "### Verdict: %s **%s**" % (verdict_label, verdict),
        "- %d failure(s), %d warning(s)" % (len(failures), len(warnings)),
        "",
    ]
    if failures:
        summary.append("**Failures (must fix):**")
        summary.extend("  - %s" % f for f in failures)
    if warnings:
        summary.append("**Warnings (investigate):**")
        summary.extend("  - %s" % w for w in warnings[:10])

    summary.append("\n**Detailed Check Results:**")
    summary.extend(check_lines)

    content = "\n".join(summary)
    return content, results


# ---------------------------------------------------------------------------
# Tool 5: Compute group comparisons
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def compute_group_comparisons(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    group_column: str = "",
    value_columns: str = "",
) -> Tuple[str, Dict]:
    """
    Tool: compute_group_comparisons
    Description:
        Compares groups within the data to surface differences, patterns, and
        potential Simpson's paradox. Computes per-group statistics (count, mean,
        median, std) for specified value columns. If no columns are specified,
        auto-selects the top categorical column for grouping and top numeric columns.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        group_column (str): Column to group by (categorical). Leave empty to auto-select.
        value_columns (str): Comma-separated numeric columns to compare. Leave empty to auto-select top 3.

    LLM Guidance:
        Use when looking for group differences, segment analysis, or comparisons
        between categories. Check Simpson's paradox: if the overall trend contradicts
        per-group trends, flag it explicitly.

    Returns:
        Tuple[str, Dict]: Group comparison narrative (content) and stats artifact (dict).
    """
    print("    * Tool: compute_group_comparisons")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    results = {}
    lines = ["## Group Comparison Analysis\n"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Auto-select group column
    if not group_column or group_column not in df.columns:
        if not categorical_cols:
            return "No categorical columns available for grouping.", {"error": "no categorical columns"}
        best_col = None
        for col in categorical_cols:
            n_unique = df[col].nunique()
            if 2 <= n_unique <= 20:
                best_col = col
                break
        if best_col is None:
            best_col = categorical_cols[0]
        group_column = best_col
        lines.append("_(Auto-selected group column: `%s`)_\n" % group_column)
    else:
        lines.append("_Group column: `%s`_\n" % group_column)

    # Auto-select value columns
    if not value_columns:
        selected_cols = numeric_cols[:5]
    else:
        selected_cols = [c.strip() for c in value_columns.split(",") if c.strip() in df.columns]
        if not selected_cols:
            selected_cols = numeric_cols[:5]

    if not selected_cols:
        return "No numeric columns found to compare across `%s` groups." % group_column, {"error": "no numeric columns"}

    lines.append("**Grouping by:** `%s` (%d groups)\n" % (group_column, df[group_column].nunique()))
    lines.append("**Comparing:** %s\n" % ", ".join("`%s`" % c for c in selected_cols))

    # Group stats
    grouped = df.groupby(group_column, observed=True)
    group_stats = {}

    for col in selected_cols:
        col_stats = grouped[col].agg(["count", "mean", "median", "std"]).round(4)
        col_stats.columns = ["count", "mean", "median", "std"]
        group_stats[col] = col_stats.to_dict(orient="index")

        lines.append("### `%s` by `%s`" % (col, group_column))
        lines.append("| Group | Count | Mean | Median | Std |")
        lines.append("|-------|-------|------|--------|-----|")

        for group_name, stats in col_stats.iterrows():
            lines.append(
                "| %s | %s | %.4g | %.4g | %.4g |" % (
                    group_name, "{:,.0f}".format(stats["count"]),
                    stats["mean"], stats["median"], stats["std"]
                )
            )

        # Overall stats
        overall_mean = df[col].mean()
        overall_median = df[col].median()
        lines.append(
            "| **Overall** | **%s** | **%.4g** | **%.4g** | -- |" % (
                "{:,}".format(len(df)), overall_mean, overall_median
            )
        )

        # Simpson's paradox hint
        group_means = col_stats["mean"].dropna()
        if len(group_means) >= 2:
            all_above = bool((group_means > overall_mean).all())
            all_below = bool((group_means < overall_mean).all())
            if all_above or all_below:
                direction_label = "above" if all_above else "below"
                lines.append(
                    "\n[WARN] Simpson's Paradox Check: All group means are %s the overall mean"
                    " -- investigate sub-group composition (size weights matter)." % direction_label
                )
            else:
                if overall_mean != 0:
                    max_diff_pct = round(
                        (float(group_means.max()) - float(group_means.min())) / abs(float(overall_mean)) * 100, 1
                    )
                    lines.append("\n_Max group spread: %.1f%% of overall mean_" % max_diff_pct)
        lines.append("")

    results["group_column"] = group_column
    results["value_columns"] = selected_cols
    results["group_stats"] = group_stats
    results["group_sizes"] = {str(k): int(v) for k, v in df[group_column].value_counts().items()}

    content = "\n".join(lines)
    return content, results


# ---------------------------------------------------------------------------
# Tool 6: Build story spine (pure formatter, no data injection)
# ---------------------------------------------------------------------------

@tool(response_format="content")
def build_story_spine(
    context: str,
    tension: str,
    insight: str,
    implication: str,
    action: str,
) -> str:
    """
    Tool: build_story_spine
    Description:
        Formats storytelling findings into the professional 5-part story spine:
        Context -> Tension -> Insight -> Implication -> Action.
        Call this tool AFTER you have analyzed the data and are ready to structure
        your narrative. Fill each part based on what you discovered.

    Parameters:
        context (str): What is happening / background situation (1-3 sentences).
        tension (str): What is surprising, problematic, or worth investigating (1-3 sentences).
        insight (str): What the data reveals / the core finding (2-4 sentences).
        implication (str): What this means for the business/stakeholders (1-3 sentences).
        action (str): What should be done next / recommendations (1-3 sentences).

    LLM Guidance:
        Use this to crystallize your analysis into a clear narrative before writing
        the final report. The story spine becomes the backbone of your executive summary.
        Call AFTER running profile/pattern/KPI tools.

    Returns:
        str: Formatted story spine as markdown.
    """
    print("    * Tool: build_story_spine")
    from ai_data_science_team.utils.storytelling import format_story_spine

    return format_story_spine(
        context=context,
        tension=tension,
        insight=insight,
        implication=implication,
        action=action,
    )
