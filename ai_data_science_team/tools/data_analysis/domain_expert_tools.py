"""
Data Domain Expert Tools

LangChain tools for the DataDomainExpertAgent (DDEA) that enable:
- Full schema inference and type analysis
- Deep per-column profiling with anomaly detection
- Edge case and nuance detection
- Join key quality analysis (single and cross-dataset)
- Validation rule generation (VAL-### style)
- ML readiness assessment
- Ordered cleaning plan generation
- DATA_DOSSIER.md compilation
"""

from typing import Annotated, Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langgraph.prebuilt import InjectedState


# ---------------------------------------------------------------------------
# Tool 1: Profile dataset schema
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def profile_dataset_schema(
    data_raw: Annotated[dict, InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: profile_dataset_schema
    Description:
        Performs exhaustive schema inference on the dataset. Returns column names,
        inferred types (with confidence and rationale), nullable status, cardinality
        tier (unique/high/medium/low), potential key columns, sentinel null
        representations, and type ambiguities that need resolution.

        This is Phase 1+2 of DDEA profiling. Call this FIRST on any new dataset.

    Parameters:
        data_raw (dict): Raw data injected from agent state.

    LLM Guidance:
        Always call this as the first step. The schema is the foundation for all
        subsequent analysis. Pay special attention to type ambiguities and columns
        that appear to be IDs, dates, or categoricals stored as numbers.

    Returns:
        Tuple[str, Dict]: Schema narrative (content) and schema artifact (dict).
    """
    print("    * Tool: profile_dataset_schema")
    import pandas as pd
    import numpy as np
    import re

    df = pd.DataFrame(data_raw)
    n_rows, n_cols = df.shape
    schema_rows = []
    schema_artifact = {
        "row_count": n_rows,
        "col_count": n_cols,
        "columns": {}
    }

    lines = ["## Schema & Type Analysis\n"]
    lines.append("**Dataset:** %d rows x %d columns\n" % (n_rows, n_cols))
    lines.append("| Column | Pandas Type | Inferred Type | Confidence | Nullable | Cardinality Tier | Key Candidate | Notes |")
    lines.append("|--------|-------------|---------------|------------|----------|-----------------|---------------|-------|")

    for col in df.columns:
        series = df[col]
        pandas_type = str(series.dtype)
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0
        cardinality = int(series.nunique())
        car_ratio = cardinality / n_rows if n_rows > 0 else 0

        # Cardinality tier
        if car_ratio > 0.95 and cardinality > 50:
            car_tier = "unique (likely ID)"
        elif car_ratio > 0.5:
            car_tier = "high"
        elif car_ratio > 0.05:
            car_tier = "medium"
        else:
            car_tier = "low"

        # Key candidate detection
        is_key = (car_ratio > 0.95 and null_count == 0 and cardinality == n_rows)
        key_flag = "YES" if is_key else ""

        # Type inference
        notes = []
        inferred_type = pandas_type
        confidence = "high"

        col_lower = col.lower().replace("_", " ").replace("-", " ")

        if pandas_type in ("object", "string"):
            # Try numeric
            numeric_try = pd.to_numeric(series, errors="coerce")
            numeric_rate = numeric_try.notna().sum() / max(series.notna().sum(), 1)

            # Try datetime
            dt_try = None
            dt_rate = 0.0
            try:
                dt_try = pd.to_datetime(series, errors="coerce")
                dt_rate = dt_try.notna().sum() / max(series.notna().sum(), 1)
            except Exception:
                pass

            if numeric_rate > 0.9:
                inferred_type = "numeric (stored as string)"
                confidence = "high" if numeric_rate > 0.98 else "medium"
                notes.append("[WARN] Should be cast to numeric")
            elif dt_rate > 0.85:
                inferred_type = "datetime (stored as string)"
                confidence = "high" if dt_rate > 0.98 else "medium"
                notes.append("[WARN] Should be parsed as datetime")
            elif cardinality <= 30 or car_ratio < 0.05:
                inferred_type = "categorical"
                confidence = "high"
            elif any(kw in col_lower for kw in ["id", "code", "key", "ref", "no", "num", "number"]):
                inferred_type = "identifier/categorical (high card)"
                confidence = "medium"
                notes.append("May be an ID or code field")
            else:
                inferred_type = "text/free-form"
                confidence = "medium"

        elif pandas_type in ("int64", "int32", "int16", "int8", "Int64"):
            if any(kw in col_lower for kw in ["id", "code", "key", "ref"]) or (car_ratio > 0.9 and n_rows > 100):
                inferred_type = "identifier (int)"
                confidence = "high"
                notes.append("Do not treat as continuous numeric")
            elif cardinality <= 20:
                inferred_type = "categorical (int-encoded)"
                confidence = "medium"
                notes.append("Consider treating as categorical")
            else:
                inferred_type = "integer"
                confidence = "high"

        elif pandas_type in ("float64", "float32", "Float64"):
            if all(series.dropna() == series.dropna().astype(int)):
                notes.append("[WARN] All values are whole numbers; may be int stored as float (possible NaN coercion)")
                inferred_type = "integer (stored as float)"
                confidence = "medium"
            else:
                inferred_type = "continuous float"
                confidence = "high"

        elif "datetime" in pandas_type:
            inferred_type = "datetime"
            confidence = "high"

        elif pandas_type == "bool":
            inferred_type = "boolean"
            confidence = "high"

        # Sentinel null detection for object columns
        if pandas_type == "object":
            sentinel_vals = ["unknown", "n/a", "na", "none", "null", "nan", "tbd", "missing",
                             "9999", "-1", "-999", "undefined", "not available", "not applicable"]
            sample_lower = series.dropna().astype(str).str.lower().str.strip()
            found_sentinels = [s for s in sentinel_vals if sample_lower.isin([s]).any()]
            if found_sentinels:
                notes.append("[WARN] Sentinel nulls found: %s" % found_sentinels[:3])

        notes_str = "; ".join(notes)[:120]
        nullable = "YES (%s%%)" % null_pct if null_count > 0 else "No"

        schema_artifact["columns"][col] = {
            "pandas_type": pandas_type,
            "inferred_type": inferred_type,
            "confidence": confidence,
            "null_count": null_count,
            "null_pct": null_pct,
            "cardinality": cardinality,
            "cardinality_tier": car_tier,
            "is_key_candidate": is_key,
            "notes": notes_str,
        }

        lines.append("| `%s` | %s | **%s** | %s | %s | %s | %s | %s |" % (
            col, pandas_type, inferred_type, confidence,
            nullable, car_tier, key_flag, notes_str
        ))

    # Summary findings
    ambiguous = [c for c, v in schema_artifact["columns"].items() if v["confidence"] != "high"]
    type_warnings = [c for c, v in schema_artifact["columns"].items() if "[WARN]" in v["notes"]]
    key_candidates = [c for c, v in schema_artifact["columns"].items() if v["is_key_candidate"]]

    lines.append("\n### Schema Summary")
    lines.append("- Columns with type ambiguity (%d): %s" % (len(ambiguous), ", ".join("`%s`" % c for c in ambiguous[:8])))
    lines.append("- Columns with type warnings (%d): %s" % (len(type_warnings), ", ".join("`%s`" % c for c in type_warnings[:8])))
    lines.append("- Key candidates (%d): %s" % (len(key_candidates), ", ".join("`%s`" % c for c in key_candidates[:5]) or "None found"))

    schema_artifact["summary"] = {
        "ambiguous_columns": ambiguous,
        "type_warnings": type_warnings,
        "key_candidates": key_candidates,
    }

    content = "\n".join(lines)
    return content, schema_artifact


# ---------------------------------------------------------------------------
# Tool 2: Deep column profiling
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def profile_columns_deep(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    columns: str = "",
    n_top_values: int = 10,
) -> Tuple[str, Dict]:
    """
    Tool: profile_columns_deep
    Description:
        Performs deep per-column profiling for each column (or specified columns).
        Returns: missing rates, cardinality, full distribution stats, top N values
        with counts, string format patterns (via regex sampling), date format
        inference, anomaly flags, and zero/negative counts for numerics.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        columns (str): Comma-separated column names to profile. Empty = all columns.
        n_top_values (int, default=10): Number of top values to include per column.

    LLM Guidance:
        Use after profile_dataset_schema. Call for all columns on a new dataset,
        or for specific high-risk columns identified in schema analysis.

    Returns:
        Tuple[str, Dict]: Profile narrative (content) and profile artifact (dict).
    """
    print("    * Tool: profile_columns_deep")
    import pandas as pd
    import numpy as np
    import re

    df = pd.DataFrame(data_raw)
    n_rows = len(df)

    # Select columns to profile
    if columns.strip():
        cols_to_profile = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if not cols_to_profile:
            cols_to_profile = df.columns.tolist()
    else:
        cols_to_profile = df.columns.tolist()

    profiles = {}
    lines = ["## Deep Column Profiles\n"]
    lines.append("Profiling %d columns on %d rows.\n" % (len(cols_to_profile), n_rows))

    for col in cols_to_profile:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isnull().sum())
        null_pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0
        non_null = series.dropna()
        cardinality = int(series.nunique())

        col_profile = {
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": null_pct,
            "cardinality": cardinality,
        }

        lines.append("### Column: `%s` (%s)" % (col, dtype))
        lines.append("- Missing: %d (%.1f%%)" % (null_count, null_pct))
        lines.append("- Cardinality: %d unique values (%.1f%% of rows)" % (cardinality, cardinality / n_rows * 100 if n_rows > 0 else 0))

        # Top values
        vc = series.value_counts(dropna=False).head(n_top_values)
        top_vals = [(str(k)[:50], int(v), round(v / n_rows * 100, 2)) for k, v in vc.items()]
        col_profile["top_values"] = top_vals
        lines.append("- Top %d values:" % min(n_top_values, len(vc)))
        for val, cnt, pct in top_vals[:8]:
            null_marker = " [NULL]" if val in ("None", "nan", "NaT") else ""
            lines.append("  - `%s`%s: %d (%.1f%%)" % (val, null_marker, cnt, pct))

        # Numeric stats
        if dtype in ("int64", "int32", "float64", "float32", "Int64", "Float64") or dtype.startswith("int") or dtype.startswith("float"):
            num_data = pd.to_numeric(series, errors="coerce").dropna()
            if len(num_data) > 0:
                stats = {
                    "min": round(float(num_data.min()), 6),
                    "max": round(float(num_data.max()), 6),
                    "mean": round(float(num_data.mean()), 6),
                    "median": round(float(num_data.median()), 6),
                    "std": round(float(num_data.std()), 6),
                    "p05": round(float(num_data.quantile(0.05)), 6),
                    "p25": round(float(num_data.quantile(0.25)), 6),
                    "p75": round(float(num_data.quantile(0.75)), 6),
                    "p95": round(float(num_data.quantile(0.95)), 6),
                    "zeros": int((num_data == 0).sum()),
                    "negatives": int((num_data < 0).sum()),
                }
                col_profile["stats"] = stats
                lines.append("- Stats: min=%.3g, max=%.3g, mean=%.3g, median=%.3g, std=%.3g" % (
                    stats["min"], stats["max"], stats["mean"], stats["median"], stats["std"]
                ))
                lines.append("- Percentiles: p05=%.3g, p25=%.3g, p75=%.3g, p95=%.3g" % (
                    stats["p05"], stats["p25"], stats["p75"], stats["p95"]
                ))
                if stats["zeros"] > 0:
                    lines.append("- [NOTE] %d zero values (%.1f%%)" % (stats["zeros"], stats["zeros"] / n_rows * 100))
                if stats["negatives"] > 0:
                    lines.append("- [NOTE] %d negative values (%.1f%%)" % (stats["negatives"], stats["negatives"] / n_rows * 100))

        # String pattern analysis
        elif dtype == "object":
            str_series = non_null.astype(str)
            if len(str_series) > 0:
                # Length distribution
                lengths = str_series.str.len()
                col_profile["length_stats"] = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": round(float(lengths.mean()), 2),
                }
                lines.append("- String lengths: min=%d, max=%d, mean=%.1f" % (
                    col_profile["length_stats"]["min"],
                    col_profile["length_stats"]["max"],
                    col_profile["length_stats"]["mean"],
                ))

                # Common patterns
                patterns_detected = []
                sample = str_series.head(500)

                # Email pattern
                email_count = int(sample.str.match(r'^[^@]+@[^@]+\.[^@]+$').sum())
                if email_count > len(sample) * 0.5:
                    patterns_detected.append("email addresses (%d%%)" % int(email_count / len(sample) * 100))

                # UUID/GUID pattern
                uuid_count = int(sample.str.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-').sum())
                if uuid_count > len(sample) * 0.3:
                    patterns_detected.append("UUID/GUID (%d%%)" % int(uuid_count / len(sample) * 100))

                # Date-like pattern
                date_count = int(sample.str.match(r'^\d{4}[-/]\d{2}[-/]\d{2}').sum())
                if date_count > len(sample) * 0.5:
                    patterns_detected.append("date-like (YYYY-MM-DD) (%d%%)" % int(date_count / len(sample) * 100))

                # Phone-like
                phone_count = int(sample.str.match(r'^[\+\(]?[\d\s\-\(\)]{7,15}$').sum())
                if phone_count > len(sample) * 0.3:
                    patterns_detected.append("phone-like (%d%%)" % int(phone_count / len(sample) * 100))

                # Whitespace issues
                ws_count = int(str_series.str.contains(r'^\s|\s$').sum())
                if ws_count > 0:
                    patterns_detected.append("[WARN] %d values with leading/trailing whitespace" % ws_count)

                # Mixed case
                has_upper = str_series.str.contains(r'[A-Z]').any()
                has_lower = str_series.str.contains(r'[a-z]').any()
                if has_upper and has_lower and cardinality > 5:
                    patterns_detected.append("[NOTE] Mixed case values (may need normalization)")

                col_profile["patterns"] = patterns_detected
                if patterns_detected:
                    lines.append("- Patterns detected: %s" % "; ".join(patterns_detected[:5]))

        # Datetime stats
        elif "datetime" in dtype:
            dt_data = series.dropna()
            if len(dt_data) > 0:
                col_profile["date_range"] = {
                    "min": str(dt_data.min()),
                    "max": str(dt_data.max()),
                    "range_days": (dt_data.max() - dt_data.min()).days,
                }
                lines.append("- Date range: %s to %s (%d days)" % (
                    str(dt_data.min())[:10], str(dt_data.max())[:10],
                    col_profile["date_range"]["range_days"]
                ))

        profiles[col] = col_profile
        lines.append("")

    content = "\n".join(lines)
    return content, profiles


# ---------------------------------------------------------------------------
# Tool 3: Detect edge cases
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def detect_edge_cases(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    min_prevalence: float = 0.001,
) -> Tuple[str, Dict]:
    """
    Tool: detect_edge_cases
    Description:
        Deep scan for edge cases and data nuances that can silently corrupt
        downstream pipelines. Detects: sentinel null values, format
        inconsistencies, mixed types in object columns, duplicate boolean
        representations (0/1/True/False/Y/N), unit inconsistencies,
        multi-value delimited strings, whitespace/invisible characters,
        and PII indicator patterns.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        min_prevalence (float, default=0.001): Minimum row fraction for
            an edge case to be reported. Set lower for rare-but-critical issues.

    LLM Guidance:
        Run this after schema profiling. The findings here drive edge-case-aware
        handling in cleaning plans and validation rules. Every finding includes
        handling strategy and risk of mishandling.

    Returns:
        Tuple[str, Dict]: Edge case narrative (content) and findings artifact (dict).
    """
    print("    * Tool: detect_edge_cases")
    import pandas as pd
    import numpy as np
    import re

    df = pd.DataFrame(data_raw)
    n_rows = len(df)
    edge_cases = []
    lines = ["## Edge Cases & Nuances\n"]
    lines.append("Scanning %d rows x %d columns (min_prevalence=%.3f)\n" % (n_rows, len(df.columns), min_prevalence))

    def add_ec(column, issue_type, description, prevalence, examples, handling, risk, do_not=None):
        ec = {
            "column": column,
            "issue_type": issue_type,
            "description": description,
            "prevalence": prevalence,
            "examples": examples,
            "handling_strategy": handling,
            "risk": risk,
            "do_not": do_not or "",
        }
        edge_cases.append(ec)
        lines.append("### EC-%03d: %s in `%s`" % (len(edge_cases), issue_type, column))
        lines.append("- Description: %s" % description)
        lines.append("- Prevalence: %s" % prevalence)
        if examples:
            lines.append("- Examples: %s" % str(examples)[:150])
        lines.append("- Handling: %s" % handling)
        lines.append("- Risk: %s" % risk)
        if do_not:
            lines.append("- Do NOT: %s" % do_not)
        lines.append("")

    sentinel_values = {
        "9999", "99999", "-1", "-999", "-9999", "999", "0000",
        "unknown", "n/a", "na", "none", "null", "nan", "tbd",
        "missing", "undefined", "not available", "not applicable",
        "?", "#n/a", "#null!", "n.a.", "n.a", "(none)", "(null)", "nil",
    }

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_null = series.dropna()
        n_non_null = len(non_null)
        if n_non_null == 0:
            continue

        # --- Sentinel null detection ---
        if dtype == "object":
            str_vals_lower = non_null.astype(str).str.lower().str.strip()
            for sentinel in sentinel_values:
                count = int((str_vals_lower == sentinel).sum())
                if count > 0 and count / n_rows >= min_prevalence:
                    examples_list = list(non_null[str_vals_lower == sentinel].head(3))
                    add_ec(
                        column=col,
                        issue_type="Sentinel null value",
                        description="Value '%s' used as null placeholder instead of actual NaN/None" % sentinel,
                        prevalence="%d rows (%.1f%%)" % (count, count / n_rows * 100),
                        examples=examples_list,
                        handling="Replace '%s' with np.nan before analysis. E.g.: df['%s'].replace('%s', np.nan)" % (sentinel, col, sentinel),
                        risk="If not replaced: incorrectly counted as valid data, statistics will be skewed",
                        do_not="Do not use fillna() before replacing sentinels or sentinel values will be filled in",
                    )

        # --- Mixed boolean representations ---
        if dtype == "object" and non_null.nunique() <= 6:
            bool_patterns = {
            "true_false": {"true", "false"},
            "yes_no": {"yes", "no"},
            "y_n": {"y", "n"},
            "1_0": {"1", "0"},
            "t_f": {"t", "f"},
            }
            vals_lower = set(non_null.astype(str).str.lower().str.strip().unique())
            for pattern_name, pattern_set in bool_patterns.items():
                overlap = vals_lower & pattern_set
                if len(overlap) == 2:
                    # Check if multiple patterns coexist
                    for other_name, other_set in bool_patterns.items():
                        if other_name != pattern_name:
                            other_overlap = vals_lower & other_set
                            if len(other_overlap) >= 1 and len(other_overlap) < len(other_set):
                                add_ec(
                                    column=col,
                                    issue_type="Mixed boolean representations",
                                    description="Multiple boolean encodings coexist: %s and %s" % (
                                        pattern_set, other_set & vals_lower
                                    ),
                                    prevalence="All %d non-null rows" % n_non_null,
                                    examples=list(non_null.unique()[:6]),
                                    handling="Standardize to a single representation (recommend True/False). Map all variants before downstream use.",
                                    risk="If mixed: boolean operations will fail or produce wrong results",
                                )
                                break
                    break

        # --- Leading/trailing whitespace ---
        if dtype == "object":
            ws_mask = non_null.astype(str).str.match(r'^\s+|\s+$')
            ws_count = int(ws_mask.sum())
            if ws_count > 0 and ws_count / n_rows >= min_prevalence:
                examples_ws = list(non_null[ws_mask].head(3))
                add_ec(
                    column=col,
                    issue_type="Whitespace contamination",
                    description="%d values have leading or trailing whitespace" % ws_count,
                    prevalence="%d rows (%.1f%%)" % (ws_count, ws_count / n_rows * 100),
                    examples=["'%s'" % str(v) for v in examples_ws],
                    handling="Apply .str.strip() before any string matching or groupby. E.g.: df['%s'] = df['%s'].str.strip()" % (col, col),
                    risk="Groupby/join will create phantom extra groups (e.g., 'East ' != 'East')",
                    do_not="Do not use isin() or == comparisons without stripping first",
                )

        # --- Multi-value delimited strings ---
        if dtype == "object" and n_non_null > 10:
            sample = non_null.head(200).astype(str)
            for delim in [";", "|", ",", "/"]:
                delim_count = int(sample.str.contains(re.escape(delim), na=False).sum())
                if delim_count > len(sample) * 0.3:
                    add_ec(
                        column=col,
                        issue_type="Multi-value delimited string",
                        description="%d%% of sampled values contain delimiter '%s', suggesting this column stores multiple values per cell" % (
                            int(delim_count / len(sample) * 100), delim
                        ),
                        prevalence="~%d%% of rows (sampled)" % int(delim_count / len(sample) * 100),
                        examples=list(sample[sample.str.contains(re.escape(delim), na=False)].head(3)),
                        handling="Use str.split('%s', expand=True) or explode() to normalize. Re-aggregate carefully to preserve parent linkage." % delim,
                        risk="Row count will explode if you explode without re-aggregation plan. Groupby results will be wrong if used raw.",
                        do_not="Do not use value_counts() on unexploded multi-value columns",
                    )
                    break

        # --- PII indicators ---
        if dtype == "object":
            sample_str = non_null.head(200).astype(str)
            pii_checks = [
                (r'^[^@]+@[^@]+\.[^@]+$', "email addresses"),
                (r'^\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$', "phone numbers"),
                (r'^\d{3}-\d{2}-\d{4}$', "SSN-like values"),
                (r'^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$', "credit card-like values"),
            ]
            for pattern, pii_type in pii_checks:
                pii_count = int(sample_str.str.match(pattern).sum())
                if pii_count > len(sample_str) * 0.2:
                    add_ec(
                        column=col,
                        issue_type="PII indicator",
                        description="Column appears to contain %s (%.0f%% of sampled values match pattern)" % (
                            pii_type, pii_count / len(sample_str) * 100
                        ),
                        prevalence="~%.0f%% of rows (sampled)" % (pii_count / len(sample_str) * 100),
                        examples=["[REDACTED for privacy]"],
                        handling="Mask, tokenize, or pseudonymize before sharing or using in models. Consult privacy policy.",
                        risk="Regulatory compliance risk (GDPR, HIPAA, CCPA). Data breach risk if stored/transmitted unprotected.",
                        do_not="Do not log raw PII values. Do not include in model features without anonymization.",
                    )

        # --- Numeric columns: potential float-stored integers ---
        if dtype in ("float64", "float32"):
            whole_count = int((non_null == non_null.astype(int)).sum()) if len(non_null) > 0 else 0
            if whole_count / max(n_non_null, 1) > 0.98 and n_non_null > 10:
                add_ec(
                    column=col,
                    issue_type="Integer stored as float",
                    description="%.0f%% of non-null values are whole numbers, suggesting int coerced to float (likely due to NaN)" % (
                        whole_count / n_non_null * 100
                    ),
                    prevalence="%d whole-number values (%.1f%%)" % (whole_count, whole_count / n_rows * 100),
                    examples=list(non_null.head(3)),
                    handling="Use pd.Int64Dtype() (nullable integer) or convert to int after fillna/dropna. Avoid float comparisons.",
                    risk="Float precision issues in groupby keys. Unintended float arithmetic on what should be integer IDs.",
                )

    if not edge_cases:
        lines.append("[OK] No significant edge cases detected at prevalence threshold %.3f." % min_prevalence)

    lines.append("\n## Edge Case Summary")
    lines.append("Total edge cases found: %d" % len(edge_cases))
    by_type = {}
    for ec in edge_cases:
        by_type.setdefault(ec["issue_type"], 0)
        by_type[ec["issue_type"]] += 1
    for ec_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append("- %s: %d" % (ec_type, count))

    content = "\n".join(lines)
    return content, {"edge_cases": edge_cases, "total_count": len(edge_cases), "by_type": by_type}


# ---------------------------------------------------------------------------
# Tool 4: Analyze join keys
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def analyze_join_keys(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    secondary_data_raw: Annotated[dict, InjectedState("secondary_data_raw")],
    join_key: str = "",
) -> Tuple[str, Dict]:
    """
    Tool: analyze_join_keys
    Description:
        Analyzes join key quality for one or two datasets. For a single dataset,
        tests key uniqueness, null rates, cardinality, and whether it can serve
        as a primary key. For two datasets, tests match rate, multiplicity
        (1:1, 1:N, M:N), orphan rate, and row explosion risk.

    Parameters:
        data_raw (dict): Primary dataset injected from agent state.
        secondary_data_raw (dict): Secondary dataset injected from agent state.
            Empty dict if only analyzing single-dataset keys.
        join_key (str): Column name to analyze. Leave empty to auto-detect
            candidates from both datasets.

    LLM Guidance:
        Call before any join/merge operation. Always check multiplicity and
        orphan rate. If M:N relationship found, warn loudly and provide
        deduplication strategy.

    Returns:
        Tuple[str, Dict]: Join analysis narrative (content) and join artifact (dict).
    """
    print("    * Tool: analyze_join_keys")
    import pandas as pd
    import numpy as np

    df1 = pd.DataFrame(data_raw)
    df2 = pd.DataFrame(secondary_data_raw) if secondary_data_raw else None
    n1 = len(df1)
    n2 = len(df2) if df2 is not None else 0

    results = {}
    lines = ["## Join Key Analysis\n"]

    mode = "cross-dataset" if df2 is not None and len(df2) > 0 else "single-dataset"
    lines.append("Mode: %s" % mode)
    lines.append("- Primary dataset: %d rows x %d cols" % (n1, len(df1.columns)))
    if df2 is not None and len(df2) > 0:
        lines.append("- Secondary dataset: %d rows x %d cols" % (n2, len(df2.columns)))

    # Auto-detect join key candidates
    if not join_key:
        candidates_1 = [c for c in df1.columns if "id" in c.lower() or "key" in c.lower() or "code" in c.lower()]
        if not candidates_1:
            # Use highest cardinality columns
            candidates_1 = sorted(df1.columns, key=lambda c: df1[c].nunique(), reverse=True)[:3]
        join_key = candidates_1[0] if candidates_1 else df1.columns[0]
        lines.append("\n_(Auto-selected join key: `%s`)_" % join_key)
    else:
        lines.append("\n_Analyzing join key: `%s`_" % join_key)

    # Single-dataset key analysis
    if join_key in df1.columns:
        key_series = df1[join_key]
        null_count = int(key_series.isnull().sum())
        null_pct = round(null_count / n1 * 100, 2)
        cardinality = int(key_series.nunique())
        is_unique = cardinality == n1 - null_count
        dup_count = n1 - cardinality - null_count

        results["primary_key_analysis"] = {
            "column": join_key,
            "null_count": null_count,
            "null_pct": null_pct,
            "cardinality": cardinality,
            "is_unique": is_unique,
            "duplicate_count": max(0, dup_count),
        }

        lines.append("\n### Primary Dataset Key: `%s`" % join_key)
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append("| Null count | %d (%.1f%%) |" % (null_count, null_pct))
        lines.append("| Cardinality | %d unique values |" % cardinality)
        lines.append("| Is Unique (PK candidate) | %s |" % ("YES" if is_unique else "NO"))
        if not is_unique:
            lines.append("| Duplicate key values | ~%d |" % max(0, dup_count))

        if null_count > 0:
            lines.append("\n[WARN] Join key `%s` has %d nulls -- will create orphan rows on inner join" % (join_key, null_count))
        if not is_unique:
            lines.append("[WARN] Key `%s` is NOT unique -- joining will cause row multiplication (N:M risk)" % join_key)
            lines.append("       Top duplicate keys: %s" % list(key_series.value_counts()[key_series.value_counts() > 1].head(5).index))

    # Cross-dataset analysis
    if df2 is not None and len(df2) > 0 and join_key in df2.columns:
        key2 = df2[join_key]
        key1 = df1[join_key] if join_key in df1.columns else pd.Series([], dtype="object")

        # Match rate
        set1 = set(key1.dropna().astype(str))
        set2 = set(key2.dropna().astype(str))
        matched = set1 & set2
        match_rate_1 = round(len(matched) / len(set1) * 100, 2) if set1 else 0
        match_rate_2 = round(len(matched) / len(set2) * 100, 2) if set2 else 0
        orphans_1 = set1 - set2
        orphans_2 = set2 - set1

        # Multiplicity
        mult1 = int(df1[join_key].value_counts().max()) if join_key in df1.columns else 1
        mult2 = int(key2.value_counts().max())

        if mult1 == 1 and mult2 == 1:
            relationship = "1:1 (safe join, no row explosion)"
        elif mult1 == 1:
            relationship = "1:N (left table is the 1 side)"
        elif mult2 == 1:
            relationship = "N:1 (right table is the 1 side)"
        else:
            relationship = "M:N [HIGH RISK] -- row explosion will occur without deduplication"

        results["cross_dataset_analysis"] = {
            "join_key": join_key,
            "match_count": len(matched),
            "match_rate_primary": match_rate_1,
            "match_rate_secondary": match_rate_2,
            "orphans_in_primary": len(orphans_1),
            "orphans_in_secondary": len(orphans_2),
            "multiplicity": relationship,
            "max_mult_primary": mult1,
            "max_mult_secondary": mult2,
        }

        lines.append("\n### Cross-Dataset Join Analysis: `%s`" % join_key)
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append("| Matched keys | %d |" % len(matched))
        lines.append("| Match rate (primary) | %.1f%% |" % match_rate_1)
        lines.append("| Match rate (secondary) | %.1f%% |" % match_rate_2)
        lines.append("| Orphans in primary | %d |" % len(orphans_1))
        lines.append("| Orphans in secondary | %d |" % len(orphans_2))
        lines.append("| Relationship type | **%s** |" % relationship)

        if "HIGH RISK" in relationship:
            lines.append("\n[FAIL] M:N join will EXPLODE rows. Deduplicate at least one side first.")
            lines.append("       Expected output rows if joined as-is: up to %d x %d = %d" % (n1, n2, n1 * n2))
        if match_rate_1 < 80:
            lines.append("\n[WARN] Low match rate (%.1f%%) -- %.0f%% of primary keys will be lost on INNER JOIN" % (
                match_rate_1, 100 - match_rate_1
            ))
        if len(orphans_1) > 0:
            lines.append("[NOTE] Primary orphan examples: %s" % list(list(orphans_1)[:5]))

        # Recommended join type
        lines.append("\n### Join Recommendation")
        if match_rate_1 >= 95 and match_rate_2 >= 95:
            lines.append("- Use INNER JOIN (both sides have >95%% match rate)")
        elif match_rate_1 < 80:
            lines.append("- Use LEFT JOIN to preserve all primary rows (%.0f%% would be lost with INNER JOIN)" % (100 - match_rate_1))
        else:
            lines.append("- Use LEFT JOIN (match rate %.1f%% -- check if orphans are expected)" % match_rate_1)

    content = "\n".join(lines)
    return content, results


# ---------------------------------------------------------------------------
# Tool 5: Generate validation rules
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def generate_validation_rules(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    strictness: str = "standard",
) -> Tuple[str, Dict]:
    """
    Tool: generate_validation_rules
    Description:
        Generates a catalog of VAL-### style validation rules based on data
        profiling. Covers: NOT NULL rules for required fields, range rules for
        numerics, format rules for strings (dates, IDs, emails), cardinality
        rules, uniqueness constraints, referential integrity hints, and
        business-logic rules inferred from column semantics.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        strictness (str, default='standard'): Rule generation mode:
            'lenient' (only clear rules), 'standard' (recommended),
            'strict' (all possible rules including inferred).

    LLM Guidance:
        Use after profile_columns_deep to generate the full validation catalog.
        These rules become the VAL-### checks in DATA_DOSSIER.md and monitoring plans.

    Returns:
        Tuple[str, Dict]: Validation rules narrative (content) and rules artifact (dict).
    """
    print("    * Tool: generate_validation_rules")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    n_rows = len(df)
    rules = []
    rule_counter = [0]

    def add_rule(column, rule_name, severity, check_expr, rationale, failure_action="block"):
        rule_counter[0] += 1
        rule = {
            "rule_id": "VAL-%03d" % rule_counter[0],
            "column": column,
            "rule": rule_name,
            "severity": severity,
            "check_expression": check_expr,
            "rationale": rationale,
            "failure_action": failure_action,
        }
        rules.append(rule)

    lines = ["## Validation Rules (VAL-### Catalog)\n"]
    lines.append("Mode: %s | Dataset: %d rows x %d cols\n" % (strictness, n_rows, len(df.columns)))

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        null_count = int(series.isnull().sum())
        null_pct = null_count / n_rows if n_rows > 0 else 0
        cardinality = int(series.nunique())
        col_lower = col.lower().replace("_", " ").replace("-", " ")

        # NOT NULL rule for seemingly required fields
        if null_count == 0:
            add_rule(
                column=col,
                rule_name="NOT NULL",
                severity="CRITICAL",
                check_expr="df['%s'].isnull().sum() == 0" % col,
                rationale="Column has no nulls in source data; new nulls indicate upstream issue",
                failure_action="block",
            )
        elif null_pct < 0.05:
            add_rule(
                column=col,
                rule_name="NULL RATE <= 5%%",
                severity="WARN",
                check_expr="df['%s'].isnull().sum() / len(df) <= 0.05" % col,
                rationale="Null rate is currently %.1f%%; >5%% may indicate upstream data loss" % (null_pct * 100),
                failure_action="warn",
            )

        # Uniqueness rule for likely key columns
        if cardinality == n_rows and null_count == 0:
            add_rule(
                column=col,
                rule_name="UNIQUE (Primary Key)",
                severity="CRITICAL",
                check_expr="df['%s'].nunique() == len(df) and df['%s'].isnull().sum() == 0" % (col, col),
                rationale="Column appears to be a primary key (100%% unique, 0 nulls)",
                failure_action="block",
            )

        # Numeric range rules
        if dtype in ("int64", "float64", "int32", "float32", "Int64", "Float64") or dtype.startswith(("int", "float")):
            num_data = pd.to_numeric(series, errors="coerce").dropna()
            if len(num_data) > 0:
                min_val = float(num_data.min())
                max_val = float(num_data.max())
                # Add range rule
                add_rule(
                    column=col,
                    rule_name="RANGE [%.3g, %.3g]" % (min_val, max_val),
                    severity="WARN",
                    check_expr="df['%s'].between(%.6g, %.6g).all()" % (col, min_val, max_val),
                    rationale="Values should stay within observed range [%.3g, %.3g]" % (min_val, max_val),
                    failure_action="warn",
                )
                # Non-negative rule for positive-expected columns
                positive_keywords = ["price", "amount", "cost", "revenue", "age", "count", "qty", "quantity", "salary", "income", "weight", "height"]
                if any(kw in col_lower for kw in positive_keywords) and min_val >= 0:
                    add_rule(
                        column=col,
                        rule_name="NON-NEGATIVE",
                        severity="ERROR",
                        check_expr="(df['%s'] >= 0).all()" % col,
                        rationale="Column '%s' should contain only non-negative values" % col,
                        failure_action="warn",
                    )

        # Categorical allowed-values rule (low cardinality)
        if (dtype == "object" or cardinality <= 20) and cardinality <= 15 and cardinality > 1:
            allowed_vals = sorted([str(v) for v in series.dropna().unique()])
            add_rule(
                column=col,
                rule_name="ALLOWED VALUES: %s" % str(allowed_vals[:8])[:60],
                severity="WARN" if strictness == "lenient" else "ERROR",
                check_expr="df['%s'].dropna().isin(%s).all()" % (col, repr(allowed_vals[:20])),
                rationale="Column has %d distinct values; new values may indicate data quality issues" % cardinality,
                failure_action="warn",
            )

        # Date column rules
        if "datetime" in dtype or any(kw in col_lower for kw in ["date", "created", "updated", "timestamp"]):
            if "datetime" in dtype:
                dt_data = series.dropna()
                if len(dt_data) > 0:
                    add_rule(
                        column=col,
                        rule_name="DATE RANGE",
                        severity="WARN",
                        check_expr="(pd.to_datetime(df['%s']) >= '%s').all()" % (col, str(dt_data.min())[:10]),
                        rationale="Dates should not be earlier than observed minimum",
                        failure_action="warn",
                    )
                    add_rule(
                        column=col,
                        rule_name="NO FUTURE DATES",
                        severity="WARN",
                        check_expr="(pd.to_datetime(df['%s']) <= pd.Timestamp.now()).all()" % col,
                        rationale="Dates should not be in the future unless intentional",
                        failure_action="warn",
                    )

        # String format rules
        if dtype == "object" and strictness in ("standard", "strict"):
            sample = series.dropna().astype(str).head(100)
            if len(sample) > 0:
                import re
                email_rate = sample.str.match(r'^[^@]+@[^@]+\.[^@]+$').mean()
                if email_rate > 0.7:
                    add_rule(
                        column=col,
                        rule_name="EMAIL FORMAT",
                        severity="WARN",
                        check_expr="df['%s'].dropna().str.match(r'^[^@]+@[^@]+\.[^@]+$').all()" % col,
                        rationale="%.0f%% of values appear to be emails; validate format" % (email_rate * 100),
                        failure_action="warn",
                    )

    # Cross-field rules (strict mode)
    if strictness == "strict":
        # Row count minimum
        add_rule(
            column="(dataset)",
            rule_name="MINIMUM ROW COUNT",
            severity="WARN",
            check_expr="len(df) >= %d" % max(1, int(n_rows * 0.9)),
            rationale="Dataset should have at least %.0f%% of baseline row count (%d)" % (90, int(n_rows * 0.9)),
            failure_action="warn",
        )

    lines.append("### Validation Rules Generated: %d\n" % len(rules))
    lines.append("| Rule ID | Column | Rule | Severity |")
    lines.append("|---------|--------|------|----------|")
    for rule in rules:
        lines.append("| %s | `%s` | %s | %s |" % (
            rule["rule_id"], rule["column"], rule["rule"][:60], rule["severity"]
        ))

    lines.append("\n### Rules by Severity")
    severity_counts = {}
    for rule in rules:
        severity_counts.setdefault(rule["severity"], 0)
        severity_counts[rule["severity"]] += 1
    for sev, cnt in sorted(severity_counts.items()):
        lines.append("- %s: %d rules" % (sev, cnt))

    content = "\n".join(lines)
    return content, {"rules": rules, "total": len(rules), "by_severity": severity_counts}


# ---------------------------------------------------------------------------
# Tool 6: Assess ML readiness
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def assess_ml_readiness(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    target_column: str = "",
    prediction_type: str = "auto",
) -> Tuple[str, Dict]:
    """
    Tool: assess_ml_readiness
    Description:
        Comprehensive ML readiness assessment covering: target variable analysis,
        leakage risk detection (temporal and target leakage), feature quality
        (missing rates, cardinality, correlation with target), encoding
        recommendations per feature type, train/test split considerations,
        class imbalance, and data volume assessment.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        target_column (str): Name of the target/label column. Leave empty
            for general feature assessment only.
        prediction_type (str, default='auto'): 'classification', 'regression',
            or 'auto' to infer from target column type.

    LLM Guidance:
        Call before any feature engineering or model training. Pay special
        attention to leakage risks (temporal and proxy leakage). Set
        target_column for complete assessment.

    Returns:
        Tuple[str, Dict]: ML readiness report (content) and assessment artifact (dict).
    """
    print("    * Tool: assess_ml_readiness")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    n_rows, n_cols = df.shape
    results = {}
    lines = ["## ML Readiness Assessment\n"]
    lines.append("Dataset: %d rows x %d features\n" % (n_rows, n_cols))

    warnings = []
    failures = []

    # --- Target analysis ---
    if target_column and target_column in df.columns:
        target = df[target_column]
        target_null = int(target.isnull().sum())
        target_card = int(target.nunique())

        # Infer prediction type
        if prediction_type == "auto":
            if target.dtype in (object, "category") or target_card <= 10:
                prediction_type = "classification"
            else:
                prediction_type = "regression"

        results["target"] = {
            "column": target_column,
            "prediction_type": prediction_type,
            "null_count": target_null,
            "cardinality": target_card,
        }
        lines.append("### Target Variable: `%s` (%s)" % (target_column, prediction_type.upper()))

        if target_null > 0:
            failures.append("Target column `%s` has %d nulls -- must be resolved before training" % (target_column, target_null))
            lines.append("[FAIL] %d null values in target -- rows must be dropped or imputed" % target_null)

        if prediction_type == "classification":
            vc = target.value_counts(normalize=True)
            class_dist = {str(k): round(float(v * 100), 2) for k, v in vc.items()}
            imbalance = round(float(vc.max() / vc.min()), 2) if len(vc) > 1 and vc.min() > 0 else None
            results["target"]["class_distribution"] = class_dist
            results["target"]["imbalance_ratio"] = imbalance

            lines.append("Class distribution:")
            for cls, pct in list(class_dist.items())[:10]:
                lines.append("  - `%s`: %.1f%%" % (cls, pct))
            if imbalance and imbalance > 5:
                warnings.append("Class imbalance ratio %.1f:1 -- consider SMOTE, class weights, or stratified splits" % imbalance)
                lines.append("[WARN] Imbalance ratio %.1f:1 -- use stratified sampling" % imbalance)
        else:
            num_target = pd.to_numeric(target, errors="coerce").dropna()
            if len(num_target) > 0:
                skew = round(float(num_target.skew()), 3)
                results["target"]["skewness"] = skew
                lines.append("Distribution: mean=%.3g, std=%.3g, skewness=%.3f" % (
                    float(num_target.mean()), float(num_target.std()), skew
                ))
                if abs(skew) > 1:
                    warnings.append("Target `%s` is heavily skewed (%.2f) -- consider log transform" % (target_column, skew))
    else:
        prediction_type = prediction_type if prediction_type != "auto" else "unknown"
        lines.append("_No target column specified. Running general feature assessment._\n")

    # --- Feature quality ---
    feature_cols = [c for c in df.columns if c != target_column]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

    lines.append("\n### Feature Quality")
    lines.append("| Feature | Type | Missing % | Cardinality | Recommendation |")
    lines.append("|---------|------|-----------|-------------|----------------|")

    feature_notes = []
    leakage_risks = []
    encoding_recommendations = []

    for col in feature_cols[:25]:
        series = df[col]
        dtype = str(series.dtype)
        null_pct = round(series.isnull().sum() / n_rows * 100, 2) if n_rows > 0 else 0
        cardinality = int(series.nunique())

        recommendation = ""
        if null_pct > 50:
            recommendation = "Drop (>50%% missing)"
            warnings.append("Feature `%s`: %.0f%% missing -- consider dropping" % (col, null_pct))
        elif null_pct > 20:
            recommendation = "Impute (>20%% missing)"
        elif null_pct > 0:
            recommendation = "Impute (<20%% missing)"
        elif dtype == "object" and cardinality > 100:
            recommendation = "Hash/embed (high cardinality text)"
            encoding_recommendations.append("`%s`: Use target encoding or hashing (cardinality=%d)" % (col, cardinality))
        elif dtype == "object" and cardinality > 20:
            recommendation = "Target encode or ordinal"
            encoding_recommendations.append("`%s`: Target encoding recommended (cardinality=%d)" % (col, cardinality))
        elif dtype == "object":
            recommendation = "OneHot encode (%d cats)" % cardinality
            encoding_recommendations.append("`%s`: One-hot encoding (cardinality=%d)" % (col, cardinality))
        elif "datetime" in dtype:
            recommendation = "Extract year/month/day features"
            feature_notes.append("`%s`: Extract datetime components as features" % col)
        else:
            recommendation = "Use as-is or scale"

        # Leakage check
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["target", "label", "outcome", "churn", "fraud", "predict"]) and col != target_column:
            leakage_risks.append("`%s`: Name suggests potential target leakage" % col)
        if target_column and "datetime" not in dtype and dtype == "object":
            # Check correlation-based proxy leakage
            pass  # LLM should reason about this

        lines.append("| `%s` | %s | %.1f%% | %d | %s |" % (col, dtype, null_pct, cardinality, recommendation[:50]))

    results["feature_notes"] = feature_notes
    results["leakage_risks"] = leakage_risks
    results["encoding_recommendations"] = encoding_recommendations

    # --- Volume assessment ---
    min_rows_for_ml = 100
    recommended_rows = 1000
    lines.append("\n### Data Volume Assessment")
    if n_rows < min_rows_for_ml:
        failures.append("Only %d rows -- likely insufficient for reliable ML (minimum ~100 recommended)" % n_rows)
        lines.append("[FAIL] %d rows is likely insufficient for ML" % n_rows)
    elif n_rows < recommended_rows:
        warnings.append("Only %d rows -- use cross-validation instead of train/test split" % n_rows)
        lines.append("[WARN] %d rows -- use cross-validation, not simple train/test split" % n_rows)
    else:
        lines.append("[OK] %d rows -- adequate for ML" % n_rows)

    # --- Leakage risks summary ---
    if leakage_risks:
        lines.append("\n### Leakage Risk Flags")
        for risk in leakage_risks:
            lines.append("- [RISK] %s" % risk)

    # --- Verdict ---
    if failures:
        verdict = "NOT_READY"
        verdict_label = "[NOT READY]"
    elif len(warnings) > 3:
        verdict = "NEEDS_WORK"
        verdict_label = "[NEEDS WORK]"
    elif warnings:
        verdict = "READY_WITH_CAVEATS"
        verdict_label = "[READY WITH CAVEATS]"
    else:
        verdict = "READY"
        verdict_label = "[READY]"

    results["verdict"] = verdict
    results["warnings"] = warnings
    results["failures"] = failures

    lines.append("\n### ML Readiness Verdict: %s **%s**" % (verdict_label, verdict))
    lines.append("- Failures: %d | Warnings: %d" % (len(failures), len(warnings)))

    content = "\n".join(lines)
    return content, results


# ---------------------------------------------------------------------------
# Tool 7: Plan cleaning steps
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def plan_cleaning_steps(
    data_raw: Annotated[dict, InjectedState("data_raw")],
    objective: str = "analytics",
) -> Tuple[str, Dict]:
    """
    Tool: plan_cleaning_steps
    Description:
        Generates an ordered, reversible cleaning and canonicalization plan
        based on data profiling. Steps are prioritized by impact: critical
        issues (type fixes, sentinel replacement) first, then standardization
        (casing, whitespace), then enrichment (date parsing, feature derivation).
        Each step includes: operation, affected columns, rationale, Python code
        hint, risk level, and reversibility assessment.

    Parameters:
        data_raw (dict): Raw data injected from agent state.
        objective (str, default='analytics'): Target objective affecting
            step priorities: 'analytics', 'etl_load', 'ml_training', 'dedupe'.

    LLM Guidance:
        Call after profile_columns_deep and detect_edge_cases. The plan
        generated here should be the basis for any data cleaning pipeline.

    Returns:
        Tuple[str, Dict]: Cleaning plan narrative (content) and plan artifact (dict).
    """
    print("    * Tool: plan_cleaning_steps")
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data_raw)
    n_rows = len(df)
    steps = []
    step_num = [0]

    def add_step(operation, columns, rationale, code_hint, risk="low", reversible=True, priority=5):
        step_num[0] += 1
        steps.append({
            "step": step_num[0],
            "operation": operation,
            "columns": columns if isinstance(columns, list) else [columns],
            "rationale": rationale,
            "code_hint": code_hint,
            "risk": risk,
            "reversible": reversible,
            "priority": priority,
        })

    lines = ["## Cleaning & Canonicalization Plan\n"]
    lines.append("Objective: %s | Dataset: %d rows x %d cols\n" % (objective, n_rows, len(df.columns)))

    # --- Phase 1: Structural fixes (CRITICAL first) ---
    lines.append("### Phase 1: Critical Structural Fixes\n")

    # Type coercion fixes
    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col]
        numeric_rate = pd.to_numeric(series, errors="coerce").notna().sum() / max(series.notna().sum(), 1)
        if numeric_rate > 0.9:
            add_step(
                operation="Cast to numeric",
                columns=col,
                rationale="`%s` is stored as string but %.0f%% of values are numeric" % (col, numeric_rate * 100),
                code_hint="df['%s'] = pd.to_numeric(df['%s'], errors='coerce')" % (col, col),
                risk="medium",
                reversible=True,
                priority=1,
            )
        else:
            try:
                dt_rate = pd.to_datetime(series, errors="coerce").notna().sum() / max(series.notna().sum(), 1)
                if dt_rate > 0.85:
                    add_step(
                        operation="Parse as datetime",
                        columns=col,
                        rationale="`%s` appears to contain datetime strings (%.0f%% parseable)" % (col, dt_rate * 100),
                        code_hint="df['%s'] = pd.to_datetime(df['%s'], errors='coerce')" % (col, col),
                        risk="medium",
                        reversible=True,
                        priority=1,
                    )
            except Exception:
                pass

    # Float-stored integers
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        non_null = df[col].dropna()
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            add_step(
                operation="Convert to nullable integer",
                columns=col,
                rationale="`%s` contains only whole numbers stored as float (likely NaN coercion artifact)" % col,
                code_hint="df['%s'] = df['%s'].astype(pd.Int64Dtype())" % (col, col),
                risk="low",
                reversible=True,
                priority=2,
            )

    # Duplicate rows
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        add_step(
            operation="Remove duplicate rows",
            columns="(all)",
            rationale="%d exact duplicate rows found" % n_dupes,
            code_hint="df = df.drop_duplicates().reset_index(drop=True)",
            risk="medium",
            reversible=False,
            priority=1,
        )

    # --- Phase 2: Null / sentinel handling ---
    lines.append("### Phase 2: Null & Sentinel Value Handling\n")

    sentinel_patterns = {
        str: ["unknown", "n/a", "na", "none", "null", "nan", "tbd", "missing", "undefined",
              "9999", "-1", "-999"],
        int: [-1, -999, -9999, 9999, 99999],
    }

    for col in df.select_dtypes(include=["object"]).columns:
        str_series = df[col].dropna().astype(str).str.lower().str.strip()
        found = [s for s in sentinel_patterns[str] if str_series.isin([s]).any()]
        if found:
            add_step(
                operation="Replace sentinel nulls",
                columns=col,
                rationale="Sentinel values %s found in `%s`" % (found[:3], col),
                code_hint="df['%s'] = df['%s'].replace(%s, np.nan)" % (col, col, repr(found[:5])),
                risk="medium",
                reversible=True,
                priority=2,
            )

    # Columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        for col in missing_cols[:8]:
            null_pct = round(df[col].isnull().sum() / n_rows * 100, 1)
            dtype = str(df[col].dtype)
            if null_pct > 50:
                add_step(
                    operation="Drop column (>50%% missing)",
                    columns=col,
                    rationale="`%s` is %.1f%% null -- likely not useful" % (col, null_pct),
                    code_hint="df = df.drop(columns=['%s'])" % col,
                    risk="high",
                    reversible=False,
                    priority=3,
                )
            elif "int" in dtype or "float" in dtype:
                add_step(
                    operation="Impute missing numeric values",
                    columns=col,
                    rationale="`%s` has %.1f%% nulls -- impute with median for robustness" % (col, null_pct),
                    code_hint="df['%s'] = df['%s'].fillna(df['%s'].median())" % (col, col, col),
                    risk="medium",
                    reversible=True,
                    priority=4,
                )
            else:
                add_step(
                    operation="Impute missing categorical values",
                    columns=col,
                    rationale="`%s` has %.1f%% nulls -- impute with mode or 'Unknown'" % (col, null_pct),
                    code_hint="df['%s'] = df['%s'].fillna('Unknown')" % (col, col),
                    risk="medium",
                    reversible=True,
                    priority=4,
                )

    # --- Phase 3: Standardization ---
    lines.append("### Phase 3: Standardization & Canonicalization\n")

    for col in df.select_dtypes(include=["object"]).columns:
        str_data = df[col].dropna().astype(str)
        # Whitespace
        if str_data.str.match(r'^\s|\s$').any():
            add_step(
                operation="Strip whitespace",
                columns=col,
                rationale="Values with leading/trailing whitespace found in `%s`" % col,
                code_hint="df['%s'] = df['%s'].str.strip()" % (col, col),
                risk="low",
                reversible=True,
                priority=5,
            )
        # Case normalization for low-cardinality
        if df[col].nunique() <= 20 and str_data.str.contains(r'[A-Z]').any() and str_data.str.contains(r'[a-z]').any():
            add_step(
                operation="Normalize case",
                columns=col,
                rationale="Mixed case detected in low-cardinality column `%s` -- may cause groupby issues" % col,
                code_hint="df['%s'] = df['%s'].str.lower().str.strip()" % (col, col),
                risk="low",
                reversible=True,
                priority=5,
            )

    # Sort by priority
    steps.sort(key=lambda x: x["priority"])
    for i, step in enumerate(steps, 1):
        step["step"] = i

    # Format output
    lines.append("\n### Complete Ordered Cleaning Plan\n")
    lines.append("| Step | Operation | Columns | Risk | Reversible |")
    lines.append("|------|-----------|---------|------|------------|")
    for step in steps:
        cols_str = ", ".join("`%s`" % c for c in step["columns"][:3])
        lines.append("| %d | %s | %s | %s | %s |" % (
            step["step"], step["operation"], cols_str,
            step["risk"].upper(), "Yes" if step["reversible"] else "No"
        ))

    lines.append("\n### Detailed Steps\n")
    for step in steps:
        lines.append("**Step %d: %s** (`%s`)" % (step["step"], step["operation"], str(step["columns"])[:60]))
        lines.append("- Rationale: %s" % step["rationale"])
        lines.append("- Code: `%s`" % step["code_hint"][:120])
        lines.append("- Risk: %s | Reversible: %s" % (step["risk"].upper(), "Yes" if step["reversible"] else "No"))
        lines.append("")

    content = "\n".join(lines)
    return content, {"steps": steps, "total": len(steps), "objective": objective}


# ---------------------------------------------------------------------------
# Tool 8: Generate DATA_DOSSIER.md
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def generate_data_dossier(
    dataset_name: str,
    inventory_summary: str = "",
    schema_summary: str = "",
    dictionary_summary: str = "",
    profiles_summary: str = "",
    relationships_summary: str = "",
    edge_cases_summary: str = "",
    validation_summary: str = "",
    cleaning_summary: str = "",
    ml_summary: str = "",
    open_questions: str = "",
) -> Tuple[str, Dict]:
    """
    Tool: generate_data_dossier
    Description:
        Compiles the authoritative DATA_DOSSIER.md from analysis findings.
        Takes free-form text summaries for each section (as gathered from
        previous tool calls and LLM analysis) and formats them into the
        standard DDEA dossier structure. This should be the LAST tool
        called after all profiling and analysis is complete.

    Parameters:
        dataset_name (str): Name of the dataset.
        inventory_summary (str): Summary from schema/inventory analysis.
        schema_summary (str): Column types and type inference findings.
        dictionary_summary (str): Field-by-field semantic descriptions.
        profiles_summary (str): Statistical profile findings.
        relationships_summary (str): Join keys and relationship analysis.
        edge_cases_summary (str): Edge cases and nuances found.
        validation_summary (str): Validation rules generated (VAL-### catalog).
        cleaning_summary (str): Ordered cleaning plan.
        ml_summary (str): ML readiness assessment.
        open_questions (str): Comma-separated list of open questions.

    LLM Guidance:
        Call this LAST after running all other profiling tools. Synthesize
        the key findings from each tool call into the summary parameters.
        The output is the DATA_DOSSIER.md that other agents will use as
        their authoritative reference.

    Returns:
        Tuple[str, Dict]: DATA_DOSSIER.md content (content) and dossier artifact (dict).
    """
    print("    * Tool: generate_data_dossier")
    from datetime import datetime

    open_q_list = [q.strip() for q in open_questions.split(",") if q.strip()] if open_questions else []

    sections = [
        "# DATA_DOSSIER: %s" % dataset_name,
        "",
        "> **Generated by DataDomainExpertAgent (DDEA)**  ",
        "> Last updated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
        "---",
        "",
        "## Table of Contents",
        "1. Dataset Inventory",
        "2. Schema & Types",
        "3. Data Dictionary",
        "4. Column Profiles",
        "5. Relationships & Join Map",
        "6. Edge Cases & Nuances",
        "7. Validation Rules",
        "8. Cleaning & Canonicalization Plan",
        "9. ML Readiness Notes",
        "10. Open Questions & Uncertainties",
        "",
        "---",
        "",
        "## 1. Dataset Inventory",
        "",
        inventory_summary or "_Run profile_dataset_schema to populate._",
        "",
        "---",
        "",
        "## 2. Schema & Types",
        "",
        schema_summary or "_Run profile_dataset_schema to populate._",
        "",
        "---",
        "",
        "## 3. Data Dictionary",
        "",
        dictionary_summary or "_LLM analysis of field semantics and business context._",
        "",
        "---",
        "",
        "## 4. Column Profiles",
        "",
        profiles_summary or "_Run profile_columns_deep to populate._",
        "",
        "---",
        "",
        "## 5. Relationships & Join Map",
        "",
        relationships_summary or "_Run analyze_join_keys to populate._",
        "",
        "---",
        "",
        "## 6. Edge Cases & Nuances",
        "",
        edge_cases_summary or "_Run detect_edge_cases to populate._",
        "",
        "---",
        "",
        "## 7. Validation Rules",
        "",
        validation_summary or "_Run generate_validation_rules to populate._",
        "",
        "---",
        "",
        "## 8. Cleaning & Canonicalization Plan",
        "",
        cleaning_summary or "_Run plan_cleaning_steps to populate._",
        "",
        "---",
        "",
        "## 9. ML Readiness Notes",
        "",
        ml_summary or "_Run assess_ml_readiness to populate._",
        "",
        "---",
        "",
        "## 10. Open Questions & Uncertainties",
        "",
    ]

    if open_q_list:
        for q in open_q_list:
            sections.append("- [ ] %s" % q)
    else:
        sections.append("_No open questions recorded._")

    sections.extend([
        "",
        "---",
        "",
        "## Change Log",
        "",
        "| Date | Change |",
        "|------|--------|",
        "| %s | Initial dossier created by DDEA |" % datetime.now().strftime("%Y-%m-%d"),
    ])

    content = "\n".join(sections)
    artifact = {
        "data_dossier": content,
        "dataset_name": dataset_name,
    }
    return content, artifact
