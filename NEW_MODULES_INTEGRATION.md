# New Modules Integration Guide

**For:** Research Intelligence App integration agent
**Repo:** `ai-data-science-team` Python module
**Date:** 2026-02-26

This document describes three new agents added to the `ai-data-science-team` module that are ready to be integrated into the research intelligence app.

---

## What Was Added

### 1. `DataDomainExpertAgent` (DDEA)
**Import path:** `from ai_data_science_team import DataDomainExpertAgent`
**File:** `ai_data_science_team/agents/data_analysis/data_domain_expert.py`

The authoritative dataset profiler. Exhaustively analyzes any dataset and produces a `DATA_DOSSIER.md` — a structured reference document covering schema, column statistics, edge cases, validation rules, join keys, ML readiness, and a cleaning plan. Also supports 11 structured A2A (Agent-to-Agent) intents for precise programmatic queries.

**Best for:** Onboarding a new dataset, auditing data quality, answering "what is in this data?", generating documentation.

---

### 2. `DataStorytellerAgent` (DIST)
**Import path:** `from ai_data_science_team import DataStorytellerAgent`
**File:** `ai_data_science_team/agents/data_analysis/data_storyteller.py`

Transforms datasets into decision-grade insights and professional narratives. Computes KPIs, detects patterns, compares segments, validates analysis integrity, and produces a full report with executive summary + technical methodology — for both business and technical audiences.

**Best for:** "What story does this data tell?", KPI reporting, executive summaries, finding hidden patterns.

---

### 3. `DataInsightsTeam` (Multi-Agent Pipeline)
**Import path:** `from ai_data_science_team import DataInsightsTeam`
**File:** `ai_data_science_team/multiagents/data_insights_team.py`

A two-stage sequential pipeline: **DDEA runs first** (profiles the dataset, generates dossier) → **DIST runs second** (receives the dossier as data context, generates narrative report). DIST never guesses at data semantics — it works from DDEA's verified findings.

**Best for:** Full end-to-end analysis in one call. The recommended default for comprehensive dataset analysis.

---

## Installation / Dependencies

No new dependencies. These agents use the same LangChain + LangGraph stack already in the module. Requires:

```
langchain
langgraph >= 0.2.74
pandas
numpy
```

---

## Quick Start

```python
import pandas as pd
from ai_data_science_team import create_llm, DataDomainExpertAgent, DataStorytellerAgent, DataInsightsTeam

llm = create_llm()  # reads from .env (OPENAI_API_KEY, etc.)
df = pd.read_csv("data/my_dataset.csv")
```

### Option A: Profile only (DDEA)

```python
ddea = DataDomainExpertAgent(model=llm)

ddea.invoke_agent(
    user_instructions="Profile this dataset completely and generate a DATA_DOSSIER.md",
    data_raw=df,
)

dossier_text = ddea.get_dossier()           # str — full DATA_DOSSIER.md markdown
dossier_md   = ddea.get_dossier(markdown=True)  # IPython Markdown object
artifacts    = ddea.get_artifacts()         # dict — structured findings from all tools
tool_calls   = ddea.get_tool_calls()        # list[str] — which tools were called
ai_message   = ddea.get_ai_message()        # str — final LLM response
```

### Option B: Storytelling with context (DIST)

```python
dist = DataStorytellerAgent(model=llm)

dist.invoke_agent(
    user_instructions="Tell me the story of this customer data. What patterns matter?",
    data_raw=df,
    data_context=dossier_text,  # Optional but recommended: pass DDEA dossier
)

story      = dist.get_story()              # str — full narrative report markdown
story_md   = dist.get_story(markdown=True) # IPython Markdown object
tool_calls = dist.get_tool_calls()         # list[str]
artifacts  = dist.get_artifacts()          # dict — last tool artifact
```

### Option C: Full pipeline (DataInsightsTeam) — recommended

```python
team = DataInsightsTeam(
    ddea_agent=DataDomainExpertAgent(model=llm),
    dist_agent=DataStorytellerAgent(model=llm),
)

team.invoke_agent(
    user_instructions="What are the key insights and business recommendations?",
    data_raw=df,
    # Optional: separate instructions for the profiling stage
    ddea_instructions="Profile completely. Flag any PII. Generate DATA_DOSSIER.md.",
)

dossier       = team.get_dossier()           # str — DDEA's DATA_DOSSIER.md
story         = team.get_story()             # str — DIST's narrative report
ddea_artifacts = team.get_ddea_artifacts()  # dict — structured DDEA findings
ddea_tools    = team.get_ddea_tool_calls()  # list[str]
dist_tools    = team.get_dist_tool_calls()  # list[str]
```

---

## Full API Reference

### `DataDomainExpertAgent`

```python
DataDomainExpertAgent(
    model,                          # LangChain LLM (required)
    create_react_agent_kwargs={},   # extra kwargs for create_react_agent
    invoke_react_agent_kwargs={},   # extra kwargs for agent invocation
    checkpointer=None,              # LangGraph checkpointer for state persistence
)
```

```python
# Synchronous
agent.invoke_agent(
    user_instructions: str,         # What to analyze / what intent to fulfill
    data_raw: pd.DataFrame,         # Primary dataset
    secondary_data_raw: pd.DataFrame = None,  # Secondary dataset for join analysis
    **kwargs,                       # Passed to graph.invoke()
)

# Async
await agent.ainvoke_agent(user_instructions, data_raw, secondary_data_raw, **kwargs)
```

**Getters:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_dossier(markdown=False)` | `str` or `Markdown` | Full DATA_DOSSIER.md content |
| `get_ai_message(markdown=False)` | `str` or `Markdown` | Final LLM response |
| `get_internal_messages(markdown=False)` | `list` or `Markdown` | All tool calls + LLM responses |
| `get_artifacts(as_dataframe=False)` | `dict` or `DataFrame` | Structured findings from all tools |
| `get_tool_calls()` | `list[str]` | Names of tools called |
| `get_response()` | `dict` | Full raw state dict |
| `show()` | — | Display agent mermaid diagram |

**DDEA Artifacts dict keys** (from `get_artifacts()`):

```python
{
    "row_count": int,
    "col_count": int,
    "columns": [...],           # from profile_dataset_schema
    "summary": {...},
    "column_profiles": {...},   # from profile_columns_deep
    "edge_cases": [...],        # from detect_edge_cases
    "rules": [...],             # from generate_validation_rules
    "steps": [...],             # from plan_cleaning_steps
    "verdict": str,             # from assess_ml_readiness ("READY", "NOT_READY", etc.)
    "leakage_risks": [...],
    "data_dossier": str,        # from generate_data_dossier (full markdown)
    "dataset_name": str,
}
```

**Supported `user_instructions` intents:**

```
"Profile this dataset completely and generate a DATA_DOSSIER.md"
"Intent: schema_and_types. What are all the column types and confidence levels?"
"Intent: edge_cases_and_nuances. What sentinel nulls and encoding issues exist?"
"Intent: validation_rules. Generate a complete VAL-### rule catalog."
"Intent: ml_readiness_report. Target column: churn. Is this ready for ML?"
"Intent: cleaning_and_canonicalization_plan. Objective: ml_training."
"Analyze the join between this dataset and the secondary dataset on order_id."
"Intent: workflow_verification. [paste pipeline code here]"
```

---

### `DataStorytellerAgent`

```python
DataStorytellerAgent(
    model,                          # LangChain LLM (required)
    data_context=None,              # Default context (dossier/dict/schema) for all calls
    create_react_agent_kwargs={},
    invoke_react_agent_kwargs={},
    checkpointer=None,
)
```

```python
# Synchronous
agent.invoke_agent(
    user_instructions: str,         # What story to tell / question to answer
    data_raw: pd.DataFrame,
    data_context=None,              # Override default context for this call
    **kwargs,
)

# Async
await agent.ainvoke_agent(user_instructions, data_raw, data_context, **kwargs)
```

**Getters:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_story(markdown=False)` | `str` or `Markdown` | Complete narrative report |
| `get_ai_message(markdown=False)` | `str` or `Markdown` | Same as get_story |
| `get_internal_messages(markdown=False)` | `list` or `Markdown` | All tool calls + LLM responses |
| `get_artifacts(as_dataframe=False)` | `dict` or `DataFrame` | Last tool artifact |
| `get_tool_calls()` | `list[str]` | Names of tools called |
| `get_response()` | `dict` | Full raw state dict |

**`data_context` parameter:** Accepts a `str`, `dict`, or `None`. If a dict, it is JSON-serialized. The content is prepended to `user_instructions` under a `## Data Context` heading so the LLM sees schema/semantic information before analyzing. Pass DDEA's `get_dossier()` output here for best results.

---

### `DataInsightsTeam`

```python
DataInsightsTeam(
    ddea_agent: DataDomainExpertAgent,  # Required
    dist_agent: DataStorytellerAgent,   # Required
    checkpointer=None,
)
```

```python
# Synchronous
team.invoke_agent(
    user_instructions: str,                 # Story question for DIST
    data_raw: pd.DataFrame or dict,
    secondary_data_raw: pd.DataFrame = None,  # For cross-dataset join analysis
    ddea_instructions: str = None,          # Optional separate instructions for DDEA
    **kwargs,
)

# Async
await team.ainvoke_agent(...)
```

**Getters:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_dossier(markdown=False)` | `str` or `Markdown` | DDEA's DATA_DOSSIER.md |
| `get_story(markdown=False)` | `str` or `Markdown` | DIST's narrative report |
| `get_ai_message(markdown=False)` | `str` or `Markdown` | Same as get_story |
| `get_ddea_artifacts()` | `dict` | Structured DDEA profiling findings |
| `get_ddea_tool_calls()` | `list[str]` | DDEA tools called |
| `get_dist_tool_calls()` | `list[str]` | DIST tools called |
| `get_response()` | `dict` | Full pipeline state dict |

---

## Available Tools (importable individually)

### Domain Expert Tools (`tools/data_analysis/domain_expert_tools.py`)

| Tool | Description |
|------|-------------|
| `profile_dataset_schema` | Schema inference: types, confidence, cardinality, key candidates |
| `profile_columns_deep` | Per-column stats, missing %, string patterns, PII detection |
| `detect_edge_cases` | Sentinel nulls, mixed booleans, whitespace, multi-value delimiters |
| `analyze_join_keys` | Key uniqueness, cross-dataset match rate, multiplicity (1:1/1:N/M:N) |
| `generate_validation_rules` | VAL-### catalog (NOT NULL, RANGE, UNIQUE, FORMAT, DATE RANGE) |
| `assess_ml_readiness` | Target analysis, leakage flags, feature quality, encoding recommendations |
| `plan_cleaning_steps` | Ordered reversible cleaning plan (structural → null handling → standardize) |
| `generate_data_dossier` | Compiles full DATA_DOSSIER.md from text summaries |

### Storytelling Tools (`tools/data_analysis/storytelling_tools.py`)

| Tool | Description |
|------|-------------|
| `profile_data_for_story` | Shape, types, missing, distributions, datetime coverage |
| `compute_dataset_kpis` | Structural, numeric, categorical, and temporal KPIs |
| `detect_data_patterns` | IQR outliers, skewness/kurtosis, correlations, time trends |
| `validate_analysis_integrity` | PASS/WARN/FAIL checks: dupes, nulls, type mismatches |
| `compute_group_comparisons` | Per-group stats + Simpson's paradox detection |
| `build_story_spine` | Formats 5-part narrative (Context → Tension → Insight → Implication → Action) |

### Utility Functions

```python
# DATA_DOSSIER.md formatting
from ai_data_science_team.utils.data_dossier import (
    format_data_dossier,           # Build full 12-section dossier from structured data
    format_validation_rule_table,  # Format VAL-### rules as markdown table
    format_edge_case_summary,      # Format edge case findings as summary table
    format_a2a_response,           # Format structured A2A response envelope
)

# Narrative formatting
from ai_data_science_team.utils.storytelling import (
    format_story_spine,            # 5-part narrative structure
    format_kpi_table,              # KPI table with definitions and caveats
    format_insights_report,        # Full insights report with all sections
    format_metrics_catalog,        # Metrics catalog table
    format_agent_story_output,     # Format raw agent response dict as story
)
```

---

## Cross-Dataset Join Analysis

`DataDomainExpertAgent` supports analyzing joins between two datasets via the `secondary_data_raw` parameter. The `analyze_join_keys` tool computes match rate, join multiplicity, orphan records, and recommends the correct join type.

```python
ddea = DataDomainExpertAgent(model=llm)

ddea.invoke_agent(
    user_instructions="""
    Analyze the join between orders (primary) and customers (secondary) on customer_id.
    Report the match rate, multiplicity, and orphan counts. Recommend the join type.
    """,
    data_raw=orders_df,
    secondary_data_raw=customers_df,
)

artifacts = ddea.get_artifacts()
join_info = artifacts.get("cross_dataset_analysis", {})
# {
#   "join_key": "customer_id",
#   "match_count": 847,
#   "match_rate_primary": 94.2,
#   "match_rate_secondary": 99.1,
#   "orphans_in_primary": 52,
#   "orphans_in_secondary": 8,
#   "multiplicity": "1:N (left table is the 1 side)",
#   "max_mult_primary": 1,
#   "max_mult_secondary": 23,
# }
```

---

## A2A Interface

`DataDomainExpertAgent` implements a structured Agent-to-Agent (A2A) interface. Other agents or pipeline nodes can query it with specific intents and receive structured, predictable responses.

**Response format:**
```
## DDEA Response: [INTENT]

**Request ID:** req-001
**Dataset:** customer_data
**Confidence:** HIGH

### Summary
[1-3 sentence direct answer]

### Details
[Structured findings]

### Evidence
- [What was checked]

### Recommended Actions
1. [Actionable next step]

### Risks & Pitfalls
- [RISK] [What could go wrong]

### Open Questions
- [ ] [What needs further investigation]
```

**Available intents:**

| Intent | What it returns |
|--------|----------------|
| `dataset_overview` | Inventory, shape, format, row/column counts |
| `schema_and_types` | All columns with inferred types, confidence, nullable status |
| `column_profile` | Per-column distributions, patterns, anomalies |
| `edge_cases_and_nuances` | Sentinel nulls, encoding issues, mixed types, PII |
| `relationships_and_joins` | Key quality, FK/PK, join recommendations, multiplicity |
| `validation_rules` | Complete VAL-### rule catalog |
| `cleaning_and_canonicalization_plan` | Ordered cleaning steps |
| `ml_readiness_report` | Target analysis, leakage risks, feature encoding |
| `workflow_verification` | Audit a pipeline for correctness and edge cases |
| `workflow_monitoring_plan` | Design automated quality checks and monitors |
| `generate_or_update_dossier` | Compile or refresh the full DATA_DOSSIER.md |

---

## Important Notes for Integration

### 1. Python Version
These agents require Python >= 3.9. Python 3.10+ is recommended. There is a pre-existing bug in `templates/agent_templates.py` that affects the **code-generating** agents (DataCleaningAgent, etc.) on Python 3.9 — the new ReAct agents (DDEA, DIST) are **not affected** by this bug.

### 2. Model Recommendations
DDEA and DIST perform significantly better with capable models. For a research intelligence app, use:
- **Best quality:** `gpt-4o`, `claude-opus-4-6` (claude-opus-4-5)
- **Good balance:** `gpt-4o-mini`, `claude-sonnet-4-6`
- **Avoid for DDEA/DIST:** Small local models (llama2, mistral-7b) — they struggle with multi-tool reasoning

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# Anthropic
llm = ChatAnthropic(model="claude-opus-4-6", api_key="...")

# OpenAI
llm = ChatOpenAI(model="gpt-4o", api_key="...")
```

### 3. No Emoji in Tool Outputs
Tool output strings are ASCII-only (Windows cp1252 compatibility). Status indicators use `[OK]`, `[WARN]`, `[FAIL]`, `[RISK]`, `[HIGH]`, `[LOW]`. Do not add emoji to tool output strings if extending these tools.

### 4. DataFrame Conversion
When invoking agents directly (not via `invoke_agent`), pass `data_raw` as a dict:
```python
# Via invoke_agent — handles conversion automatically
agent.invoke_agent(data_raw=df)

# Via compiled graph directly — must convert manually
graph.invoke({"data_raw": df.to_dict(), ...})
```

### 5. Async Support
All three agents support async invocation:
```python
await ddea.ainvoke_agent(user_instructions="...", data_raw=df)
await dist.ainvoke_agent(user_instructions="...", data_raw=df)
await team.ainvoke_agent(user_instructions="...", data_raw=df)
```

### 6. State Persistence (Checkpointing)
For stateful workflows or multi-turn conversations:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
ddea = DataDomainExpertAgent(model=llm, checkpointer=checkpointer)

# With thread_id for multi-turn
ddea.invoke_agent(
    user_instructions="Profile this dataset",
    data_raw=df,
    config={"configurable": {"thread_id": "session-123"}},
)
```

---

## Registry Discovery

All new agents are registered and discoverable:

```python
from ai_data_science_team import get_agent_registry

registry = get_agent_registry()

# Find by name
meta = registry.get_agent_metadata("data_domain_expert")
print(meta.description)
print(meta.capabilities)

# Find by capability
agents = registry.get_agents_by_capability("data_dossier")
# ["data_domain_expert", "data_insights_team"]

agents = registry.get_agents_by_capability("data_storytelling")
# ["data_storyteller", "data_insights_team"]

# All data_analysis agents
agents = registry.get_agents_by_category("data_analysis")
# ["data_visualization", "eda_tools", "data_storyteller", "data_domain_expert"]

# All multi-agents
agents = registry.get_agents_by_category("multi_agent")
# ["pandas_data_analyst", "sql_data_analyst", "data_insights_team"]
```

---

## Suggested App Integration Patterns

### Pattern: Dataset Onboarding Widget
When a user uploads a new dataset, automatically run DDEA to profile it and display the dossier:
```python
# On file upload
ddea = DataDomainExpertAgent(model=llm)
ddea.invoke_agent("Profile this dataset and generate a DATA_DOSSIER.md", data_raw=df)
st.markdown(ddea.get_dossier())
```

### Pattern: One-Click Full Analysis
Let users request a full analysis with a single button:
```python
team = DataInsightsTeam(
    ddea_agent=DataDomainExpertAgent(model=llm),
    dist_agent=DataStorytellerAgent(model=llm),
)
team.invoke_agent(user_instructions=user_question, data_raw=df)

tab1, tab2 = st.tabs(["Data Story", "Data Dossier"])
with tab1:
    st.markdown(team.get_story())
with tab2:
    st.markdown(team.get_dossier())
```

### Pattern: Validation Before ML
Before training a model, check ML readiness:
```python
ddea = DataDomainExpertAgent(model=llm)
ddea.invoke_agent(
    user_instructions=f"Intent: ml_readiness_report. Target column: {target_col}",
    data_raw=df,
)
artifacts = ddea.get_artifacts()
verdict = artifacts.get("verdict", "UNKNOWN")
# "READY" | "READY_WITH_CAVEATS" | "NEEDS_WORK" | "NOT_READY"
```

### Pattern: Join Safety Check
Before merging datasets:
```python
ddea = DataDomainExpertAgent(model=llm)
ddea.invoke_agent(
    user_instructions="Intent: relationships_and_joins. Check join safety on order_id.",
    data_raw=left_df,
    secondary_data_raw=right_df,
)
join_info = ddea.get_artifacts().get("cross_dataset_analysis", {})
multiplicity = join_info.get("multiplicity", "unknown")
```

---

## File Locations Summary

```
ai_data_science_team/
├── agents/data_analysis/
│   ├── data_domain_expert.py      # DataDomainExpertAgent class + factory
│   └── data_storyteller.py        # DataStorytellerAgent class + factory
├── multiagents/
│   └── data_insights_team.py      # DataInsightsTeam class + factory
├── tools/data_analysis/
│   ├── domain_expert_tools.py     # 8 DDEA tools
│   └── storytelling_tools.py      # 6 DIST tools
└── utils/
    ├── data_dossier.py            # DATA_DOSSIER.md + A2A formatting
    └── storytelling.py            # Narrative report formatting
```
