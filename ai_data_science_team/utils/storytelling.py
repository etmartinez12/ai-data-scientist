"""
Utility functions for data storytelling and narrative generation.

These utilities help format outputs from the DataStorytellerAgent into
professional narratives and reports.
"""

from typing import Dict, List, Optional, Any


def format_story_spine(
    context: str,
    tension: str,
    insight: str,
    implication: str,
    action: str,
) -> str:
    """
    Format findings into a 5-part story spine structure.

    The story spine follows: Context → Tension → Insight → Implication → Action

    Parameters
    ----------
    context : str
        What is happening / background situation.
    tension : str
        What is surprising, problematic, or worth investigating.
    insight : str
        What the data reveals / the key finding.
    implication : str
        What this means for the business / stakeholders.
    action : str
        What should be done next / recommendations.

    Returns
    -------
    str
        Formatted story spine as markdown.
    """
    return f"""## Data Story Spine

**Context** (What is happening):
{context}

**Tension** (What is surprising or problematic):
{tension}

**Insight** (What the data reveals):
{insight}

**Implication** (What this means):
{implication}

**Action** (What to do next):
{action}
"""


def format_kpi_table(kpis: List[Dict[str, Any]]) -> str:
    """
    Format a list of KPI dictionaries as a markdown table.

    Parameters
    ----------
    kpis : list of dict
        Each dict should have keys: name, value, definition, caveats.
        Keys 'definition' and 'caveats' are optional.

    Returns
    -------
    str
        Markdown table of KPIs.
    """
    if not kpis:
        return "_No KPIs defined._"

    has_definition = any("definition" in k for k in kpis)
    has_caveats = any("caveats" in k for k in kpis)

    if has_definition and has_caveats:
        header = "| Metric | Value | Definition | Caveats |\n|--------|-------|------------|---------|"
        rows = [
            f"| {k.get('name', '')} | {k.get('value', '')} | {k.get('definition', '')} | {k.get('caveats', '')} |"
            for k in kpis
        ]
    elif has_definition:
        header = "| Metric | Value | Definition |\n|--------|-------|------------|"
        rows = [
            f"| {k.get('name', '')} | {k.get('value', '')} | {k.get('definition', '')} |"
            for k in kpis
        ]
    else:
        header = "| Metric | Value |\n|--------|-------|"
        rows = [
            f"| {k.get('name', '')} | {k.get('value', '')} |"
            for k in kpis
        ]

    return "\n".join([header] + rows)


def format_insights_report(
    title: str,
    executive_summary: List[str],
    key_questions: List[str],
    metrics_table: str,
    methodology_technical: str,
    methodology_plain: str,
    recommendations: List[str],
    open_questions: List[str],
    integrity_checks: Optional[str] = None,
    artifacts: Optional[List[str]] = None,
) -> str:
    """
    Format a complete INSIGHTS_REPORT.md following the DIST deliverables standard.

    Parameters
    ----------
    title : str
        Report title.
    executive_summary : list of str
        5–10 bullet points for the executive summary.
    key_questions : list of str
        Questions the analysis set out to answer.
    metrics_table : str
        Pre-formatted KPI markdown table (use format_kpi_table).
    methodology_technical : str
        Technical methodology description.
    methodology_plain : str
        Plain-language methodology description.
    recommendations : list of str
        Prioritized recommendations / next actions.
    open_questions : list of str
        Unresolved questions and tests needed.
    integrity_checks : str, optional
        Integrity and reconciliation check results.
    artifacts : list of str, optional
        Paths/names of generated artifacts.

    Returns
    -------
    str
        Complete formatted INSIGHTS_REPORT.md content.
    """
    exec_bullets = "\n".join(f"- {b}" for b in executive_summary)
    questions_list = "\n".join(f"- {q}" for q in key_questions)
    recs_list = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations))
    open_q_list = "\n".join(f"- {q}" for q in open_questions)
    artifacts_section = (
        "\n".join(f"- `{a}`" for a in artifacts) if artifacts else "_None generated._"
    )
    integrity_section = integrity_checks if integrity_checks else "_Not performed._"

    return f"""# {title}

---

## 1. Executive Summary

{exec_bullets}

---

## 2. Key Questions Answered

{questions_list}

---

## 3. Key Metrics

{metrics_table}

---

## 4. Integrity & Reconciliation Checks

{integrity_section}

---

## 5. Methodology (Technical)

{methodology_technical}

---

## 6. Methodology (Plain English)

{methodology_plain}

---

## 7. Recommendations & Next Actions

{recs_list}

---

## 8. Open Questions & Uncertainties

{open_q_list}

---

## 9. Artifacts

{artifacts_section}
"""


def format_metrics_catalog(metrics: List[Dict[str, Any]]) -> str:
    """
    Format a METRICS_CATALOG.md from a list of metric definitions.

    Parameters
    ----------
    metrics : list of dict
        Each dict should have keys:
        - name: Metric name
        - purpose: Why we measure this
        - grain: Unit/grain (e.g., "per customer per month")
        - denominator: Denominator definition
        - filters: Any eligibility filters
        - formula: Computation formula
        - caveats: Known caveats or edge cases
        - validation: Validation checks (VAL-### style)

    Returns
    -------
    str
        Formatted METRICS_CATALOG.md content.
    """
    if not metrics:
        return "# Metrics Catalog\n\n_No metrics defined._"

    sections = ["# Metrics Catalog\n"]
    sections.append("| # | Metric | Grain | Formula | Caveats |\n|---|--------|-------|---------|---------|")

    for i, m in enumerate(metrics, 1):
        sections.append(
            f"| {i} | **{m.get('name', '')}** | {m.get('grain', '')} | "
            f"`{m.get('formula', '')}` | {m.get('caveats', '')} |"
        )

    sections.append("\n---\n\n## Detailed Definitions\n")

    for m in metrics:
        sections.append(f"### {m.get('name', 'Unnamed Metric')}\n")
        if m.get("purpose"):
            sections.append(f"**Purpose:** {m['purpose']}\n")
        if m.get("grain"):
            sections.append(f"**Grain:** {m['grain']}\n")
        if m.get("denominator"):
            sections.append(f"**Denominator:** {m['denominator']}\n")
        if m.get("filters"):
            sections.append(f"**Filters/Eligibility:** {m['filters']}\n")
        if m.get("formula"):
            sections.append(f"**Formula:** `{m['formula']}`\n")
        if m.get("caveats"):
            sections.append(f"**Caveats:** {m['caveats']}\n")
        if m.get("validation"):
            sections.append(f"**Validation:** {m['validation']}\n")
        sections.append("")

    return "\n".join(sections)


def format_agent_story_output(response: Dict[str, Any]) -> str:
    """
    Format the raw agent response into a readable story output.

    Parameters
    ----------
    response : dict
        The agent's response dictionary from invoke_agent.

    Returns
    -------
    str
        Formatted story output as markdown.
    """
    if not response:
        return "_No story generated._"

    messages = response.get("messages", [])
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", str(last_message))
        return content

    return "_No story content found in response._"
