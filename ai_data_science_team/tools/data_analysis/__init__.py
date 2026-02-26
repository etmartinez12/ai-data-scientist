"""
Data Analysis Tools

Tools for exploratory data analysis and statistical analysis:
- EDA functions
- Data profiling
- Missing data analysis
- Correlation analysis
- Summary statistics
- Storytelling tools (KPI computation, pattern detection, narrative construction)
"""

from ai_data_science_team.tools.data_analysis.eda_tools import *
from ai_data_science_team.tools.data_analysis.storytelling_tools import (
    profile_data_for_story,
    compute_dataset_kpis,
    detect_data_patterns,
    validate_analysis_integrity,
    compute_group_comparisons,
    build_story_spine,
)
from ai_data_science_team.tools.data_analysis.domain_expert_tools import (
    profile_dataset_schema,
    profile_columns_deep,
    detect_edge_cases,
    analyze_join_keys,
    generate_validation_rules,
    assess_ml_readiness,
    plan_cleaning_steps,
    generate_data_dossier,
)

__all__ = [
    # Re-export all functions from eda_tools module
    # Functions will be available via auto-discovery
    # Storytelling tools
    "profile_data_for_story",
    "compute_dataset_kpis",
    "detect_data_patterns",
    "validate_analysis_integrity",
    "compute_group_comparisons",
    "build_story_spine",
    # Domain expert tools
    "profile_dataset_schema",
    "profile_columns_deep",
    "detect_edge_cases",
    "analyze_join_keys",
    "generate_validation_rules",
    "assess_ml_readiness",
    "plan_cleaning_steps",
    "generate_data_dossier",
]
