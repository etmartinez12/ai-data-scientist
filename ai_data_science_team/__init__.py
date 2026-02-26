"""
AI Data Science Team

An AI-powered data science team of agents for performing data science tasks.

Main Components:
- Core: LLM provider factory, configuration management
- Agents: Data preparation, access, analysis, and machine learning agents
- Tools: Reusable functions for data operations
- Multi-agents: Composite agents combining multiple capabilities
"""

# Core infrastructure
from ai_data_science_team.core import (
    create_llm,
    get_available_providers,
    get_config,
    LLMConfig,
)

# All agents (from new categorized structure)
from ai_data_science_team.agents import (
    # Data Preparation
    DataCleaningAgent,
    DataWranglingAgent,
    FeatureEngineeringAgent,
    # Data Access
    DataLoaderToolsAgent,
    SQLDatabaseAgent,
    # Data Analysis
    DataVisualizationAgent,
    EDAToolsAgent,
    DataStorytellerAgent,
    DataDomainExpertAgent,
    # Machine Learning
    H2OMLAgent,
    MLflowToolsAgent,
    # Registry
    get_agent_registry,
)

# Multi-agents
from ai_data_science_team.multiagents import (
    SQLDataAnalyst,
    PandasDataAnalyst,
    DataInsightsTeam,
)

# Tools
from ai_data_science_team.tools import (
    get_tool_registry,
)

__all__ = [
    # Core
    "create_llm",
    "get_available_providers",
    "get_config",
    "LLMConfig",
    # Data Preparation Agents
    "DataCleaningAgent",
    "DataWranglingAgent",
    "FeatureEngineeringAgent",
    # Data Access Agents
    "DataLoaderToolsAgent",
    "SQLDatabaseAgent",
    # Data Analysis Agents
    "DataVisualizationAgent",
    "EDAToolsAgent",
    "DataStorytellerAgent",
    "DataDomainExpertAgent",
    # Machine Learning Agents
    "H2OMLAgent",
    "MLflowToolsAgent",
    # Multi-agents
    "SQLDataAnalyst",
    "PandasDataAnalyst",
    "DataInsightsTeam",
    # Registries
    "get_agent_registry",
    "get_tool_registry",
]
