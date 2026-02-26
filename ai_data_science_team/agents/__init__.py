"""
Agents for AI Data Science Team

This module contains all agents for data science tasks:
- Data preparation: cleaning, wrangling, feature engineering
- Data access: loading, SQL database connections
- Data analysis: visualization, EDA, data storytelling
- Machine learning: H2O AutoML, MLflow

All agents can be discovered through the agent registry.
"""

# Agent Registry
from ai_data_science_team.agents.registry import (
    AgentRegistry,
    AgentMetadata,
    get_agent_registry,
    register_agent,
)

# Import from new categorized structure
from ai_data_science_team.agents.data_preparation import (
    DataCleaningAgent,
    make_data_cleaning_agent,
    DataWranglingAgent,
    make_data_wrangling_agent,
    FeatureEngineeringAgent,
    make_feature_engineering_agent,
)

from ai_data_science_team.agents.data_access import (
    DataLoaderToolsAgent,
    make_data_loader_tools_agent,
    SQLDatabaseAgent,
    make_sql_database_agent,
)

from ai_data_science_team.agents.data_analysis import (
    DataVisualizationAgent,
    make_data_visualization_agent,
    EDAToolsAgent,
    make_eda_tools_agent,
    DataStorytellerAgent,
    make_data_storyteller_agent,
    DataDomainExpertAgent,
    make_data_domain_expert_agent,
)

from ai_data_science_team.agents.machine_learning import (
    H2OMLAgent,
    make_h2o_ml_agent,
    MLflowToolsAgent,
    make_mlflow_tools_agent,
)

__all__ = [
    # Registry
    "AgentRegistry",
    "AgentMetadata",
    "get_agent_registry",
    "register_agent",
    # Data Preparation Agents
    "DataCleaningAgent",
    "make_data_cleaning_agent",
    "DataWranglingAgent",
    "make_data_wrangling_agent",
    "FeatureEngineeringAgent",
    "make_feature_engineering_agent",
    # Data Access Agents
    "DataLoaderToolsAgent",
    "make_data_loader_tools_agent",
    "SQLDatabaseAgent",
    "make_sql_database_agent",
    # Data Analysis Agents
    "DataVisualizationAgent",
    "make_data_visualization_agent",
    "EDAToolsAgent",
    "make_eda_tools_agent",
    "DataStorytellerAgent",
    "make_data_storyteller_agent",
    "DataDomainExpertAgent",
    "make_data_domain_expert_agent",
    # Machine Learning Agents
    "H2OMLAgent",
    "make_h2o_ml_agent",
    "MLflowToolsAgent",
    "make_mlflow_tools_agent",
]
