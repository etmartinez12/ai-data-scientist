"""
Agent Registry

Manages registration and discovery of all agents in the system.
Provides query interfaces for finding agents by category, name, or capabilities.
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field


@dataclass
class AgentMetadata:
    """
    Metadata for an agent.

    Attributes:
        name: Agent name (unique identifier)
        category: Agent category (e.g., "data_preparation", "data_analysis", "machine_learning")
        description: Brief description of what the agent does
        agent_class: The agent class
        required_tools: List of tools this agent uses
        capabilities: List of capabilities/tasks this agent can perform
        example_usage: Example code for using the agent
    """
    name: str
    category: str
    description: str
    agent_class: Type
    required_tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    example_usage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "agent_class": self.agent_class.__name__,
            "required_tools": self.required_tools,
            "capabilities": self.capabilities,
            "example_usage": self.example_usage,
        }


class AgentRegistry:
    """
    Central registry for all agents.

    Manages agent registration, discovery, and querying.
    """

    def __init__(self):
        self._agents: Dict[str, AgentMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._capabilities_index: Dict[str, List[str]] = {}

    def register_agent(
        self,
        name: str,
        category: str,
        description: str,
        agent_class: Type,
        required_tools: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        example_usage: Optional[str] = None
    ):
        """
        Register an agent.

        Args:
            name: Agent name (unique identifier)
            category: Agent category
            description: Agent description
            agent_class: The agent class
            required_tools: List of tools this agent uses
            capabilities: List of capabilities/tasks
            example_usage: Example code
        """
        metadata = AgentMetadata(
            name=name,
            category=category,
            description=description,
            agent_class=agent_class,
            required_tools=required_tools or [],
            capabilities=capabilities or [],
            example_usage=example_usage
        )

        self._agents[name] = metadata

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Update capabilities index
        for capability in metadata.capabilities:
            if capability not in self._capabilities_index:
                self._capabilities_index[capability] = []
            if name not in self._capabilities_index[capability]:
                self._capabilities_index[capability].append(name)

    def get_agent_class(self, name: str) -> Optional[Type]:
        """
        Get agent class by name.

        Args:
            name: Agent name

        Returns:
            Agent class or None if not found
        """
        metadata = self._agents.get(name)
        return metadata.agent_class if metadata else None

    def get_agent_metadata(self, name: str) -> Optional[AgentMetadata]:
        """
        Get agent metadata by name.

        Args:
            name: Agent name

        Returns:
            AgentMetadata or None if not found
        """
        return self._agents.get(name)

    def get_agents_by_category(self, category: str) -> List[str]:
        """
        Get all agent names in a category.

        Args:
            category: Category name

        Returns:
            List of agent names
        """
        return self._categories.get(category, [])

    def get_agents_by_capability(self, capability: str) -> List[str]:
        """
        Get all agent names with a specific capability.

        Args:
            capability: Capability name

        Returns:
            List of agent names
        """
        return self._capabilities_index.get(capability, [])

    def list_agents(self) -> List[str]:
        """
        List all registered agent names.

        Returns:
            List of agent names
        """
        return list(self._agents.keys())

    def list_categories(self) -> List[str]:
        """
        List all agent categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def list_capabilities(self) -> List[str]:
        """
        List all registered capabilities.

        Returns:
            List of capability names
        """
        return list(self._capabilities_index.keys())

    def search_agents(self, query: str) -> List[str]:
        """
        Search agents by name, description, or capabilities.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching agent names
        """
        query = query.lower()
        matches = []

        for name, metadata in self._agents.items():
            if (query in name.lower() or
                query in metadata.description.lower() or
                query in metadata.category.lower() or
                any(query in cap.lower() for cap in metadata.capabilities)):
                matches.append(name)

        return matches

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all agents.

        Returns:
            Dictionary mapping agent names to metadata dictionaries
        """
        return {
            name: metadata.to_dict()
            for name, metadata in self._agents.items()
        }

    def get_category_summary(self) -> Dict[str, int]:
        """
        Get count of agents per category.

        Returns:
            Dictionary mapping categories to agent counts
        """
        return {
            category: len(agents)
            for category, agents in self._categories.items()
        }

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"<AgentRegistry: {len(self._agents)} agents in {len(self._categories)} categories>"


# Global registry instance
_global_registry = None


def get_agent_registry() -> AgentRegistry:
    """
    Get the global agent registry instance.

    Returns:
        AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
        _register_all_agents()
    return _global_registry


def _register_all_agents():
    """Register all available agents in the system."""
    registry = _global_registry

    # Import agents to register them
    try:
        from ai_data_science_team.agents.data_preparation.data_cleaning import DataCleaningAgent
        registry.register_agent(
            name="data_cleaning",
            category="data_preparation",
            description="Performs data cleaning including handling missing values, outliers, duplicates, and type conversions",
            agent_class=DataCleaningAgent,
            required_tools=["dataframe"],
            capabilities=["remove_missing", "handle_outliers", "remove_duplicates", "type_conversion"],
            example_usage="agent = DataCleaningAgent(model=llm)\nagent.invoke_agent(user_instructions='Clean the data', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_preparation.data_wrangling import DataWranglingAgent
        registry.register_agent(
            name="data_wrangling",
            category="data_preparation",
            description="Merges, joins, and transforms data into analysis-ready format",
            agent_class=DataWranglingAgent,
            required_tools=["dataframe"],
            capabilities=["merge", "join", "transform", "reshape", "aggregate"],
            example_usage="agent = DataWranglingAgent(model=llm)\nagent.invoke_agent(user_instructions='Merge datasets', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_preparation.feature_engineering import FeatureEngineeringAgent
        registry.register_agent(
            name="feature_engineering",
            category="data_preparation",
            description="Creates and transforms features for machine learning models",
            agent_class=FeatureEngineeringAgent,
            required_tools=["dataframe"],
            capabilities=["create_features", "encode_categorical", "scale_features", "feature_selection"],
            example_usage="agent = FeatureEngineeringAgent(model=llm)\nagent.invoke_agent(user_instructions='Create features', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_access.data_loader import DataLoaderToolsAgent
        registry.register_agent(
            name="data_loader",
            category="data_access",
            description="Loads data from various file formats (CSV, Excel, Parquet, Pickle)",
            agent_class=DataLoaderToolsAgent,
            required_tools=["data_loader"],
            capabilities=["load_csv", "load_excel", "load_parquet", "load_pickle"],
            example_usage="agent = DataLoaderToolsAgent(model=llm)\nagent.invoke_agent(user_instructions='Load the data', file_path='data.csv')"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_access.sql_database import SQLDatabaseAgent
        registry.register_agent(
            name="sql_database",
            category="data_access",
            description="Connects to SQL databases and executes queries",
            agent_class=SQLDatabaseAgent,
            required_tools=["sql"],
            capabilities=["query_database", "create_pipeline", "join_tables", "aggregate_data"],
            example_usage="agent = SQLDatabaseAgent(model=llm, connection_string='sqlite:///db.sqlite')\nagent.invoke_agent(user_instructions='Get customer data')"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_analysis.data_visualization import DataVisualizationAgent
        registry.register_agent(
            name="data_visualization",
            category="data_analysis",
            description="Creates interactive visualizations using Plotly",
            agent_class=DataVisualizationAgent,
            required_tools=["plotly"],
            capabilities=["create_plots", "interactive_viz", "dashboard", "charts"],
            example_usage="agent = DataVisualizationAgent(model=llm)\nagent.invoke_agent(user_instructions='Create a scatter plot', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_analysis.eda import EDAToolsAgent
        registry.register_agent(
            name="eda_tools",
            category="data_analysis",
            description="Performs automated exploratory data analysis with reporting",
            agent_class=EDAToolsAgent,
            required_tools=["eda"],
            capabilities=["eda_report", "missing_analysis", "correlation_analysis", "distribution_analysis"],
            example_usage="agent = EDAToolsAgent(model=llm)\nagent.invoke_agent(user_instructions='Perform EDA', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_analysis.data_storyteller import DataStorytellerAgent
        registry.register_agent(
            name="data_storyteller",
            category="data_analysis",
            description="Transforms datasets into decision-grade insights, KPIs, and professional narratives for both technical and non-technical audiences (DIST — Data Insights Storyteller)",
            agent_class=DataStorytellerAgent,
            required_tools=["eda", "storytelling"],
            capabilities=[
                "data_storytelling", "kpi_engineering", "pattern_discovery",
                "narrative_report", "integrity_validation", "group_comparison",
                "executive_summary", "metrics_catalog", "anomaly_detection",
                "workflow_verification",
            ],
            example_usage=(
                "agent = DataStorytellerAgent(model=llm)\n"
                "agent.invoke_agent(\n"
                "    user_instructions='Tell me the story of this sales data',\n"
                "    data_raw=df\n"
                ")\n"
                "story = agent.get_story(markdown=True)"
            )
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.data_analysis.data_domain_expert import DataDomainExpertAgent
        registry.register_agent(
            name="data_domain_expert",
            category="data_analysis",
            description="Authoritative dataset profiler and documenter. Exhaustively profiles datasets, generates DATA_DOSSIER.md, and answers structured A2A queries about data structure, quality, edge cases, and semantics (DDEA — Data Domain Expert Agent)",
            agent_class=DataDomainExpertAgent,
            required_tools=["eda", "domain_expert"],
            capabilities=[
                "dataset_profiling", "schema_inference", "column_profiling",
                "edge_case_detection", "join_analysis", "validation_rules",
                "cleaning_plan", "ml_readiness", "data_dossier",
                "workflow_verification", "workflow_monitoring",
                "a2a_interface", "pii_detection",
            ],
            example_usage=(
                "agent = DataDomainExpertAgent(model=llm)\n"
                "agent.invoke_agent(\n"
                "    user_instructions='Profile this dataset and generate a DATA_DOSSIER.md',\n"
                "    data_raw=df\n"
                ")\n"
                "dossier = agent.get_dossier(markdown=True)"
            )
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.machine_learning.h2o_ml import H2OMLAgent
        registry.register_agent(
            name="h2o_ml",
            category="machine_learning",
            description="Builds and trains machine learning models using H2O AutoML",
            agent_class=H2OMLAgent,
            required_tools=["h2o"],
            capabilities=["automl", "classification", "regression", "model_training", "leaderboard"],
            example_usage="agent = H2OMLAgent(model=llm)\nagent.invoke_agent(user_instructions='Build classification model', data_raw=df, target_variable='churn')"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.agents.machine_learning.mlflow_tools import MLflowToolsAgent
        registry.register_agent(
            name="mlflow_tools",
            category="machine_learning",
            description="Manages ML experiments and models with MLflow",
            agent_class=MLflowToolsAgent,
            required_tools=["mlflow"],
            capabilities=["experiment_tracking", "model_registry", "model_deployment", "artifact_logging"],
            example_usage="agent = MLflowToolsAgent(model=llm)\nagent.invoke_agent(user_instructions='Log model to MLflow')"
        )
    except ImportError:
        pass

    # Multi-agents
    try:
        from ai_data_science_team.multiagents.pandas_data_analyst import PandasDataAnalyst
        registry.register_agent(
            name="pandas_data_analyst",
            category="multi_agent",
            description="Combines data wrangling and visualization capabilities",
            agent_class=PandasDataAnalyst,
            required_tools=["dataframe", "plotly"],
            capabilities=["wrangle", "visualize", "analyze", "transform_and_plot"],
            example_usage="agent = PandasDataAnalyst(model=llm, data_wrangling_agent=dw_agent, data_visualization_agent=dv_agent)\nagent.invoke_agent(user_instructions='Analyze and visualize', data_raw=df)"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.multiagents.sql_data_analyst import SQLDataAnalyst
        registry.register_agent(
            name="sql_data_analyst",
            category="multi_agent",
            description="Combines SQL database access and visualization",
            agent_class=SQLDataAnalyst,
            required_tools=["sql", "plotly"],
            capabilities=["query_and_visualize", "database_analysis", "sql_reporting"],
            example_usage="agent = SQLDataAnalyst(model=llm, sql_agent=sql_agent, viz_agent=viz_agent)\nagent.invoke_agent(user_instructions='Query and visualize sales data')"
        )
    except ImportError:
        pass

    try:
        from ai_data_science_team.multiagents.data_insights_team import DataInsightsTeam
        registry.register_agent(
            name="data_insights_team",
            category="multi_agent",
            description="Sequential DDEA -> DIST pipeline: profiles dataset exhaustively (DDEA), then generates a decision-grade narrative report (DIST). The dossier from DDEA is automatically injected as context for the storyteller.",
            agent_class=DataInsightsTeam,
            required_tools=["domain_expert", "storytelling", "eda"],
            capabilities=[
                "full_data_analysis", "dataset_profiling", "data_storytelling",
                "kpi_engineering", "data_dossier", "narrative_report",
                "end_to_end_analysis", "schema_inference", "edge_case_detection",
            ],
            example_usage=(
                "team = DataInsightsTeam(ddea_agent=ddea, dist_agent=dist)\n"
                "team.invoke_agent(\n"
                "    user_instructions='Tell me the key story in this data',\n"
                "    data_raw=df\n"
                ")\n"
                "dossier = team.get_dossier(markdown=True)\n"
                "story = team.get_story(markdown=True)"
            )
        )
    except ImportError:
        pass


def register_agent(
    name: str,
    category: str,
    description: str,
    agent_class: Type,
    required_tools: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    example_usage: Optional[str] = None
):
    """
    Register an agent in the global registry.

    Args:
        name: Agent name
        category: Agent category
        description: Agent description
        agent_class: The agent class
        required_tools: List of required tools
        capabilities: List of capabilities
        example_usage: Example usage code
    """
    registry = get_agent_registry()
    registry.register_agent(
        name=name,
        category=category,
        description=description,
        agent_class=agent_class,
        required_tools=required_tools,
        capabilities=capabilities,
        example_usage=example_usage
    )
