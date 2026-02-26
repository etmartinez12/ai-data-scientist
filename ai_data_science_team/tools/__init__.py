"""
Tools for AI Data Science Team

This module contains reusable tools that agents can use for various tasks:
- Data loading and I/O
- DataFrame operations and analysis
- SQL database operations
- Exploratory data analysis (EDA)
- Machine learning (H2O, MLflow)

Tools can be accessed directly or through the tool registry for discovery.
"""

# Base classes and registry
from ai_data_science_team.tools.base import (
    BaseTool,
    FunctionTool,
    ToolMetadata,
    tool,
)

from ai_data_science_team.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    register_tool,
)

# Import tool modules to make functions available
# These imports ensure all tools are discoverable
from ai_data_science_team.tools import (
    dataframe,
    data_loading,
    data_analysis,
    database,
    ml,
)

# Backward compatibility - keep old module names accessible
from ai_data_science_team.tools.data_loading import loaders as data_loader
from ai_data_science_team.tools.data_analysis import eda_tools as eda
from ai_data_science_team.tools.database import sql_tools as sql
from ai_data_science_team.tools.ml import h2o_tools as h2o
from ai_data_science_team.tools.ml import mlflow_tools as mlflow

__all__ = [
    # Base classes
    "BaseTool",
    "FunctionTool",
    "ToolMetadata",
    "tool",
    # Registry
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    # Tool modules (new structure)
    "dataframe",
    "data_loading",
    "data_analysis",
    "database",
    "ml",
    # Backward compatibility (old module names)
    "data_loader",
    "sql",
    "eda",
    "h2o",
    "mlflow",
]
