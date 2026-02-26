# Adding Custom Tools

This guide explains how to create and register custom tools for AI Data Science Team agents.

## Table of Contents

- [What are Tools?](#what-are-tools)
- [Quick Start](#quick-start)
- [Tool Registry System](#tool-registry-system)
- [Creating Tools](#creating-tools)
- [Registering Tools](#registering-tools)
- [Examples](#examples)
- [Best Practices](#best-practices)

## What are Tools?

Tools are reusable Python functions that agents can use to perform specific tasks:
- Data loading and I/O operations
- DataFrame manipulations
- SQL queries
- Statistical calculations
- Machine learning operations

**Key Difference from Agents:**
- **Tools** are individual functions for specific tasks
- **Agents** use LLMs to generate code that may call tools

## Quick Start

Creating a tool is simple:

```python
from ai_data_science_team.tools import register_tool

@register_tool(
    name="calculate_statistics",
    category="statistics",
    description="Calculate basic statistics for a DataFrame column",
    parameters=["data", "column_name"],
    returns="Dictionary with mean, median, std",
    examples=["stats = calculate_statistics(df, 'age')"]
)
def calculate_statistics(data, column_name):
    import pandas as pd

    series = data[column_name]
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max()
    }
```

That's it! The tool is now:
- Registered in the global tool registry
- Discoverable via `get_tool_registry()`
- Accessible to agents

## Tool Registry System

The tool registry provides:
- **Auto-discovery**: Tools are automatically found and registered
- **Metadata**: Each tool has name, category, description, parameters
- **Querying**: Find tools by category, name, or search terms

### Using the Registry

```python
from ai_data_science_team.tools import get_tool_registry

# Get the registry
registry = get_tool_registry()

# List all tools
tools = registry.list_tools()

# Get tools by category
stats_tools = registry.get_tools_by_category("statistics")

# Search for tools
calc_tools = registry.search_tools("calculate")

# Get a specific tool
tool_func = registry.get_tool("calculate_statistics")
result = tool_func(df, "age")

# Get tool metadata
metadata = registry.get_tool_metadata("calculate_statistics")
print(metadata.description)
print(metadata.parameters)
```

## Creating Tools

### Method 1: Decorator (Recommended)

Use the `@register_tool` decorator:

```python
from ai_data_science_team.tools import register_tool

@register_tool(
    name="my_tool",
    category="data_processing",
    description="Process data in a specific way",
    parameters=["data", "option"],
    returns="Processed data",
    examples=["result = my_tool(df, option='fast')"]
)
def my_tool(data, option='default'):
    # Your implementation
    processed = data.copy()
    # ... processing logic
    return processed
```

**Parameters:**
- `name`: Unique tool identifier (required)
- `category`: Tool category for organization (required)
- `description`: What the tool does (required)
- `parameters`: List of parameter names (required)
- `returns`: Description of return value (required)
- `examples`: Usage examples (optional)

### Method 2: Manual Registration

```python
from ai_data_science_team.tools import get_tool_registry

def my_tool(data, option='default'):
    # Implementation
    return data

# Get registry
registry = get_tool_registry()

# Register manually
registry.register_tool(
    name="my_tool",
    func=my_tool,
    category="data_processing",
    description="Process data",
    parameters=["data", "option"],
    returns="Processed data"
)
```

### Method 3: Class-Based Tools

For more complex tools, use classes:

```python
from ai_data_science_team.tools.base import BaseTool, ToolMetadata

class StatisticalAnalyzer(BaseTool):
    def get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="statistical_analyzer",
            category="statistics",
            description="Comprehensive statistical analysis",
            parameters=["data", "columns"],
            returns="Analysis results dictionary"
        )

    def execute(self, data, columns=None):
        import pandas as pd

        if columns is None:
            columns = data.select_dtypes(include='number').columns

        results = {}
        for col in columns:
            results[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std()
            }

        return results

# Register the tool
tool = StatisticalAnalyzer()
registry = get_tool_registry()
registry.register_from_function(tool)
```

## Tool Categories

Organize tools into logical categories:

```
tools/
├── data_loading/       # File I/O, data loading
├── data_analysis/      # EDA, statistics, profiling
├── database/           # SQL operations
├── ml/                 # Machine learning utilities
└── custom/             # Your custom category
```

### Creating a New Category

1. Create directory: `ai_data_science_team/tools/my_category/`
2. Add `__init__.py`:

```python
"""
My Custom Tools

Tools for specific domain tasks.
"""

# Import tools to make them discoverable
from ai_data_science_team.tools.my_category.my_tools import *

__all__ = [
    # Tool names
]
```

3. Create tool file: `my_tools.py`:

```python
from ai_data_science_team.tools import register_tool

@register_tool(
    name="domain_specific_tool",
    category="my_category",
    description="Does something domain-specific",
    parameters=["input_data"],
    returns="Processed output"
)
def domain_specific_tool(input_data):
    # Implementation
    return processed_output
```

## Examples

### Example 1: Data Validation Tool

```python
from ai_data_science_team.tools import register_tool
import pandas as pd

@register_tool(
    name="validate_dataframe",
    category="data_quality",
    description="Validate DataFrame structure and data quality",
    parameters=["data", "required_columns", "allow_missing"],
    returns="Validation results dictionary",
    examples=[
        "results = validate_dataframe(df, ['id', 'name'], allow_missing=False)"
    ]
)
def validate_dataframe(data, required_columns=None, allow_missing=True):
    """
    Validate a DataFrame for required structure and data quality.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to validate
    required_columns : list, optional
        List of required column names
    allow_missing : bool
        Whether to allow missing values

    Returns
    -------
    dict : Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {missing_cols}")

    # Check for missing values
    if not allow_missing:
        null_counts = data.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            results['valid'] = False
            results['errors'].append(f"Columns with nulls: {cols_with_nulls.to_dict()}")

    # Add statistics
    results['statistics'] = {
        'rows': len(data),
        'columns': len(data.columns),
        'missing_cells': data.isnull().sum().sum(),
        'duplicates': data.duplicated().sum()
    }

    return results
```

### Example 2: Time Series Tool

```python
from ai_data_science_team.tools import register_tool
import pandas as pd

@register_tool(
    name="detect_seasonality",
    category="time_series",
    description="Detect seasonal patterns in time series data",
    parameters=["data", "date_column", "value_column", "freq"],
    returns="Seasonality analysis results"
)
def detect_seasonality(data, date_column, value_column, freq='M'):
    """
    Detect seasonal patterns in time series.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time series data
    date_column : str
        Name of date column
    value_column : str
        Name of value column
    freq : str
        Frequency ('D', 'W', 'M', 'Q', 'Y')

    Returns
    -------
    dict : Seasonality metrics
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)

    # Decompose
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(
        df[value_column],
        model='additive',
        period=12 if freq == 'M' else 4
    )

    return {
        'has_trend': decomposition.trend.notna().any(),
        'has_seasonality': decomposition.seasonal.std() > 0,
        'seasonal_strength': decomposition.seasonal.std() / df[value_column].std(),
        'trend_direction': 'increasing' if decomposition.trend.diff().mean() > 0 else 'decreasing'
    }
```

### Example 3: ML Feature Tool

```python
from ai_data_science_team.tools import register_tool
import pandas as pd
import numpy as np

@register_tool(
    name="create_interaction_features",
    category="feature_engineering",
    description="Create interaction features from numeric columns",
    parameters=["data", "columns", "degree"],
    returns="DataFrame with interaction features"
)
def create_interaction_features(data, columns=None, degree=2):
    """
    Create polynomial and interaction features.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    columns : list, optional
        Columns to use (default: all numeric)
    degree : int
        Polynomial degree (default: 2)

    Returns
    -------
    pd.DataFrame : Data with new features
    """
    from sklearn.preprocessing import PolynomialFeatures

    df = data.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[columns])

    # Get feature names
    feature_names = poly.get_feature_names_out(columns)

    # Add to DataFrame
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

    # Combine
    result = pd.concat([df, poly_df], axis=1)

    return result
```

### Example 4: Database Tool

```python
from ai_data_science_team.tools import register_tool
import pandas as pd

@register_tool(
    name="execute_sql_with_params",
    category="database",
    description="Execute parameterized SQL query safely",
    parameters=["connection_string", "query", "params"],
    returns="Query results as DataFrame"
)
def execute_sql_with_params(connection_string, query, params=None):
    """
    Execute SQL query with parameters (SQL injection safe).

    Parameters
    ----------
    connection_string : str
        Database connection string
    query : str
        SQL query with placeholders (?)
    params : tuple or dict, optional
        Query parameters

    Returns
    -------
    pd.DataFrame : Query results

    Examples
    --------
    >>> df = execute_sql_with_params(
    ...     "sqlite:///db.sqlite",
    ...     "SELECT * FROM users WHERE age > ?",
    ...     (25,)
    ... )
    """
    import sqlalchemy as sql

    engine = sql.create_engine(connection_string)

    with engine.connect() as conn:
        result = pd.read_sql_query(query, conn, params=params)

    return result
```

## Best Practices

### 1. Single Responsibility

Each tool should do one thing well:

```python
# Good: Specific, focused
@register_tool(...)
def calculate_mean(data, column):
    return data[column].mean()

# Bad: Does too many things
@register_tool(...)
def analyze_everything(data):
    # Calculates stats, creates plots, fits models...
    pass
```

### 2. Clear Documentation

Document parameters, returns, and provide examples:

```python
@register_tool(...)
def my_tool(data, param):
    """
    Brief description.

    Parameters
    ----------
    data : pd.DataFrame
        Description of data parameter
    param : str
        Description of param

    Returns
    -------
    type : Description of return value

    Examples
    --------
    >>> result = my_tool(df, 'value')
    """
    pass
```

### 3. Type Hints

Use type hints for clarity:

```python
from typing import Dict, List, Optional
import pandas as pd

@register_tool(...)
def process_data(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    options: Optional[Dict] = None
) -> pd.DataFrame:
    """Process data with options."""
    pass
```

### 4. Error Handling

Handle errors gracefully:

```python
@register_tool(...)
def safe_calculation(data, column):
    """Calculate metric safely."""
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    if not pd.api.types.is_numeric_dtype(data[column]):
        raise TypeError(f"Column '{column}' must be numeric")

    return data[column].mean()
```

### 5. Performance

Optimize for common cases:

```python
@register_tool(...)
def efficient_groupby(data, group_col, agg_col, agg_func='mean'):
    """Efficient groupby operation."""
    # Use pandas built-in optimizations
    return data.groupby(group_col)[agg_col].agg(agg_func)
```

### 6. Testable

Write testable tools:

```python
import pytest
import pandas as pd

def test_calculate_mean():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    result = calculate_mean(df, 'A')
    assert result == 3.0

def test_calculate_mean_missing_column():
    df = pd.DataFrame({'A': [1, 2, 3]})
    with pytest.raises(KeyError):
        calculate_mean(df, 'B')
```

### 7. Reusability

Make tools reusable across contexts:

```python
@register_tool(...)
def normalize_column(data, column, method='zscore'):
    """
    Normalize a column using different methods.

    Supports: zscore, minmax, robust
    """
    if method == 'zscore':
        return (data[column] - data[column].mean()) / data[column].std()
    elif method == 'minmax':
        return (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    elif method == 'robust':
        return (data[column] - data[column].median()) / (data[column].quantile(0.75) - data[column].quantile(0.25))
```

## Tool Discovery and Usage

### Discovery

```python
from ai_data_science_team.tools import get_tool_registry

registry = get_tool_registry()

# Browse all tools
print(f"Total tools: {len(registry)}")

# By category
for category in registry.list_categories():
    tools = registry.get_tools_by_category(category)
    print(f"\n{category}: {len(tools)} tools")
    for tool_name in tools:
        metadata = registry.get_tool_metadata(tool_name)
        print(f"  - {tool_name}: {metadata.description}")
```

### Using Tools in Agents

Tools are automatically available to agents:

```python
from ai_data_science_team import DataWranglingAgent, create_llm

llm = create_llm()
agent = DataWranglingAgent(model=llm)

# The agent can use registered tools in generated code
agent.invoke_agent(
    user_instructions="Use the validate_dataframe tool to check data quality",
    data_raw=df
)
```

### Direct Usage

```python
from ai_data_science_team.tools import get_tool_registry

registry = get_tool_registry()

# Get and use a tool
validate_func = registry.get_tool("validate_dataframe")
results = validate_func(df, required_columns=['id', 'name'])

print(results)
```

## Advanced Topics

### Custom Tool Base Classes

For complex tool families:

```python
from ai_data_science_team.tools.base import BaseTool

class MLMetricTool(BaseTool):
    """Base class for ML metric tools."""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def get_metadata(self):
        return ToolMetadata(
            name=self.name,
            category="ml_metrics",
            description=self.description,
            parameters=["y_true", "y_pred"],
            returns="Metric value"
        )

    def execute(self, y_true, y_pred):
        raise NotImplementedError
```

### Tool Dependencies

Manage tool dependencies:

```python
@register_tool(...)
def advanced_tool(data):
    """Tool that uses other tools."""
    from ai_data_science_team.tools import get_tool_registry

    registry = get_tool_registry()

    # Use another tool
    validate = registry.get_tool("validate_dataframe")
    validation = validate(data)

    if not validation['valid']:
        raise ValueError("Data validation failed")

    # Proceed with processing
    return processed_data
```

## Troubleshooting

### Tool not appearing in registry

Make sure:
1. Tool file is in `tools/` directory or subdirectory
2. Tool is decorated with `@register_tool`
3. Tool module is imported in `__init__.py`

### Import errors

Check that all dependencies are installed:

```python
@register_tool(...)
def tool_with_deps(data):
    try:
        import specialized_library
    except ImportError:
        raise ImportError("This tool requires: pip install specialized_library")

    return result
```

## Next Steps

1. Create your custom tool
2. Test thoroughly
3. Document with examples
4. Register in appropriate category
5. Share with the community

See also:
- [ADDING_AGENTS.md](ADDING_AGENTS.md) - Creating custom agents
- [CONFIGURATION.md](CONFIGURATION.md) - LLM setup
