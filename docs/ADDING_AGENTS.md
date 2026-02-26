# Adding Custom Agents

This guide explains how to create custom agents for AI Data Science Team.

## Table of Contents

- [Quick Start](#quick-start)
- [Agent Architecture](#agent-architecture)
- [Step-by-Step Guide](#step-by-step-guide)
- [Registering Your Agent](#registering-your-agent)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Quick Start

Creating a custom agent involves:
1. Inheriting from `BaseAgent`
2. Implementing required methods
3. Registering in the agent registry

**Minimal Example:**

```python
from ai_data_science_team.templates import BaseAgent, create_coding_agent_graph

class MyCustomAgent(BaseAgent):
    def __init__(self, model, log=False, log_path=None, **kwargs):
        self._params = {
            "model": model,
            "log": log,
            "log_path": log_path,
            **kwargs
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            agent_name="my_custom_agent",
            model=self._params["model"],
            system_prompt="You are a helpful data science assistant.",
            log=self._params["log"],
            log_path=self._params["log_path"]
        )

    def invoke_agent(self, user_instructions, data_raw, **kwargs):
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw
        }, **kwargs)
        self.response = response
        return response
```

## Agent Architecture

### Core Components

1. **BaseAgent**: Abstract base class providing the agent interface
2. **State Graph**: LangGraph workflow that orchestrates the agent
3. **Nodes**: Individual steps in the workflow (code generation, execution, error handling)
4. **State**: Shared state passed between nodes

### Agent Workflow

Standard agent workflow:
```
User Input → Generate Code → Execute Code → Handle Errors → Return Results
                ↑                                      ↓
                └──────────── Retry on Error ─────────┘
```

### Key Classes and Functions

**From `templates`:**
- `BaseAgent`: Base class for all agents
- `create_coding_agent_graph()`: Creates standard agent workflow
- `node_func_execute_agent_code_on_data()`: Executes generated code
- `node_func_fix_agent_code()`: Handles error retry logic
- `node_func_human_review()`: Implements human-in-the-loop
- `node_func_report_agent_outputs()`: Formats final response

## Step-by-Step Guide

### Step 1: Create Agent Class

Create a new file in the appropriate category:
```
ai_data_science_team/agents/
├── data_preparation/      # For cleaning, wrangling, feature engineering
├── data_access/          # For loading data, SQL queries
├── data_analysis/        # For EDA, visualization
└── machine_learning/     # For ML model training, evaluation
```

Example: `ai_data_science_team/agents/data_analysis/my_analyzer.py`

```python
from ai_data_science_team.templates import (
    BaseAgent,
    create_coding_agent_graph,
)
import pandas as pd

class MyDataAnalyzer(BaseAgent):
    """
    Custom agent for analyzing data patterns.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use for code generation
    n_samples : int, optional
        Number of samples to show the LLM (default: 30)
    log : bool, optional
        Whether to log generated code (default: False)
    log_path : str, optional
        Directory for log files
    file_name : str, optional
        Name of the log file (default: "my_analyzer.py")
    function_name : str, optional
        Name of the generated function (default: "my_analyzer")
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="my_analyzer.py",
        function_name="my_analyzer",
        **kwargs
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            **kwargs
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create the agent's state graph."""
        return create_coding_agent_graph(
            agent_name="my_data_analyzer",
            model=self._params["model"],
            system_prompt=self._get_system_prompt(),
            n_samples=self._params["n_samples"],
            log=self._params["log"],
            log_path=self._params["log_path"],
            file_name=self._params["file_name"],
            function_name=self._params["function_name"],
        )

    def _get_system_prompt(self):
        """Define the agent's behavior."""
        return """
        You are a data analysis expert. Generate Python code to analyze data patterns.

        Your code should:
        1. Accept a pandas DataFrame as input
        2. Analyze patterns, trends, and anomalies
        3. Return a summary dictionary with findings

        Example:
        def my_analyzer(data_raw):
            import pandas as pd

            summary = {
                'rows': len(data_raw),
                'columns': len(data_raw.columns),
                'numeric_cols': data_raw.select_dtypes(include='number').columns.tolist(),
                'missing_values': data_raw.isnull().sum().to_dict(),
            }

            return summary
        """

    def invoke_agent(self, user_instructions, data_raw, max_retries=3, **kwargs):
        """
        Run the agent on the provided data.

        Parameters
        ----------
        user_instructions : str
            Natural language instructions
        data_raw : pd.DataFrame
            Input DataFrame
        max_retries : int
            Maximum retry attempts on errors

        Returns
        -------
        dict : Agent response containing results and generated code
        """
        if isinstance(data_raw, pd.DataFrame):
            data_dict = data_raw.to_dict()
        else:
            data_dict = data_raw

        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_dict,
            "max_retries": max_retries,
            "retry_count": 0,
        }, **kwargs)

        self.response = response
        return response

    def get_analysis_results(self):
        """Get analysis results from last invocation."""
        if not self.response:
            raise ValueError("Agent hasn't been invoked yet")
        return self.response.get("analysis_results", {})

    def get_function_code(self, markdown=False):
        """Get the generated Python code."""
        if not self.response:
            raise ValueError("Agent hasn't been invoked yet")

        code = self.response.get("function_code", "")
        if markdown:
            return f"```python\n{code}\n```"
        return code
```

### Step 2: Add to Category __init__.py

Update `ai_data_science_team/agents/data_analysis/__init__.py`:

```python
from ai_data_science_team.agents.data_analysis.my_analyzer import MyDataAnalyzer

__all__ = [
    # ... existing exports
    "MyDataAnalyzer",
]
```

### Step 3: Update Main Agents __init__.py

Update `ai_data_science_team/agents/__init__.py`:

```python
from ai_data_science_team.agents.data_analysis import (
    # ... existing imports
    MyDataAnalyzer,
)

__all__ = [
    # ... existing exports
    "MyDataAnalyzer",
]
```

## Registering Your Agent

### Option 1: Manual Registration

Add to `ai_data_science_team/agents/registry.py` in the `_register_all_agents()` function:

```python
def _register_all_agents():
    registry = _global_registry

    # ... existing registrations

    try:
        from ai_data_science_team.agents.data_analysis.my_analyzer import MyDataAnalyzer
        registry.register_agent(
            name="my_analyzer",
            category="data_analysis",
            description="Analyzes data patterns and trends",
            agent_class=MyDataAnalyzer,
            required_tools=["dataframe"],
            capabilities=["pattern_analysis", "trend_detection", "anomaly_detection"],
            example_usage="agent = MyDataAnalyzer(model=llm)\nagent.invoke_agent(user_instructions='Analyze patterns', data_raw=df)"
        )
    except ImportError:
        pass
```

### Option 2: Decorator Registration

```python
from ai_data_science_team.agents import register_agent

@register_agent(
    name="my_analyzer",
    category="data_analysis",
    description="Analyzes data patterns and trends",
    capabilities=["pattern_analysis", "trend_detection"]
)
class MyDataAnalyzer(BaseAgent):
    # ... implementation
```

## Examples

### Example 1: Simple Analysis Agent

```python
from ai_data_science_team.templates import BaseAgent, create_coding_agent_graph

class StatisticalAnalyzer(BaseAgent):
    """Performs statistical analysis on data."""

    def __init__(self, model, **kwargs):
        self._params = {"model": model, **kwargs}
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            agent_name="statistical_analyzer",
            model=self._params["model"],
            system_prompt="""
            Generate Python code for statistical analysis.
            Return mean, median, std, correlations.
            """,
        )

    def invoke_agent(self, user_instructions, data_raw, **kwargs):
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw
        }, **kwargs)
        self.response = response
        return response
```

### Example 2: Agent with Custom Tools

```python
from ai_data_science_team.templates import BaseAgent, create_coding_agent_graph
from ai_data_science_team.tools import get_dataframe_summary

class AdvancedAnalyzer(BaseAgent):
    """Agent with access to custom tools."""

    def __init__(self, model, custom_tools=None, **kwargs):
        self._params = {
            "model": model,
            "custom_tools": custom_tools or {},
            **kwargs
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _get_system_prompt(self):
        tools_doc = "\n".join([
            f"- {name}: {tool.__doc__}"
            for name, tool in self._params["custom_tools"].items()
        ])

        return f"""
        You have access to these tools:
        {tools_doc}

        Use these tools in your generated code.
        """

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            agent_name="advanced_analyzer",
            model=self._params["model"],
            system_prompt=self._get_system_prompt(),
        )

    def invoke_agent(self, user_instructions, data_raw, **kwargs):
        # Inject tools into execution context
        exec_globals = {
            **self._params["custom_tools"],
            'pd': pd,
        }

        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "data_raw": data_raw,
            "exec_globals": exec_globals
        }, **kwargs)

        self.response = response
        return response
```

## Best Practices

### 1. Clear System Prompts

Write detailed system prompts that explain:
- What the agent should do
- Input/output formats
- Code requirements
- Example code

### 2. Error Handling

Let the standard retry mechanism handle errors, but you can customize:

```python
def _make_compiled_graph(self):
    return create_coding_agent_graph(
        agent_name="my_agent",
        model=self._params["model"],
        system_prompt=self._get_system_prompt(),
        max_retries=5,  # Custom retry count
        bypass_explain_code=True,  # Skip code explanation
    )
```

### 3. Data Validation

Validate inputs in `invoke_agent()`:

```python
def invoke_agent(self, user_instructions, data_raw, **kwargs):
    # Validate data
    if not isinstance(data_raw, pd.DataFrame):
        raise TypeError("data_raw must be a pandas DataFrame")

    if data_raw.empty:
        raise ValueError("data_raw cannot be empty")

    # Continue with invocation
    response = self._compiled_graph.invoke({...})
    return response
```

### 4. Helper Methods

Add convenience methods for accessing results:

```python
def get_cleaned_data(self):
    """Get the cleaned DataFrame."""
    return pd.DataFrame(self.response.get("data_cleaned", {}))

def get_summary(self):
    """Get analysis summary."""
    return self.response.get("summary", {})

def get_function_code(self, markdown=False):
    """Get generated code."""
    code = self.response.get("function_code", "")
    return f"```python\n{code}\n```" if markdown else code
```

### 5. Logging

Enable logging for debugging:

```python
agent = MyAgent(
    model=llm,
    log=True,
    log_path="logs/",
    file_name="my_agent_code.py"
)
```

### 6. Testing

Test your agent thoroughly:

```python
import pytest
import pandas as pd
from ai_data_science_team import create_llm

def test_my_analyzer():
    llm = create_llm(provider="openai", model="gpt-3.5-turbo")
    agent = MyDataAnalyzer(model=llm)

    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    response = agent.invoke_agent(
        user_instructions="Analyze the data",
        data_raw=df
    )

    assert response is not None
    assert "analysis_results" in response
```

### 7. Documentation

Document your agent with:
- Docstrings for the class and methods
- Parameter descriptions
- Usage examples
- Return value specifications

## Common Patterns

### Pattern 1: Data Transformation Agent

```python
class TransformAgent(BaseAgent):
    """Transforms data based on user instructions."""

    def _get_system_prompt(self):
        return """
        Generate code that transforms the input DataFrame.
        The function should accept data_raw and return transformed_data.
        """

    def get_transformed_data(self):
        return pd.DataFrame(self.response.get("transformed_data", {}))
```

### Pattern 2: Reporting Agent

```python
class ReportingAgent(BaseAgent):
    """Generates reports from data."""

    def _get_system_prompt(self):
        return """
        Generate code that creates a comprehensive report.
        Return a dictionary with summary statistics, charts, and insights.
        """

    def get_report(self):
        return self.response.get("report", {})

    def save_report(self, output_path):
        import json
        with open(output_path, 'w') as f:
            json.dump(self.get_report(), f, indent=2)
```

### Pattern 3: Multi-Step Agent

For complex multi-step workflows, combine multiple agents:

```python
from ai_data_science_team import DataCleaningAgent, MyDataAnalyzer

class ComprehensiveAnalysisPipeline:
    def __init__(self, model):
        self.cleaner = DataCleaningAgent(model=model)
        self.analyzer = MyDataAnalyzer(model=model)

    def run(self, data_raw, clean_instructions, analyze_instructions):
        # Step 1: Clean
        self.cleaner.invoke_agent(clean_instructions, data_raw)
        cleaned_data = self.cleaner.get_data_cleaned()

        # Step 2: Analyze
        self.analyzer.invoke_agent(analyze_instructions, cleaned_data)
        results = self.analyzer.get_analysis_results()

        return results
```

## Troubleshooting

### Agent not generating correct code

- Improve system prompt clarity
- Provide more examples
- Increase n_samples to give LLM more context

### Execution errors

- Check generated code in logs
- Validate data formats
- Ensure required packages are installed

### Performance issues

- Reduce n_samples for faster processing
- Use faster LLM models (gpt-3.5-turbo, llama2)
- Cache results when possible

## Next Steps

1. Create your custom agent
2. Test thoroughly with different data
3. Add to registry for discoverability
4. Share with the community via Pull Request

See also:
- [CONFIGURATION.md](CONFIGURATION.md) - LLM setup
- [ADDING_TOOLS.md](ADDING_TOOLS.md) - Creating custom tools
