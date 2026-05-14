<div align="center">
  <h1>AI Data Science Team</h1>
  <em>AI-powered data science agents for automated data analysis and machine learning</em>
</div>


# AI Data Science Team

**AI-powered data science agents for automated data analysis, feature engineering, and machine learning workflows.**

*Beta - This Python library is under active development. There may be breaking changes that occur until release of 0.1.0.*

---

The AI Data Science Team of Copilots includes Agents that specialize data cleaning, preparation, feature engineering, modeling (machine learning), and interpretation of various business problems like:

- Churn Modeling
- Employee Attrition
- Lead Scoring
- Insurance Risk
- Credit Card Risk
- And more

## Table of Contents

- [Your AI Data Science Team (An Army Of Agents)](#your-ai-data-science-team-an-army-of-agents)
  - [Table of Contents](#table-of-contents)
  - [Companies That Want A Custom AI Data Science Team (And AI Apps)](#companies-that-want-a-custom-ai-data-science-team-and-ai-apps)
  - [Generative AI for Data Scientists Workshop](#generative-ai-for-data-scientists-workshop)
  - [Data Science Agents](#data-science-agents)
    - [Data Science Apps](#data-science-apps)
    - [Multi-Agents](#multi-agents)
      - [Agentic Applications](#agentic-applications)
    - [Agents Available Now](#agents-available-now)
      - [Standard Agents](#standard-agents)
      - [Machine Learning Agents](#machine-learning-agents)
      - [Data Science Agents](#data-science-agents-1)
      - [Multi-Agents](#multi-agents)
    - [Agents Coming Soon](#agents-coming-soon)
  - [Disclaimer](#disclaimer)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Example: H2O Machine Learning Agent](#example-h2o-machine-learning-agent)
  - [Contributing](#contributing)
  - [License](#license)



## Features

- **Multi-LLM Support**: Works with OpenAI, Ollama (local models), Anthropic Claude, Azure, Google, and custom endpoints
- **Specialized Agents**: Purpose-built agents for data cleaning, wrangling, visualization, EDA, and machine learning
- **Tool Registry**: Discoverable registry system for easy extension with custom tools and agents
- **Reproducible Workflows**: Agents generate Python code that can be inspected, logged, and reused
- **Multi-Agent Orchestration**: Combine multiple agents for complex workflows

## Data Science Agents

This project is a work in progress. Additional data science agents will be released soon. 

### Data Science Apps

**Open Pandas AI Data Analyst:** Load an Excel or CSV file and ask it questions. Get data and charts back.

**SQL Database Agent:** Connects any SQL Database, generates SQL queries from natural language, and returns data as a downloadable table.

**Exploratory Data Copilot:** An AI-powered data science app that performs automated exploratory data analysis (EDA) with EDA Reporting, Missing Data Analysis, Correlation Analysis, and more.

[See all available apps here](/apps)

### Multi-Agents

**Pandas Data Analyst Agent:** Combines the ability to wrangle, transform, and analyze data with an optional data visualization agent that can create interactive plots.


#### Agentic Applications

1. **Exploratory Data Copilot**: An AI-powered data science app that performs automated exploratory data analysis (EDA) with EDA Reporting, Missing Data Analysis, Correlation Analysis, and more. [See Application](/apps/exploratory-copilot-app/)

2. **SQL Database Agent App:** Connects any SQL Database, generates SQL queries from natural language, and returns data as a downloadable table. [See Application](/apps/sql-database-agent-app/)

### Agents Available Now

#### Standard Agents

1. **Data Wrangling Agent:** Merges, joins, and transforms data into analysis-ready format.
2. **Data Visualization Agent:** Creates interactive visualizations using Plotly. Returns JSON-serializable plotly figures.
3. **Data Cleaning Agent:** Handles missing values, outliers, duplicates, and data type conversions.
4. **Feature Engineering Agent:** Creates and transforms features for machine learning models.
5. **SQL Database Agent:** Connects to SQL databases, generates queries, and extracts data.
6. **Data Loader Tools Agent:** Loads data from CSV, Excel, Parquet, and Pickle files.


#### Machine Learning Agents

1. **H2O Machine Learning Agent:** Builds and trains ML models using H2O AutoML for classification and regression.
2. **MLflow Tools Agent:** Manages ML experiments, models, and artifacts with MLflow. 11+ tools for MLOps workflows.

#### Data Science Agents

1. **EDA Tools Agent:** Performs automated exploratory data analysis with reporting, missing data analysis, and correlation analysis.


#### Multi-Agents

1. **Pandas Data Analyst Agent:** Combines data wrangling and visualization capabilities for comprehensive data analysis.
2. **SQL Data Analyst Agent:** Combines SQL database access with visualization for complete database analytics workflows.

### Agents Coming Soon

1. **Data Analyst:** Analyzes data structure, creates exploratory visualizations, and performs correlation analysis to identify relationships.
2. **Interpretability Agent:** Performs Interpretable ML to explain why the model returned predictions including which features were the most important to the model.
3. **Supervisor:** Forms task list. Moderates sub-agents. Returns completed assignment. 

## Disclaimer

**This project is provided "as-is" without warranties or guarantees.**

- No warranties or guarantees provided
- Use at your own risk
- Always validate agent-generated code before execution in production environments
- Test thoroughly with your specific use cases

By using this software, you agree to these terms.

## Installation

### Basic Installation

Install the core package via PyPI (note: beta version, breaking changes may occur until 0.1.0):

```bash
pip install ai-data-science-team
```

### LLM Provider Support

Choose your LLM provider(s):

```bash
# OpenAI (default)
pip install "ai-data-science-team[openai]"

# Ollama (local models)
pip install "ai-data-science-team[ollama]"

# Anthropic Claude
pip install "ai-data-science-team[anthropic]"

# All providers
pip install "ai-data-science-team[all_providers]"

# Machine Learning tools (H2O, MLflow)
pip install "ai-data-science-team[machine_learning]"

# Everything
pip install "ai-data-science-team[all]"
```

### From Source

```bash
git clone https://github.com/etmartinez12/ai-data-scientist.git
cd ai-data-science-team
pip install -e ".[all]"
```

## Usage

### Quick Start

```python
from ai_data_science_team import create_llm, DataCleaningAgent
import pandas as pd

# Option 1: Create LLM from environment variables (set LLM_PROVIDER, LLM_MODEL, etc.)
llm = create_llm()

# Option 2: Explicitly specify provider
llm = create_llm(provider="openai", model="gpt-4o-mini", api_key="sk-...")
# or
llm = create_llm(provider="ollama", model="llama2")  # Local model
# or
llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Load your data
df = pd.read_csv("data.csv")

# Create and run an agent
agent = DataCleaningAgent(model=llm, log=True, log_path="logs/")
agent.invoke_agent(
    user_instructions="Remove duplicates and handle missing values",
    data_raw=df
)

# Get cleaned data
cleaned_df = agent.get_data_cleaned()
```

### Example: H2O Machine Learning Agent

```python
from ai_data_science_team import create_llm, H2OMLAgent
import pandas as pd

# Initialize LLM
llm = create_llm(provider="openai", model="gpt-4o-mini")

# Load data
df = pd.read_csv("data/churn_data.csv")

# Create H2O ML Agent
ml_agent = H2OMLAgent(
    model=llm,
    log=True,
    log_path="logs/",
    model_directory="h2o_models/",
    enable_mlflow=True,
)

# Train models
ml_agent.invoke_agent(
    data_raw=df.drop(columns=["customerID"]),
    user_instructions="Build classification model for 'Churn'. Max runtime 30 seconds.",
    target_variable="Churn"
)

# Get leaderboard
leaderboard = ml_agent.get_leaderboard()
```

### Configuration

Create a `.env` file in your project root:

```bash
# LLM Configuration
LLM_PROVIDER=openai  # or ollama, anthropic, azure, google, custom
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here

# For Ollama (local)
# LLM_PROVIDER=ollama
# LLM_MODEL=llama2
# LLM_BASE_URL=http://localhost:11434

# For Anthropic
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-20241022
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Then use `create_llm()` without parameters to auto-load from environment.

## Contributing

Contributions are welcome.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow, and pull request guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Security

To report a vulnerability, please follow [SECURITY.md](SECURITY.md).
