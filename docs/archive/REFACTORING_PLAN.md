# AI Data Science Team - Comprehensive Refactoring Plan

## Context

The ai-data-science-team package is a functional LangChain/LangGraph-based system for AI-powered data science workflows. However, it needs significant improvements:

1. **Branding Issues**: Contains references to original author (Matt Dancho) and business-science.io throughout 9+ files
2. **Limited LLM Support**: Hardcoded to OpenAI only, no support for Ollama or custom endpoints
3. **Poor Tool Organization**: No registry system, tools scattered across files, tight coupling between agents and tools
4. **Limited Extensibility**: Manual registration required for new agents/tools, no discovery mechanism
5. **Inconsistent Documentation**: Examples only show OpenAI usage, no patterns for other providers

**User Requirements:**
- Remove all branding (Matt Dancho, business-science, AI Bootcamp)
- Support multiple LLM backends: OpenAI, Ollama, and custom endpoints
- Restructure to follow agent + skills pattern (like Claude Code)
- Make architecture simple, well-documented, and extensible
- Easy to add new agents and tools
- Update author to: Elias Tommy Martinez (martinez.elias12@gmail.com)

## Goals

1. **Rebrand**: Remove all external branding, update metadata
2. **Multi-LLM Support**: Create provider abstraction supporting OpenAI, Ollama, and custom endpoints
3. **Tool Registry**: Implement discoverable registry system for tools and agents
4. **Better Architecture**: Organize code into clear, logical structure
5. **Comprehensive Documentation**: Update all docs, examples, and inline comments
6. **Maintain Backward Compatibility**: Keep existing agent APIs working

## Architecture Changes

### New Directory Structure

```
ai_data_science_team/
├── core/                           # NEW: Core abstractions
│   ├── __init__.py
│   ├── llm_provider.py            # NEW: LLM provider factory
│   ├── config.py                  # NEW: Configuration management
│   └── base_agent.py              # MOVED: From templates/
├── agents/                         # EXISTING: Refactored
│   ├── __init__.py                # Updated exports + registry
│   ├── registry.py                # NEW: Agent registry
│   ├── data_preparation/          # NEW: Organized by category
│   │   ├── __init__.py
│   │   ├── data_cleaning.py
│   │   ├── data_wrangling.py
│   │   └── feature_engineering.py
│   ├── data_access/               # NEW: Organized by category
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── sql_database.py
│   ├── data_analysis/             # NEW: Organized by category
│   │   ├── __init__.py
│   │   ├── data_visualization.py
│   │   └── eda.py
│   └── machine_learning/          # NEW: Organized by category
│       ├── __init__.py
│       ├── h2o_ml.py
│       └── mlflow_tools.py
├── multiagents/                    # EXISTING: Minimal changes
│   ├── __init__.py
│   ├── pandas_data_analyst.py
│   └── sql_data_analyst.py
├── tools/                          # EXISTING: Major refactor
│   ├── __init__.py                # Updated with all exports
│   ├── registry.py                # NEW: Tool registry
│   ├── base.py                    # NEW: Tool base classes
│   ├── data_loading/              # NEW: Organized by category
│   │   ├── __init__.py
│   │   └── loaders.py
│   ├── data_analysis/             # NEW: Organized by category
│   │   ├── __init__.py
│   │   └── eda_tools.py
│   ├── database/                  # NEW: Organized by category
│   │   ├── __init__.py
│   │   └── sql_tools.py
│   └── ml/                        # NEW: Organized by category
│       ├── __init__.py
│       ├── h2o_tools.py
│       └── mlflow_tools.py
├── templates/                      # EXISTING: Simplified
│   ├── __init__.py
│   └── node_functions.py          # Renamed from agent_templates.py
├── utils/                          # EXISTING: Keep as-is
├── parsers/                        # EXISTING: Keep as-is
├── config/                         # NEW: Configuration files
│   ├── llm_providers.yaml         # NEW: LLM provider configs
│   └── example.env                # NEW: Example environment file
└── __init__.py                     # Updated main exports
```

### LLM Provider Architecture

**New File: `core/llm_provider.py`**

Provides factory pattern for creating LLM instances:
- `create_llm(provider, model, **kwargs)` - Main factory function
- Support for: OpenAI, Anthropic, Ollama, Azure, Google, Custom endpoints
- Reads from environment variables and config file
- Falls back to current behavior (pass LLM instance) for backward compatibility

**New File: `core/config.py`**

Configuration management using Pydantic:
- Read from `.env` file
- Load `config/llm_providers.yaml`
- Validate configuration
- Provide defaults

**Environment Variables:**
```bash
LLM_PROVIDER=openai              # or ollama, anthropic, azure, custom
LLM_MODEL=gpt-4o-mini            # provider-specific model name
LLM_API_KEY=sk-...               # API key for provider
LLM_BASE_URL=http://localhost:11434  # for Ollama or custom endpoints
LLM_TEMPERATURE=0.7              # optional parameters
LLM_MAX_TOKENS=4096              # optional parameters
```

### Tool Registry Architecture

**New File: `tools/registry.py`**

- `ToolRegistry` class to manage all tools
- Auto-discovery of tools in tools/ subdirectories
- Metadata for each tool (name, category, description, parameters)
- Query interface: `get_tools_by_category()`, `get_tool()`, `list_all_tools()`

**New File: `tools/base.py`**

- `BaseTool` abstract class
- Standardized interface for all tools
- Consistent response format handling

### Agent Registry Architecture

**New File: `agents/registry.py`**

- `AgentRegistry` class to manage all agents
- Metadata for each agent (name, category, description, required tools)
- Query interface: `get_agents_by_category()`, `get_agent()`, `list_all_agents()`

## Critical Files to Modify

### Phase 1: Core Infrastructure (Foundation)

1. **Create: `core/llm_provider.py`**
   - Factory function for LLM creation
   - Support OpenAI, Ollama, Anthropic, Azure, Google, custom endpoints
   - Auto-detect provider from env vars

2. **Create: `core/config.py`**
   - Pydantic settings for configuration management
   - Read from .env and YAML files
   - Validation and defaults

3. **Create: `config/llm_providers.yaml`**
   - Provider-specific configuration templates
   - Model lists per provider
   - Default parameters

4. **Create: `config/example.env`**
   - Example environment file with all variables documented
   - Safe to commit (no actual keys)

5. **Move: `templates/agent_templates.py` → `core/base_agent.py`**
   - Extract BaseAgent class to core
   - Keep node functions in templates/node_functions.py

### Phase 2: Tool Reorganization

6. **Create: `tools/registry.py`**
   - ToolRegistry class implementation
   - Auto-discovery from subdirectories
   - Metadata storage

7. **Create: `tools/base.py`**
   - BaseTool abstract class
   - Standardized interfaces

8. **Reorganize: `tools/*.py` → `tools/*/`**
   - Move `data_loader.py` → `data_loading/loaders.py`
   - Move `eda.py` → `data_analysis/eda_tools.py`
   - Move `sql.py` → `database/sql_tools.py`
   - Move `h2o.py` → `ml/h2o_tools.py`
   - Move `mlflow.py` → `ml/mlflow_tools.py`
   - Keep `dataframe.py` as shared utilities

9. **Update: `tools/__init__.py`**
   - Export all tools (currently empty!)
   - Export registry functions
   - Make tools easily importable

### Phase 3: Agent Reorganization

10. **Create: `agents/registry.py`**
    - AgentRegistry class implementation
    - Agent metadata storage

11. **Reorganize: `agents/*.py` → `agents/*/`**
    - `data_cleaning_agent.py` → `data_preparation/data_cleaning.py`
    - `data_wrangling_agent.py` → `data_preparation/data_wrangling.py`
    - `feature_engineering_agent.py` → `data_preparation/feature_engineering.py`
    - `data_loader_tools_agent.py` → `data_access/data_loader.py`
    - `sql_database_agent.py` → `data_access/sql_database.py`
    - `data_visualization_agent.py` → `data_analysis/data_visualization.py`
    - Move `ds_agents/eda_tools_agent.py` → `agents/data_analysis/eda.py`
    - Move `ml_agents/*.py` → `agents/machine_learning/`

12. **Update: `agents/__init__.py`**
    - Import from new structure
    - Export registry
    - Maintain backward compatibility

13. **Delete: `ds_agents/` and `ml_agents/` directories**
    - Consolidate into `agents/` with categories

### Phase 4: Rebrand & Update Metadata

14. **Update: `setup.py`**
    - Change author to: Elias Tommy Martinez
   - Change email to: martinez.elias12@gmail.com
    - Update URL (or remove if no new GitHub repo)
    - Add new dependencies: `python-dotenv`, `pyyaml`, `langchain-ollama`
    - Make OpenAI optional: `extras_require`

15. **Update: `README.md`**
    - Remove all business-science references
    - Remove Matt Dancho references
    - Update installation instructions
    - Add multi-LLM provider examples
    - Add Ollama setup instructions
    - Update examples to show provider selection

16. **Update: Agent files (9 files)**
    - Remove business-science GitHub URLs from docstrings
    - Update example code in docstrings
    - Keep functionality unchanged

17. **Update: Streamlit apps (3 files)**
    - Remove BUSINESS SCIENCE header comments
    - Add provider selector UI
    - Support multiple LLM backends
    - Update installation instructions

18. **Update: `ai_data_science_team/orchestration.py`**
    - Remove business-science header comment

### Phase 5: Documentation & Examples

19. **Update: `CLAUDE.md`**
    - Document new architecture
    - Explain LLM provider system
    - Show tool/agent registry usage
    - Update file structure documentation

20. **Update: `requirements.txt`**
    - Add: `python-dotenv`
    - Add: `pyyaml`
    - Add: `langchain-ollama` (or make optional)
    - Consider making `langchain_openai` optional

21. **Create: `docs/CONFIGURATION.md`**
    - Comprehensive guide to LLM configuration
    - Examples for each provider
    - Troubleshooting guide

22. **Create: `docs/ADDING_AGENTS.md`**
    - Guide for adding new agents
    - Step-by-step instructions
    - Example code

23. **Create: `docs/ADDING_TOOLS.md`**
    - Guide for adding new tools
    - Tool interface documentation
    - Example code

## Implementation Steps

### Step 1: Create Core Infrastructure (No Breaking Changes)

1. Create `core/` directory
2. Create `core/llm_provider.py` with factory pattern
3. Create `core/config.py` with settings management
4. Create `config/` directory with example files
5. Add `python-dotenv`, `pyyaml` to requirements.txt
6. Test LLM provider factory independently

**Verification**: Run test script that creates LLMs for OpenAI, Ollama, and custom endpoints

### Step 2: Implement Tool Registry (Additive)

1. Create `tools/registry.py`
2. Create `tools/base.py`
3. Update `tools/__init__.py` to export all tools
4. Test registry with existing tools (no reorganization yet)

**Verification**: Import and query tool registry, verify all tools are discoverable

### Step 3: Reorganize Tools (Structural Change)

1. Create category subdirectories in `tools/`
2. Move tool files to new locations
3. Update imports in `tools/__init__.py`
4. Update imports in agent files
5. Update tool registration in registry
6. Test all agents still work

**Verification**: Run example notebooks to ensure agents can still find tools

### Step 4: Implement Agent Registry (Additive)

1. Create `agents/registry.py`
2. Register all existing agents
3. Update `agents/__init__.py` to export registry
4. Test registry queries

**Verification**: Query agent registry and verify metadata

### Step 5: Reorganize Agents (Structural Change)

1. Create category subdirectories in `agents/`
2. Move agent files to new locations
3. Update imports in `agents/__init__.py`
4. Update imports in `multiagents/`
5. Delete `ds_agents/` and `ml_agents/` directories
6. Update imports in main `__init__.py`

**Verification**: Run all example notebooks to ensure imports work

### Step 6: Rebrand All Files

1. Update `setup.py` metadata
2. Update `README.md` - remove all branding
3. Update agent docstrings (9 files)
4. Update app header comments (3 files)
5. Update orchestration.py header

**Verification**: Grep for "Matt Dancho", "business-science", "mdancho" - should find nothing

### Step 7: Update Agents for Multi-LLM Support

1. Update example notebooks to show provider usage
2. Update Streamlit apps with provider selectors
3. Add LLM provider examples to docstrings
4. Test with OpenAI, Ollama, and custom endpoints

**Verification**: Run agents with each provider type

### Step 8: Documentation

1. Update `CLAUDE.md`
2. Create `docs/CONFIGURATION.md`
3. Create `docs/ADDING_AGENTS.md`
4. Create `docs/ADDING_TOOLS.md`
5. Update example notebooks with provider configuration

**Verification**: Follow docs to configure different providers

## Backward Compatibility Strategy

To maintain backward compatibility:

1. **Keep existing imports working**: Old imports should still work via `__init__.py` re-exports
   ```python
   # Old: from ai_data_science_team.agents import DataCleaningAgent
   # Still works via:
   from ai_data_science_team.agents.data_preparation.data_cleaning import DataCleaningAgent as _DataCleaningAgent
   DataCleaningAgent = _DataCleaningAgent
   ```

2. **LLM provider is optional**: If user passes `model` parameter (LLM instance), use it directly
   ```python
   # Old way still works
   llm = ChatOpenAI(model="gpt-4o-mini")
   agent = DataCleaningAgent(model=llm)

   # New way (optional)
   agent = DataCleaningAgent()  # Uses env vars/config
   ```

3. **Config is optional**: If no `.env` or config file, fall back to requiring explicit model parameter

## Files to Delete

1. `ds_agents/` directory (move contents to `agents/data_analysis/`)
2. `ml_agents/` directory (move contents to `agents/machine_learning/`)
3. Any `.DS_Store` files (Mac artifacts)

## New Dependencies

Add to `requirements.txt`:
- `python-dotenv` - Environment variable management
- `pyyaml` - YAML configuration parsing
- `langchain-ollama` - Ollama support (or make optional)

Update `setup.py` extras_require:
```python
extras_require={
    "openai": ["langchain_openai", "openai"],
    "ollama": ["langchain_ollama"],
    "anthropic": ["langchain_anthropic"],
    "all_providers": ["langchain_openai", "openai", "langchain_ollama", "langchain_anthropic"],
    "machine_learning": ["h2o", "mlflow"],
    "data_science": ["pytimetk", "missingno", "sweetviz"],
    "all": ["h2o", "mlflow", "pytimetk", "missingno", "sweetviz", "langchain_openai", "openai"],
}
```

## Testing & Verification Plan

### End-to-End Verification

1. **Test Tool Registry**
   ```python
   from ai_data_science_team.tools import get_tool_registry
   registry = get_tool_registry()
   tools = registry.list_tools()
   assert len(tools) > 0
   ```

2. **Test Agent Registry**
   ```python
   from ai_data_science_team.agents import get_agent_registry
   registry = get_agent_registry()
   agents = registry.list_agents()
   assert len(agents) == 11  # All agents present
   ```

3. **Test LLM Providers**
   ```python
   from ai_data_science_team.core import create_llm

   # OpenAI
   llm = create_llm(provider="openai", model="gpt-4o-mini", api_key="...")

   # Ollama
   llm = create_llm(provider="ollama", model="llama2", base_url="http://localhost:11434")

   # Custom
   llm = create_llm(provider="custom", model="custom-model", base_url="https://api.internal.com", api_key="...")
   ```

4. **Test Backward Compatibility**
   ```python
   # Old imports still work
   from ai_data_science_team.agents import DataCleaningAgent
   from ai_data_science_team.ml_agents import H2OMLAgent  # Should fail gracefully with helpful message
   ```

5. **Test Agent Functionality**
   ```python
   # Run data cleaning agent with OpenAI
   agent = DataCleaningAgent(model=create_llm("openai", "gpt-4o-mini"))
   agent.invoke_agent(user_instructions="...", data_raw=df)

   # Run with Ollama
   agent = DataCleaningAgent(model=create_llm("ollama", "llama2"))
   agent.invoke_agent(user_instructions="...", data_raw=df)
   ```

6. **Test Streamlit Apps**
   - Run `pandas-data-analyst-app`
   - Select different providers
   - Verify functionality

7. **Test Example Notebooks**
   - Run all notebooks in `examples/`
   - Verify no broken imports
   - Test with different providers

8. **Verify Branding Removed**
   ```bash
   grep -r "Matt Dancho" --exclude-dir=.git
   grep -r "business-science" --exclude-dir=.git
   grep -r "mdancho" --exclude-dir=.git
   # Should return no results
   ```

## Key Design Decisions

1. **Keep package name**: User chose to keep `ai-data-science-team` despite rebranding
2. **Keep all agents**: Not removing any functionality, all 11 agents maintained
3. **Config + env vars**: Using both `.env` files and YAML config for flexibility
4. **Categorize agents**: Group by function (data_preparation, data_access, data_analysis, machine_learning)
5. **Backward compatibility**: Old imports still work via re-exports
6. **Optional providers**: LLM providers are extras_require, not hard dependencies

## Success Criteria

- ✅ All branding removed (Matt Dancho, business-science)
- ✅ OpenAI, Ollama, and custom endpoints all working
- ✅ Tool registry discoverable and queryable
- ✅ Agent registry discoverable and queryable
- ✅ All existing agents still functional
- ✅ Backward compatible imports working
- ✅ Example notebooks updated and tested
- ✅ Streamlit apps support multiple providers
- ✅ Documentation comprehensive and accurate
- ✅ Easy to add new agents/tools (documented process)
