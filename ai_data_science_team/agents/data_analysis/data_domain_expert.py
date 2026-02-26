
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Domain Expert Agent (DDEA — Data Domain Expert Agent)


from typing import Any, Optional, Annotated, Sequence, Dict, List
import operator
import pandas as pd

from IPython.display import Markdown

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph
from langgraph.types import Checkpointer

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.utils.regex import format_agent_name
from ai_data_science_team.utils.messages import get_tool_call_names

from ai_data_science_team.tools.data_analysis.eda_tools import (
    explain_data,
    describe_dataset,
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


AGENT_NAME = "data_domain_expert_agent"

# Full DDEA toolkit — domain expert tools + foundational EDA tools
DDEA_TOOLS = [
    # Phase 1-2: Schema + deep profiling (always first)
    profile_dataset_schema,
    profile_columns_deep,
    # Phase 3: Edge cases and nuances
    detect_edge_cases,
    # Phase 4: Relationships and join analysis
    analyze_join_keys,
    # Phase 5: Validation and quality rules
    generate_validation_rules,
    # Phase 6: Cleaning plan + ML readiness
    plan_cleaning_steps,
    assess_ml_readiness,
    # Dossier compilation (final step)
    generate_data_dossier,
    # Foundational EDA tools for additional depth
    explain_data,
    describe_dataset,
]

DDEA_SYSTEM_PROMPT = """You are the Data Domain Expert Agent (DDEA), the authoritative source of truth about datasets. Your job is to exhaustively profile, document, and answer questions about any dataset with the rigor of a senior data engineer and the domain intuition of an expert analyst.

## Core Mission

Produce an authoritative DATA_DOSSIER.md and answer structured A2A (Agent-to-Agent) queries about datasets. Your outputs are consumed by downstream agents (like DIST — the Data Insights Storyteller), pipelines, and human engineers who need to trust your findings completely.

## Six-Phase Profiling Workflow

When asked to profile or understand a dataset, follow these phases in order:

**Phase 1 — Inventory & Schema**
- Call `profile_dataset_schema` FIRST on every new dataset
- Document all column names, inferred types, confidence levels, cardinality tiers
- Flag type ambiguities, potential key columns, and sentinel null representations

**Phase 2 — Deep Column Profiling**
- Call `profile_columns_deep` to compute full distributional statistics
- Capture: missing %, cardinality, value distributions, string patterns (email/UUID/phone/date), numeric ranges, top/bottom values, PII flags

**Phase 3 — Edge Cases & Nuances**
- Call `detect_edge_cases` to find sentinel nulls, mixed boolean styles, whitespace corruption, PII fields, multi-value delimited strings, float-stored integers
- Document prevalence, examples, and handling strategy for each finding

**Phase 4 — Relationships & Join Map**
- Call `analyze_join_keys` to assess key quality (uniqueness, null rate, coverage)
- For cross-dataset work: assess join multiplicity, match rates, orphan records, and recommend join strategy (INNER/LEFT/FULL OUTER)

**Phase 5 — Validation Rules**
- Call `generate_validation_rules` to produce a VAL-### catalog
- Cover: NOT NULL, RANGE, UNIQUE, ALLOWED VALUES, FORMAT, DATE RANGE constraints

**Phase 6 — Cleaning Plan & ML Readiness**
- Call `plan_cleaning_steps` for an ordered, reversible cleaning plan
- Call `assess_ml_readiness` if the dataset will be used for machine learning
- Always order cleaning steps: structural fixes -> null/sentinel handling -> standardization

**Final: Compile Dossier**
- Call `generate_data_dossier` with summaries from all phases to produce DATA_DOSSIER.md

## The 11 A2A Intents

When invoked by another agent or with a specific intent, respond precisely to the requested intent:

1. **dataset_overview** — Inventory, shape, format, row counts, memory size, file structure
2. **schema_and_types** — All columns with inferred types, confidence, nullable status, ambiguities
3. **column_profile** — Per-column distributional statistics, patterns, anomalies
4. **edge_cases_and_nuances** — Sentinel nulls, encoding issues, mixed types, PII, multi-value fields
5. **relationships_and_joins** — Key quality, FK/PK relationships, join recommendations, multiplicity
6. **validation_rules** — Complete VAL-### rule catalog with check expressions
7. **cleaning_and_canonicalization_plan** — Ordered cleaning steps with rationale, reversibility, risk
8. **ml_readiness_report** — Target analysis, leakage risks, feature quality, encoding recommendations
9. **workflow_verification** — Audit a provided pipeline/workflow for correctness and edge-case handling
10. **workflow_monitoring_plan** — Design automated quality checks and monitors for a pipeline
11. **generate_or_update_dossier** — Compile or refresh the full DATA_DOSSIER.md

## A2A Response Format

When responding to a specific intent, structure your response as:

```
## DDEA Response: [INTENT]

**Request ID:** [request_id]
**Dataset:** [dataset_name]
**Confidence:** [HIGH / MEDIUM / LOW]

### Summary
[1-3 sentence direct answer]

### Details
[Structured findings specific to the intent]

### Evidence
- [What was checked / computed]

### Recommended Actions
1. [Actionable next step]

### Risks & Pitfalls
- [RISK] [What could go wrong]

### Open Questions
- [ ] [What needs further investigation]
```

## Operating Principles (Non-Negotiable)

1. **Evidence over assertion.** Every finding must cite a count, percentage, or example.
2. **Silent failures are the enemy.** Explicitly flag any assumption that, if wrong, could corrupt downstream analysis.
3. **Grain integrity.** Always state the unit of analysis (one row = one what?).
4. **Distinguish:** [Observation] (measured) vs [Hypothesis] (inferred) vs [Unknown] (untested).
5. **Multiplicity discipline.** For joins: always compute and report the join multiplicity before recommending.
6. **PII caution.** Flag potential PII before profiling. Do not output raw dumps of sensitive values.
7. **Cleaning reversibility.** Every cleaning step should be annotated as reversible or irreversible.
8. **ML leakage vigilance.** For any column that could encode the target label, flag it explicitly.

## Critical Checks (Always Perform)

- **Sentinel nulls**: -1, 0, 999, 'N/A', 'None', 'null', '', ' ' — do not treat as valid values
- **Boolean heterogeneity**: 0/1, True/False, 'yes'/'no', 'Y'/'N' mixed in same column
- **Whitespace corruption**: Leading/trailing spaces that defeat joins and groupbys
- **Float-stored integers**: IDs or counts stored as float64 due to missing values (NaN propagation)
- **Multi-value strings**: Pipe-delimited, comma-separated, or semicolon-joined values in single cells
- **Type drift**: Columns that are mostly numeric but contain occasional string values
- **Date ambiguity**: MM/DD vs DD/MM formats; timezone-naive vs timezone-aware
- **Cardinality explosion**: String columns with very high cardinality that may be IDs

## Dossier First Principle

You maintain the DATA_DOSSIER.md as the single source of truth. Every finding you produce contributes to this document. When asked to generate or update the dossier, compile ALL findings from your profiling phases into the 12-section format:

1. Dataset Inventory
2. Schema & Types
3. Data Dictionary
4. Column Profiles
5. Relationships & Join Map
6. Edge Cases & Nuances
7. Validation Rules
8. Cleaning & Canonicalization Plan
9. ML Readiness Notes
10. Workflow Assurance Notes
11. Open Questions & Uncertainties
12. Change Log

You are NOT a general-purpose chatbot. You are a forensic data profiler and authoritative domain documenter. Your findings are trusted by downstream ML engineers, analysts, and automated pipelines. Precision and completeness are your primary obligations.
"""


class DataDomainExpertAgent(BaseAgent):
    """
    Data Domain Expert Agent (DDEA).

    The authoritative source of truth about datasets. Exhaustively profiles,
    documents, and answers structured questions about any dataset. Works as a
    standalone agent or as the first stage of the DataInsightsTeam pipeline
    (feeding its DATA_DOSSIER.md to the DataStorytellerAgent).

    The agent follows the six-phase DDEA profiling workflow:
    1. **Schema** — Infer all column types, cardinality, key candidates
    2. **Deep Profile** — Per-column statistics, patterns, PII detection
    3. **Edge Cases** — Sentinels, mixed booleans, whitespace, multi-value fields
    4. **Relationships** — Key quality, FK/PK, join recommendations
    5. **Validation** — VAL-### rule catalog
    6. **Cleaning + ML** — Ordered cleaning plan, ML readiness assessment

    It also supports 11 structured A2A (Agent-to-Agent) intents for precise,
    programmatic consumption by other agents.

    Parameters
    ----------
    model : Any
        The language model (LangChain LLM). Capable models (Opus, GPT-4o) recommended
        for comprehensive profiling.
    create_react_agent_kwargs : dict, optional
        Additional kwargs passed to `create_react_agent`.
    invoke_react_agent_kwargs : dict, optional
        Additional kwargs passed to agent invocation.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Methods
    -------
    invoke_agent(user_instructions, data_raw, secondary_data_raw, **kwargs)
        Synchronously runs the domain expert agent.
    ainvoke_agent(user_instructions, data_raw, secondary_data_raw, **kwargs)
        Asynchronously runs the domain expert agent.
    get_dossier(markdown=False)
        Returns the DATA_DOSSIER.md content produced by the agent.
    get_ai_message(markdown=False)
        Returns the final AI message.
    get_internal_messages(markdown=False)
        Returns all internal messages (tool calls + responses).
    get_artifacts(as_dataframe=False)
        Returns data artifacts from tool calls.
    get_tool_calls()
        Returns the list of tools called during analysis.
    get_response()
        Returns the full response dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataDomainExpertAgent

    llm = ChatOpenAI(model="gpt-4o")

    ddea = DataDomainExpertAgent(model=llm)

    df = pd.read_csv("data/customers.csv")

    # Full profiling — generate DATA_DOSSIER.md
    ddea.invoke_agent(
        user_instructions="Profile this dataset completely and generate a DATA_DOSSIER.md",
        data_raw=df,
    )

    # Get the dossier as markdown
    dossier = ddea.get_dossier(markdown=True)
    ```

    Cross-dataset join analysis:
    ```python
    orders_df = pd.read_csv("data/orders.csv")
    customers_df = pd.read_csv("data/customers.csv")

    ddea.invoke_agent(
        user_instructions="Analyze the join between orders and customers on customer_id. What is the join multiplicity and match rate?",
        data_raw=orders_df,
        secondary_data_raw=customers_df,
    )
    ```

    A2A intent invocation:
    ```python
    ddea.invoke_agent(
        user_instructions="Intent: schema_and_types. Request ID: req-001. Dataset: customer_data",
        data_raw=df,
    )
    ```

    Returns
    -------
    DataDomainExpertAgent : BaseAgent
        A data domain expert agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model: Any,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled state graph for the domain expert agent."""
        self.response = None
        return make_data_domain_expert_agent(**self._params)

    def update_params(self, **kwargs):
        """Updates the agent's parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(
        self,
        user_instructions: str = None,
        data_raw: pd.DataFrame = None,
        secondary_data_raw: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        Synchronously runs the Data Domain Expert Agent.

        Parameters
        ----------
        user_instructions : str, optional
            What to profile, what intent to fulfill, or what question to answer.
            Examples:
            - "Profile this dataset completely and generate a DATA_DOSSIER.md"
            - "Intent: schema_and_types. What are all the column types?"
            - "Analyze the join between this dataset and the secondary dataset on customer_id"
            - "Assess ML readiness for predicting churn (target: churned)"
            - "Generate validation rules for this dataset"
        data_raw : pd.DataFrame, optional
            The primary dataset to profile as a pandas DataFrame.
        secondary_data_raw : pd.DataFrame, optional
            A secondary dataset for cross-dataset join analysis. Required when
            using the `analyze_join_keys` tool for cross-dataset multiplicity checks.
        **kwargs
            Additional keyword arguments passed to the graph's invoke().

        Returns
        -------
        None
            Response is stored in self.response. Use getter methods to access results.
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions or "Profile this dataset completely.",
                "data_raw": data_raw.to_dict() if data_raw is not None else {},
                "secondary_data_raw": secondary_data_raw.to_dict() if secondary_data_raw is not None else {},
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_agent(
        self,
        user_instructions: str = None,
        data_raw: pd.DataFrame = None,
        secondary_data_raw: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        Asynchronously runs the Data Domain Expert Agent.

        Parameters
        ----------
        user_instructions : str, optional
            What to profile or question to answer.
        data_raw : pd.DataFrame, optional
            The primary dataset to analyze.
        secondary_data_raw : pd.DataFrame, optional
            Secondary dataset for cross-dataset join analysis.
        **kwargs
            Additional keyword arguments passed to the graph's ainvoke().

        Returns
        -------
        None
            Response is stored in self.response. Use getter methods to access results.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions or "Profile this dataset completely.",
                "data_raw": data_raw.to_dict() if data_raw is not None else {},
                "secondary_data_raw": secondary_data_raw.to_dict() if secondary_data_raw is not None else {},
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_dossier(self, markdown: bool = False):
        """
        Returns the DATA_DOSSIER.md content produced by the agent.

        This is the primary artifact of the DDEA — the authoritative dataset
        reference document. If the agent was asked to generate a full dossier,
        this returns the complete 12-section DATA_DOSSIER.md content.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns as IPython Markdown object for notebook rendering.

        Returns
        -------
        str or IPython.display.Markdown
            The DATA_DOSSIER.md content, or the final AI message if no explicit
            dossier artifact was produced.
        """
        if not self.response:
            return None

        # Check for explicit dossier artifact from generate_data_dossier tool
        artifacts = self.response.get("ddea_artifacts", None)
        if isinstance(artifacts, dict) and "data_dossier" in artifacts:
            content = artifacts["data_dossier"]
            if markdown:
                return Markdown(content)
            return content

        # Fall back to the final AI message
        messages = self.response.get("messages", [])
        if messages:
            content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            if markdown:
                return Markdown(content)
            return content
        return None

    def get_ai_message(self, markdown: bool = False):
        """
        Returns the final AI message from the agent.

        Parameters
        ----------
        markdown : bool
            If True, returns as IPython Markdown.

        Returns
        -------
        str or Markdown
        """
        if not self.response:
            return None
        messages = self.response.get("messages", [])
        if messages:
            content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            if markdown:
                return Markdown(content)
            return content
        return None

    def get_internal_messages(self, markdown: bool = False):
        """
        Returns all internal messages (tool calls + AI responses).

        Parameters
        ----------
        markdown : bool
            If True, returns formatted markdown string.

        Returns
        -------
        list or str
        """
        if not self.response:
            return None

        msgs = self.response.get("internal_messages", [])
        if not markdown:
            return msgs

        parts = []
        for msg in msgs:
            msg_type = getattr(msg, "type", "unknown").upper()
            msg_id = getattr(msg, "id", "")
            content = getattr(msg, "content", str(msg))
            parts.append("### %s\n\nID: %s\n\nContent:\n\n%s" % (msg_type, msg_id, content))

        return Markdown("\n\n".join(parts))

    def get_artifacts(self, as_dataframe: bool = False):
        """
        Returns the artifacts collected by tool calls.

        The DDEA artifacts dictionary contains the structured findings from each
        profiling tool: schema, column profiles, edge cases, validation rules,
        cleaning plan, ML readiness assessment, and the dossier itself.

        Parameters
        ----------
        as_dataframe : bool
            If True, attempts to return as DataFrame.

        Returns
        -------
        dict or pd.DataFrame
        """
        if not self.response:
            return None
        artifacts = self.response.get("ddea_artifacts", None)
        if as_dataframe and isinstance(artifacts, dict):
            return pd.DataFrame(artifacts)
        return artifacts

    def get_tool_calls(self) -> List[str]:
        """Returns the list of tool names called during profiling."""
        if self.response:
            return self.response.get("tool_calls", [])
        return []

    def get_response(self) -> Optional[Dict]:
        """Returns the full agent response dictionary."""
        return self.response

    def show(self):
        """Displays the agent's mermaid diagram."""
        return self._compiled_graph.show()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_data_domain_expert_agent(
    model: Any,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates a Data Domain Expert Agent (DDEA).

    The agent uses a ReAct (Reason + Act) workflow with a comprehensive toolkit
    for exhaustive dataset profiling, documentation, validation rule generation,
    ML readiness assessment, and DATA_DOSSIER.md compilation.

    Supports cross-dataset join analysis via the `secondary_data_raw` state field,
    which is injected into the `analyze_join_keys` tool automatically.

    Parameters
    ----------
    model : Any
        The language model for tool-calling. Capable models (GPT-4o, Claude Opus)
        recommended for thorough profiling.
    create_react_agent_kwargs : dict, optional
        Additional kwargs for `create_react_agent`.
    invoke_react_agent_kwargs : dict, optional
        Additional kwargs for agent invocation.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Returns
    -------
    app : CompiledStateGraph
        The compiled data domain expert agent.

    Examples
    --------
    ```python
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents.data_analysis.data_domain_expert import make_data_domain_expert_agent

    llm = ChatOpenAI(model="gpt-4o")
    agent = make_data_domain_expert_agent(llm)

    response = agent.invoke({
        "user_instructions": "Profile this customer dataset and generate a DATA_DOSSIER.md",
        "data_raw": df.to_dict(),
        "secondary_data_raw": {},
    })
    ```
    """

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        secondary_data_raw: dict
        data_dossier: str
        ddea_artifacts: dict
        tool_calls: list

    def domain_expert_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    * RUN DATA DOMAIN EXPERT REACT AGENT")

        tool_node = ToolNode(tools=DDEA_TOOLS)

        # Build the agent with DDEA system prompt
        agent_kwargs = dict(create_react_agent_kwargs or {})
        agent_kwargs["state_modifier"] = DDEA_SYSTEM_PROMPT

        ddea_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **agent_kwargs,
        )

        response = ddea_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
                "data_raw": state.get("data_raw", {}),
                "secondary_data_raw": state.get("secondary_data_raw", {}),
            },
            invoke_react_agent_kwargs or {},
        )

        print("    * POST-PROCESSING DDEA RESULTS")

        internal_messages = response.get("messages", [])
        if not internal_messages:
            return {
                "internal_messages": [],
                "data_dossier": "",
                "ddea_artifacts": None,
                "tool_calls": [],
            }

        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        # Collect all artifacts from tool results — DDEA accumulates findings
        # The last artifact from generate_data_dossier will be most complete
        ddea_artifacts = {}
        data_dossier = ""
        for msg in internal_messages:
            artifact = None
            if hasattr(msg, "artifact") and msg.artifact is not None:
                artifact = msg.artifact
            elif isinstance(msg, dict) and "artifact" in msg:
                artifact = msg["artifact"]

            if isinstance(artifact, dict):
                # Merge all artifacts; later tools' findings overwrite earlier ones for same keys
                ddea_artifacts.update(artifact)
                # Extract the dossier specifically if produced
                if "data_dossier" in artifact:
                    data_dossier = artifact["data_dossier"]

        # If no explicit dossier was generated, use the final AI message
        if not data_dossier and internal_messages:
            data_dossier = internal_messages[-1].content if hasattr(internal_messages[-1], "content") else ""

        tool_calls = get_tool_call_names(internal_messages)

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "data_dossier": data_dossier,
            "ddea_artifacts": ddea_artifacts if ddea_artifacts else None,
            "tool_calls": tool_calls,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("domain_expert_agent", domain_expert_agent)
    workflow.add_edge(START, "domain_expert_agent")
    workflow.add_edge("domain_expert_agent", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app
