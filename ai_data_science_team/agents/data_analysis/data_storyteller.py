
# AI DATA SCIENCE TEAM
# ***
# * Agents: Data Storyteller Agent (DIST — Data Insights Storyteller)


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
    visualize_missing,
    generate_correlation_funnel,
)
from ai_data_science_team.tools.data_analysis.storytelling_tools import (
    profile_data_for_story,
    compute_dataset_kpis,
    detect_data_patterns,
    validate_analysis_integrity,
    compute_group_comparisons,
    build_story_spine,
)


AGENT_NAME = "data_storyteller_agent"

# Full DIST toolkit — EDA tools + storytelling tools
STORYTELLER_TOOLS = [
    # Profiling & EDA (foundational understanding)
    profile_data_for_story,
    explain_data,
    describe_dataset,
    visualize_missing,
    generate_correlation_funnel,
    # Storytelling-specific analysis
    compute_dataset_kpis,
    detect_data_patterns,
    validate_analysis_integrity,
    compute_group_comparisons,
    # Narrative construction
    build_story_spine,
]

DIST_SYSTEM_PROMPT = """You are the Data Insights Storyteller (DIST), an elite data scientist and narrative expert. Your job is to transform datasets into decision-grade insights and professional narratives that both technical teams and business stakeholders can trust and act on.

## Core Mission

Turn one or more datasets into actionable insight by producing:
- **Metrics & KPIs**: correct grain, denominators, eligibility rules, and validation checks
- **Rigorous EDA**: distribution + uncertainty framing, subgroup comparisons, anomaly investigation
- **Pattern discovery**: find what is surprising, non-obvious, or strategically important
- **Storytelling deliverables**: professional narratives that explain what's happening and why it matters
- **Workflow verification**: catch silent failure modes before they corrupt analysis

## Operating Principles (Non-Negotiable)

1. **Never fabricate.** If something is unknown, explicitly state it and specify the test needed.
2. **Integrity first.** Protect correctness: no silent row loss, no unintended row explosion, no unsafe coercion.
3. **Separate evidence from inference.** Label as: [Observation], [Hypothesis], [Unknown].
4. **Quantify uncertainty.** Use effect sizes, confidence intervals, robust stats.
5. **Avoid causal claims** unless an identification strategy exists. Keep claims correlational and state confounders.
6. **Two audiences always.** Produce both: executive narrative (plain language) and technical narrative (methods, assumptions, limits).
7. **Reproducibility.** Every derived metric must be traceable to a definition and data.

## Tool Workflow (Follow This Order)

1. **PROFILE FIRST** — Call `profile_data_for_story` to deeply understand the dataset before anything else.
2. **VALIDATE** — Call `validate_analysis_integrity` to catch quality issues early.
3. **ANALYZE** — Call `compute_dataset_kpis` and `detect_data_patterns` to find key metrics and patterns.
4. **COMPARE** — Call `compute_group_comparisons` if group/segment analysis is relevant.
5. **SUPPLEMENT** — Use `explain_data`, `describe_dataset`, `visualize_missing`, or `generate_correlation_funnel` for additional depth.
6. **STRUCTURE** — Call `build_story_spine` to crystallize your narrative framework.
7. **REPORT** — In your FINAL message, produce a complete structured report.

## Critical Checks (Always Perform)

- **Simpson's paradox**: Compare aggregates vs. subgroups — overall trend may contradict group trends.
- **Denominator drift**: Eligibility/filter changes can fake improvement.
- **Selection/survivorship bias**: Identify who or what is missing and why.
- **Long-tail concentration**: Pareto effects; note if top 20% drives 80%+ of a metric.
- **Drift over time**: Schema or value distribution shifts; seasonality.
- **Join artifacts**: Row explosion or orphan keys from merges.
- **Unit/time pitfalls**: % vs fraction, dollars vs cents, DST/timezone issues.

## Final Report Format (ALWAYS use this structure)

Your final response MUST be a complete, structured report in markdown with these sections:

---
# Data Story: [Descriptive Title]

## Story Spine
[Context → Tension → Insight → Implication → Action — use build_story_spine tool first]

## Executive Summary (5–10 bullets)
- Key finding 1
- Key finding 2
...

## Key Metrics
[KPI table with definition, value, caveats]

## Patterns & Discoveries
[What you found — label each as [Observation], [Hypothesis], or [Unknown]]

## Integrity & Data Quality
[Results from validate_analysis_integrity — PASS/WARN/FAIL with details]

## Methodology (Technical)
[Methods used, tools called, data transformations, assumptions]

## Methodology (Plain English)
[Non-technical explanation of what was done and why]

## Recommendations & Next Actions
1. [Priority action 1]
2. [Priority action 2]
...

## Open Questions & Uncertainties
- [What requires further investigation]
---

## Response Modes

You operate in two modes:

**Responder Mode** (for targeted questions): Provide concise, structured answers with
KPI definitions, method selection, or interpretation of a specific chart/metric.

**Report Mode** (for full analysis): Produce the complete narrative report structure above.

When the user asks for a "story", "narrative", "report", "insights", or "full analysis" —
use Report Mode. For specific questions, use Responder Mode.

## Safety & Privacy

- Flag potential PII/sensitive fields before analysis.
- Do not output raw dumps of sensitive values; use redacted examples.
- For publishing: recommend k-anonymity style checks where applicable.

You are NOT a generic chart generator. You are decision-grade, forensic, methodologically explicit, and collaborative. Your output prevents "beautiful but wrong" analytics.
"""


class DataStorytellerAgent(BaseAgent):
    """
    Data Insights Storyteller (DIST) Agent.

    An elite data analysis agent that transforms datasets into decision-grade insights
    and professional narratives. It can work alongside other agents (EDAToolsAgent,
    DataVisualizationAgent, etc.) or independently.

    The agent follows a structured workflow:
    1. **Profile** the data to understand structure and quality
    2. **Validate** data integrity to catch silent failure modes
    3. **Analyze** KPIs and patterns to find the story
    4. **Compare** groups/segments for deeper insight
    5. **Structure** findings into a story spine
    6. **Report** with a professional executive + technical narrative

    Parameters
    ----------
    model : Any
        The language model (LangChain LLM) for the agent. Opus-class models
        recommended for best storytelling quality.
    data_context : str or dict, optional
        Additional context about the data: a data dossier, data dictionary,
        schema description, or RAG retrieval result. This is injected into
        the user instructions so the agent has richer semantic understanding.
    create_react_agent_kwargs : dict, optional
        Additional kwargs passed to `create_react_agent`.
    invoke_react_agent_kwargs : dict, optional
        Additional kwargs passed to agent invocation.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Methods
    -------
    invoke_agent(user_instructions, data_raw, data_context, **kwargs)
        Synchronously runs the storyteller agent.
    ainvoke_agent(user_instructions, data_raw, data_context, **kwargs)
        Asynchronously runs the storyteller agent.
    get_story(markdown=False)
        Returns the complete story/narrative from the agent.
    get_ai_message(markdown=False)
        Returns the final AI message.
    get_internal_messages(markdown=False)
        Returns all internal messages (tool calls + responses).
    get_tool_calls()
        Returns the list of tools called during analysis.
    get_artifacts()
        Returns data artifacts from tool calls.
    get_response()
        Returns the full response dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataStorytellerAgent

    llm = ChatOpenAI(model="gpt-4o")  # Use a capable model for best results

    storyteller = DataStorytellerAgent(model=llm)

    df = pd.read_csv("data/sales.csv")

    storyteller.invoke_agent(
        user_instructions="Tell me the story of our sales performance. What's working, what isn't, and what should we focus on?",
        data_raw=df,
    )

    # Get the full story as markdown
    story = storyteller.get_story(markdown=True)

    # Get the raw response
    response = storyteller.get_response()
    ```

    With data context (data dossier or dictionary):
    ```python
    data_context = \"\"\"
    ## Data Dictionary
    - order_id: Unique order identifier
    - customer_id: Customer identifier
    - revenue: Total revenue in USD
    - product_category: One of [Electronics, Clothing, Food, Home]
    - order_date: Date of order (YYYY-MM-DD)
    - region: Sales region (North, South, East, West)
    \"\"\"

    storyteller.invoke_agent(
        user_instructions="Find patterns in sales by region and category. What story does this data tell?",
        data_raw=df,
        data_context=data_context,
    )
    ```

    Returns
    -------
    DataStorytellerAgent : BaseAgent
        A data storyteller agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model: Any,
        data_context: Optional[Any] = None,
        create_react_agent_kwargs: Optional[Dict] = {},
        invoke_react_agent_kwargs: Optional[Dict] = {},
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "data_context": data_context,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "invoke_react_agent_kwargs": invoke_react_agent_kwargs,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled state graph for the storyteller agent."""
        self.response = None
        return make_data_storyteller_agent(**self._params)

    def update_params(self, **kwargs):
        """Updates the agent's parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(
        self,
        user_instructions: str = None,
        data_raw: pd.DataFrame = None,
        data_context: Optional[Any] = None,
        **kwargs,
    ):
        """
        Synchronously runs the Data Storyteller Agent.

        Parameters
        ----------
        user_instructions : str, optional
            What story to tell, what question to answer, or what to analyze.
            Examples:
            - "Tell me the story of our sales performance"
            - "What are the most important KPIs and patterns in this customer data?"
            - "Identify hidden segments and explain what differentiates them"
            - "Verify this analysis pipeline and flag any integrity issues"
        data_raw : pd.DataFrame, optional
            The dataset to analyze as a pandas DataFrame.
        data_context : str or dict, optional
            Additional data context (data dossier, dictionary, schema, RAG context).
            Overrides the data_context set at initialization for this call.
        **kwargs
            Additional keyword arguments passed to the graph's invoke().

        Returns
        -------
        None
            Response is stored in self.response. Use getter methods to access results.
        """
        # Merge data_context: call-level overrides init-level
        effective_context = data_context if data_context is not None else self._params.get("data_context")
        context_str = _format_data_context(effective_context)
        full_instructions = _build_full_instructions(user_instructions, context_str)

        response = self._compiled_graph.invoke(
            {
                "user_instructions": full_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else {},
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_agent(
        self,
        user_instructions: str = None,
        data_raw: pd.DataFrame = None,
        data_context: Optional[Any] = None,
        **kwargs,
    ):
        """
        Asynchronously runs the Data Storyteller Agent.

        Parameters
        ----------
        user_instructions : str, optional
            What story to tell or question to answer.
        data_raw : pd.DataFrame, optional
            The dataset to analyze.
        data_context : str or dict, optional
            Additional data context (data dossier, dictionary, etc.).
        **kwargs
            Additional keyword arguments passed to the graph's ainvoke().

        Returns
        -------
        None
            Response is stored in self.response. Use getter methods to access results.
        """
        effective_context = data_context if data_context is not None else self._params.get("data_context")
        context_str = _format_data_context(effective_context)
        full_instructions = _build_full_instructions(user_instructions, context_str)

        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": full_instructions,
                "data_raw": data_raw.to_dict() if data_raw is not None else {},
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_story(self, markdown: bool = False):
        """
        Returns the complete story/narrative from the agent.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns as IPython Markdown object for notebook rendering.

        Returns
        -------
        str or IPython.display.Markdown
            The full data story/narrative.
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
        return self.get_story(markdown=markdown)

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
            parts.append(f"### {msg_type}\n\nID: {msg_id}\n\nContent:\n\n{content}")

        return Markdown("\n\n".join(parts))

    def get_artifacts(self, as_dataframe: bool = False):
        """
        Returns the artifacts collected by tool calls.

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
        artifacts = self.response.get("story_artifacts", None)
        if as_dataframe and isinstance(artifacts, dict):
            return pd.DataFrame(artifacts)
        return artifacts

    def get_tool_calls(self) -> List[str]:
        """Returns the list of tool names called during analysis."""
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

def make_data_storyteller_agent(
    model: Any,
    data_context: Optional[Any] = None,
    create_react_agent_kwargs: Optional[Dict] = {},
    invoke_react_agent_kwargs: Optional[Dict] = {},
    checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates a Data Insights Storyteller (DIST) Agent.

    The agent uses a ReAct (Reason + Act) workflow with a comprehensive toolkit
    for data profiling, KPI computation, pattern detection, integrity validation,
    and narrative construction. It produces professional reports for both technical
    and non-technical audiences.

    Parameters
    ----------
    model : Any
        The language model for tool-calling. Opus-class models recommended.
    data_context : str or dict, optional
        Default data context (data dossier, dictionary, RAG result) to inject
        into all analyses. Can be overridden per-call in invoke_agent().
    create_react_agent_kwargs : dict, optional
        Additional kwargs for `create_react_agent`.
    invoke_react_agent_kwargs : dict, optional
        Additional kwargs for agent invocation.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Returns
    -------
    app : CompiledStateGraph
        The compiled data storyteller agent.

    Examples
    --------
    ```python
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents.data_analysis.data_storyteller import make_data_storyteller_agent

    llm = ChatOpenAI(model="gpt-4o")
    agent = make_data_storyteller_agent(llm)

    response = agent.invoke({
        "user_instructions": "Tell me the key story in this customer churn data.",
        "data_raw": df.to_dict(),
    })
    ```
    """

    class GraphState(AgentState):
        internal_messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        data_raw: dict
        story_artifacts: dict
        tool_calls: list

    def storyteller_agent(state):
        print(format_agent_name(AGENT_NAME))
        print("    * RUN DATA STORYTELLER REACT AGENT")

        tool_node = ToolNode(tools=STORYTELLER_TOOLS)

        # Build the agent with DIST system prompt
        agent_kwargs = dict(create_react_agent_kwargs or {})
        agent_kwargs["state_modifier"] = DIST_SYSTEM_PROMPT

        dist_agent = create_react_agent(
            model,
            tools=tool_node,
            state_schema=GraphState,
            checkpointer=checkpointer,
            **agent_kwargs,
        )

        response = dist_agent.invoke(
            {
                "messages": [("user", state["user_instructions"])],
                "data_raw": state.get("data_raw", {}),
            },
            invoke_react_agent_kwargs or {},
        )

        print("    * POST-PROCESSING STORY RESULTS")

        internal_messages = response.get("messages", [])
        if not internal_messages:
            return {"internal_messages": [], "story_artifacts": None, "tool_calls": []}

        last_ai_message = AIMessage(internal_messages[-1].content, role=AGENT_NAME)

        # Collect the last artifact from tool results
        last_tool_artifact = None
        for msg in reversed(internal_messages):
            if hasattr(msg, "artifact") and msg.artifact is not None:
                last_tool_artifact = msg.artifact
                break
            if isinstance(msg, dict) and "artifact" in msg:
                last_tool_artifact = msg["artifact"]
                break

        tool_calls = get_tool_call_names(internal_messages)

        return {
            "messages": [last_ai_message],
            "internal_messages": internal_messages,
            "story_artifacts": last_tool_artifact,
            "tool_calls": tool_calls,
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("storyteller_agent", storyteller_agent)
    workflow.add_edge(START, "storyteller_agent")
    workflow.add_edge("storyteller_agent", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_data_context(data_context: Optional[Any]) -> str:
    """Convert data_context (str, dict, or None) to a clean string."""
    if data_context is None:
        return ""
    if isinstance(data_context, str):
        return data_context.strip()
    if isinstance(data_context, dict):
        import json
        try:
            return json.dumps(data_context, indent=2, default=str)
        except Exception:
            return str(data_context)
    return str(data_context)


def _build_full_instructions(user_instructions: Optional[str], context_str: str) -> str:
    """Combine user instructions with optional data context."""
    base = user_instructions or "Analyze this dataset and tell me the story it represents."
    if context_str:
        return f"""{base}

---
## Data Context (Dossier / Dictionary / Schema)

{context_str}
---
"""
    return base
