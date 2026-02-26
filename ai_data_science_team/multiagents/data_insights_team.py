"""
DataInsightsTeam: Multi-Agent Pipeline (DDEA -> DIST)

A sequential two-agent pipeline that combines the Data Domain Expert Agent (DDEA)
and the Data Insights Storyteller (DIST) for authoritative, narrative-ready data analysis.

Pipeline:
    START -> run_ddea (profile + dossier) -> run_dist (story + report) -> END

DDEA runs first to produce the authoritative DATA_DOSSIER.md. That dossier is then
automatically injected as data_context for DIST, so the storyteller has complete
domain knowledge before beginning its analysis.
"""

from typing import Annotated, Optional, Sequence, Union
import operator

import pandas as pd
from IPython.display import Markdown

from langchain_core.messages import BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from typing import TypedDict

from ai_data_science_team.templates import BaseAgent
from ai_data_science_team.agents.data_analysis.data_domain_expert import (
    DataDomainExpertAgent,
)
from ai_data_science_team.agents.data_analysis.data_storyteller import (
    DataStorytellerAgent,
    _build_full_instructions,
)


AGENT_NAME = "data_insights_team"


class DataInsightsTeam(BaseAgent):
    """
    DataInsightsTeam: Multi-Agent Pipeline (DDEA + DIST).

    Combines the Data Domain Expert Agent (DDEA) and the Data Insights Storyteller
    (DIST) in a sequential pipeline for complete, authoritative data analysis.

    The pipeline runs in two stages:
    1. **DDEA** profiles the dataset exhaustively and generates a DATA_DOSSIER.md —
       the authoritative source of truth about the data's structure, quality,
       edge cases, and semantic meaning.
    2. **DIST** receives the dossier as data context and produces a decision-grade
       narrative report: KPIs, pattern discoveries, storytelling, and recommendations.

    This approach ensures DIST never makes assumptions about data types, grain, or
    semantics — it works from DDEA's verified findings.

    Parameters
    ----------
    ddea_agent : DataDomainExpertAgent
        The Data Domain Expert Agent for dataset profiling.
    dist_agent : DataStorytellerAgent
        The Data Insights Storyteller for narrative report generation.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Methods
    -------
    invoke_agent(user_instructions, data_raw, secondary_data_raw, ddea_instructions, **kwargs)
        Synchronously runs the full DDEA -> DIST pipeline.
    ainvoke_agent(user_instructions, data_raw, secondary_data_raw, ddea_instructions, **kwargs)
        Asynchronously runs the full DDEA -> DIST pipeline.
    get_dossier(markdown=False)
        Returns the DATA_DOSSIER.md produced by DDEA.
    get_story(markdown=False)
        Returns the narrative report produced by DIST.
    get_ai_message(markdown=False)
        Returns the final AI message (DIST's report).
    get_ddea_artifacts()
        Returns DDEA's structured profiling artifacts.
    get_ddea_tool_calls()
        Returns the list of DDEA tools called.
    get_dist_tool_calls()
        Returns the list of DIST tools called.
    get_response()
        Returns the full pipeline response dictionary.
    show()
        Displays the multi-agent mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import DataDomainExpertAgent, DataStorytellerAgent
    from ai_data_science_team.multiagents import DataInsightsTeam

    llm = ChatOpenAI(model="gpt-4o")

    ddea = DataDomainExpertAgent(model=llm)
    dist = DataStorytellerAgent(model=llm)

    team = DataInsightsTeam(ddea_agent=ddea, dist_agent=dist)

    df = pd.read_csv("data/sales.csv")

    team.invoke_agent(
        user_instructions="Tell me the story of our sales performance. What KPIs matter most and what should we focus on?",
        data_raw=df,
    )

    # Get the authoritative dossier
    dossier = team.get_dossier(markdown=True)

    # Get the narrative story
    story = team.get_story(markdown=True)
    ```

    With separate instructions for each stage:
    ```python
    team.invoke_agent(
        ddea_instructions="Profile this dataset and pay special attention to the revenue and region columns. Generate a complete DATA_DOSSIER.md.",
        user_instructions="Find the most important sales insights and recommendations for the executive team.",
        data_raw=df,
    )
    ```

    Returns
    -------
    DataInsightsTeam : BaseAgent
        A multi-agent pipeline implemented as a compiled state graph.
    """

    def __init__(
        self,
        ddea_agent: DataDomainExpertAgent,
        dist_agent: DataStorytellerAgent,
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "ddea_agent": ddea_agent,
            "dist_agent": dist_agent,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled multi-agent pipeline."""
        self.response = None
        return make_data_insights_team(
            ddea_agent=self._params["ddea_agent"]._compiled_graph,
            dist_agent=self._params["dist_agent"]._compiled_graph,
            checkpointer=self._params["checkpointer"],
        )

    def update_params(self, **kwargs):
        """Updates parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(
        self,
        user_instructions: str = None,
        data_raw: Union[pd.DataFrame, dict] = None,
        secondary_data_raw: Optional[Union[pd.DataFrame, dict]] = None,
        ddea_instructions: Optional[str] = None,
        **kwargs,
    ):
        """
        Synchronously runs the full DDEA -> DIST pipeline.

        Parameters
        ----------
        user_instructions : str, optional
            The storytelling/analysis question for DIST.
            Examples:
            - "Tell me the key story in this customer dataset"
            - "What are the most important KPIs and patterns? Make recommendations."
            - "Identify the most significant business insights"
        data_raw : pd.DataFrame or dict, optional
            The primary dataset to analyze.
        secondary_data_raw : pd.DataFrame or dict, optional
            A secondary dataset for cross-dataset join analysis (DDEA phase).
        ddea_instructions : str, optional
            Specific profiling instructions for DDEA. If None, defaults to a
            comprehensive full-profile request.
            Examples:
            - "Profile this dataset completely and generate a DATA_DOSSIER.md"
            - "Focus on the customer_id and revenue columns. Check for PII."
        **kwargs
            Additional keyword arguments passed to the graph's invoke().

        Returns
        -------
        None
            Response stored in self.response. Use getter methods.
        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions or "Analyze this dataset and tell me the most important story.",
                "ddea_instructions": ddea_instructions or "Profile this dataset completely and generate a comprehensive DATA_DOSSIER.md.",
                "data_raw": _to_dict(data_raw),
                "secondary_data_raw": _to_dict(secondary_data_raw),
            },
            **kwargs,
        )
        self.response = response
        return None

    async def ainvoke_agent(
        self,
        user_instructions: str = None,
        data_raw: Union[pd.DataFrame, dict] = None,
        secondary_data_raw: Optional[Union[pd.DataFrame, dict]] = None,
        ddea_instructions: Optional[str] = None,
        **kwargs,
    ):
        """
        Asynchronously runs the full DDEA -> DIST pipeline.

        Parameters
        ----------
        user_instructions : str, optional
            The storytelling/analysis question for DIST.
        data_raw : pd.DataFrame or dict, optional
            The primary dataset to analyze.
        secondary_data_raw : pd.DataFrame or dict, optional
            Secondary dataset for cross-dataset join analysis.
        ddea_instructions : str, optional
            Specific profiling instructions for DDEA.
        **kwargs
            Additional kwargs passed to the graph's ainvoke().

        Returns
        -------
        None
            Response stored in self.response. Use getter methods.
        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions or "Analyze this dataset and tell me the most important story.",
                "ddea_instructions": ddea_instructions or "Profile this dataset completely and generate a comprehensive DATA_DOSSIER.md.",
                "data_raw": _to_dict(data_raw),
                "secondary_data_raw": _to_dict(secondary_data_raw),
            },
            **kwargs,
        )
        self.response = response
        return None

    def get_dossier(self, markdown: bool = False):
        """
        Returns the DATA_DOSSIER.md produced by DDEA.

        This is the authoritative dataset reference document containing schema,
        column profiles, edge cases, validation rules, and cleaning plan.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns as IPython Markdown for notebook rendering.

        Returns
        -------
        str or IPython.display.Markdown
        """
        if not self.response:
            return None
        content = self.response.get("data_dossier", "")
        if markdown:
            return Markdown(content) if content else None
        return content

    def get_story(self, markdown: bool = False):
        """
        Returns the narrative report produced by DIST.

        This is the complete data story: executive summary, KPIs, patterns,
        integrity checks, methodology, and recommendations.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns as IPython Markdown.

        Returns
        -------
        str or IPython.display.Markdown
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
        """Returns the final AI message (DIST's narrative report)."""
        return self.get_story(markdown=markdown)

    def get_ddea_artifacts(self):
        """
        Returns DDEA's structured profiling artifacts.

        The artifacts dictionary contains findings from each profiling tool:
        schema, column profiles, edge cases, validation rules, cleaning plan,
        ML readiness assessment, and the dossier content.

        Returns
        -------
        dict or None
        """
        if not self.response:
            return None
        return self.response.get("ddea_artifacts", None)

    def get_ddea_tool_calls(self):
        """Returns the list of DDEA tools called during profiling."""
        if self.response:
            return self.response.get("ddea_tool_calls", [])
        return []

    def get_dist_tool_calls(self):
        """Returns the list of DIST tools called during storytelling."""
        if self.response:
            return self.response.get("dist_tool_calls", [])
        return []

    def get_response(self):
        """Returns the full pipeline response dictionary."""
        return self.response

    def show(self):
        """Displays the multi-agent pipeline mermaid diagram."""
        return self._compiled_graph.show()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_data_insights_team(
    ddea_agent: CompiledStateGraph,
    dist_agent: CompiledStateGraph,
    checkpointer: Optional[Checkpointer] = None,
):
    """
    Creates the DataInsightsTeam multi-agent pipeline.

    Wires DDEA and DIST in a sequential pipeline:
    START -> run_ddea -> run_dist -> END

    DDEA's DATA_DOSSIER.md is automatically injected as DIST's data_context,
    ensuring the storyteller has complete domain knowledge from the profiler.

    Parameters
    ----------
    ddea_agent : CompiledStateGraph
        The compiled Data Domain Expert Agent graph.
    dist_agent : CompiledStateGraph
        The compiled Data Storyteller Agent graph.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for state persistence.

    Returns
    -------
    app : CompiledStateGraph
        The compiled DataInsightsTeam pipeline.
    """

    class TeamState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        ddea_instructions: str
        data_raw: dict
        secondary_data_raw: dict
        # DDEA outputs
        data_dossier: str
        ddea_artifacts: dict
        ddea_tool_calls: list
        # DIST outputs
        story_artifacts: dict
        dist_tool_calls: list

    def run_ddea(state: TeamState):
        """Run the Data Domain Expert Agent to profile the dataset."""
        print("---DATA INSIGHTS TEAM---")
        print("************************")
        print("---STAGE 1: DATA DOMAIN EXPERT (DDEA)---")

        response = ddea_agent.invoke(
            {
                "user_instructions": state.get("ddea_instructions", "Profile this dataset completely and generate a comprehensive DATA_DOSSIER.md."),
                "data_raw": state.get("data_raw", {}),
                "secondary_data_raw": state.get("secondary_data_raw", {}),
            }
        )

        # Extract DDEA outputs
        dossier = response.get("data_dossier", "")
        ddea_artifacts = response.get("ddea_artifacts", None)
        ddea_tool_calls = response.get("tool_calls", [])
        messages = response.get("messages", [])

        print("    * DDEA complete. Dossier length: %d chars" % len(dossier))

        return {
            "messages": messages,
            "data_dossier": dossier,
            "ddea_artifacts": ddea_artifacts,
            "ddea_tool_calls": ddea_tool_calls,
        }

    def run_dist(state: TeamState):
        """Run the Data Insights Storyteller with the DDEA dossier as context."""
        print("---STAGE 2: DATA INSIGHTS STORYTELLER (DIST)---")

        # Inject the DDEA dossier as DIST's data context
        dossier = state.get("data_dossier", "")
        user_instructions = state.get("user_instructions", "Analyze this dataset and tell me the most important story.")

        # Use the DIST helper to combine instructions with dossier context
        full_instructions = _build_full_instructions(user_instructions, dossier)

        response = dist_agent.invoke(
            {
                "user_instructions": full_instructions,
                "data_raw": state.get("data_raw", {}),
            }
        )

        messages = response.get("messages", [])
        story_artifacts = response.get("story_artifacts", None)
        dist_tool_calls = response.get("tool_calls", [])

        print("    * DIST complete. Story generated.")
        print("---DATA INSIGHTS TEAM COMPLETE---")

        return {
            "messages": messages,
            "story_artifacts": story_artifacts,
            "dist_tool_calls": dist_tool_calls,
        }

    workflow = StateGraph(TeamState)
    workflow.add_node("run_ddea", run_ddea)
    workflow.add_node("run_dist", run_dist)

    workflow.add_edge(START, "run_ddea")
    workflow.add_edge("run_ddea", "run_dist")
    workflow.add_edge("run_dist", END)

    app = workflow.compile(
        checkpointer=checkpointer,
        name=AGENT_NAME,
    )

    return app


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_dict(data: Optional[Union[pd.DataFrame, dict]]) -> dict:
    """Convert DataFrame or dict to dict for state passing."""
    if data is None:
        return {}
    if isinstance(data, pd.DataFrame):
        return data.to_dict()
    if isinstance(data, dict):
        return data
    return {}
