"""
Base classes for tools in the AI Data Science Team.

Tools are reusable functions that agents can use to perform specific tasks.
All tools should inherit from BaseTool for consistency and discoverability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ToolMetadata:
    """
    Metadata for a tool.

    Attributes:
        name: Tool name (unique identifier)
        category: Tool category (e.g., "data_loading", "data_analysis", "ml")
        description: Brief description of what the tool does
        parameters: List of parameter names
        returns: Description of return value
        examples: List of usage examples
    """
    name: str
    category: str
    description: str
    parameters: List[str]
    returns: str
    examples: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples or [],
        }


class BaseTool(ABC):
    """
    Abstract base class for tools.

    All tools should inherit from this class and implement:
    - get_metadata(): Return tool metadata
    - execute(): Perform the tool's action
    """

    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """
        Get tool metadata.

        Returns:
            ToolMetadata instance
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        pass

    def __repr__(self) -> str:
        metadata = self.get_metadata()
        return f"<{self.__class__.__name__}: {metadata.name} ({metadata.category})>"


class FunctionTool(BaseTool):
    """
    Wrapper for function-based tools.

    Wraps a regular function to conform to the BaseTool interface.
    """

    def __init__(
        self,
        func: callable,
        name: str,
        category: str,
        description: str,
        parameters: List[str],
        returns: str,
        examples: Optional[List[str]] = None
    ):
        self.func = func
        self._metadata = ToolMetadata(
            name=name,
            category=category,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=examples
        )

    def get_metadata(self) -> ToolMetadata:
        return self._metadata

    def execute(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


def tool(
    name: str,
    category: str,
    description: str,
    parameters: List[str],
    returns: str,
    examples: Optional[List[str]] = None
):
    """
    Decorator to register a function as a tool.

    Args:
        name: Tool name
        category: Tool category
        description: Tool description
        parameters: List of parameter names
        returns: Description of return value
        examples: Usage examples

    Example:
        @tool(
            name="calculate_mean",
            category="statistics",
            description="Calculate the mean of a series",
            parameters=["data"],
            returns="Mean value",
            examples=["mean = calculate_mean(df['age'])"]
        )
        def calculate_mean(data):
            return data.mean()
    """
    def decorator(func):
        func._tool_metadata = ToolMetadata(
            name=name,
            category=category,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=examples
        )
        func._is_tool = True
        return func
    return decorator
