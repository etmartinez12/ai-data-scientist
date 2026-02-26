"""
Tool Registry

Manages registration and discovery of all tools in the system.
Provides query interfaces for finding tools by category, name, or other criteria.
"""

from typing import Dict, List, Optional, Any, Callable
from ai_data_science_team.tools.base import ToolMetadata, FunctionTool
import inspect
import importlib
import pkgutil
from pathlib import Path


class ToolRegistry:
    """
    Central registry for all tools.

    Manages tool registration, discovery, and querying.
    """

    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._tool_functions: Dict[str, Callable] = {}
        self._categories: Dict[str, List[str]] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        category: str,
        description: str,
        parameters: List[str],
        returns: str,
        examples: Optional[List[str]] = None
    ):
        """
        Register a tool.

        Args:
            name: Tool name (unique identifier)
            func: Tool function
            category: Tool category
            description: Tool description
            parameters: List of parameter names
            returns: Description of return value
            examples: Usage examples
        """
        metadata = ToolMetadata(
            name=name,
            category=category,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=examples
        )

        self._tools[name] = metadata
        self._tool_functions[name] = func

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

    def register_from_function(self, func: Callable):
        """
        Register a tool from a function with _tool_metadata attribute.

        Args:
            func: Function with _tool_metadata attribute
        """
        if not hasattr(func, '_tool_metadata'):
            raise ValueError(f"Function {func.__name__} is not decorated as a tool")

        metadata = func._tool_metadata
        self.register_tool(
            name=metadata.name,
            func=func,
            category=metadata.category,
            description=metadata.description,
            parameters=metadata.parameters,
            returns=metadata.returns,
            examples=metadata.examples
        )

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get tool function by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        return self._tool_functions.get(name)

    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool metadata by name.

        Args:
            name: Tool name

        Returns:
            ToolMetadata or None if not found
        """
        return self._tools.get(name)

    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get all tool names in a category.

        Args:
            category: Category name

        Returns:
            List of tool names
        """
        return self._categories.get(category, [])

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_categories(self) -> List[str]:
        """
        List all tool categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def search_tools(self, query: str) -> List[str]:
        """
        Search tools by name or description.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching tool names
        """
        query = query.lower()
        matches = []

        for name, metadata in self._tools.items():
            if (query in name.lower() or
                query in metadata.description.lower() or
                query in metadata.category.lower()):
                matches.append(name)

        return matches

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all tools.

        Returns:
            Dictionary mapping tool names to metadata dictionaries
        """
        return {
            name: metadata.to_dict()
            for name, metadata in self._tools.items()
        }

    def auto_discover_tools(self, package_path: Optional[str] = None):
        """
        Auto-discover tools in the tools package.

        Scans all modules in the tools package and registers functions
        with _tool_metadata attribute.

        Args:
            package_path: Path to tools package (defaults to current package)
        """
        if package_path is None:
            # Default to current package's parent
            package_path = str(Path(__file__).parent)

        # Import all modules in the package
        package_name = "ai_data_science_team.tools"

        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return

        # Iterate through all modules in the package
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[package_path],
            prefix=f"{package_name}.",
            onerror=lambda x: None
        ):
            if ispkg or modname.endswith(('.base', '.registry', '__init__')):
                continue

            try:
                module = importlib.import_module(modname)

                # Find all functions with _is_tool attribute
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if hasattr(obj, '_is_tool') and obj._is_tool:
                        self.register_from_function(obj)

            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"<ToolRegistry: {len(self._tools)} tools in {len(self._categories)} categories>"


# Global registry instance
_global_registry = None


def get_tool_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        # Auto-discover tools on first access
        _global_registry.auto_discover_tools()
    return _global_registry


def register_tool(
    name: str,
    category: str,
    description: str,
    parameters: List[str],
    returns: str,
    examples: Optional[List[str]] = None
):
    """
    Decorator to register a tool in the global registry.

    Args:
        name: Tool name
        category: Tool category
        description: Tool description
        parameters: List of parameter names
        returns: Description of return value
        examples: Usage examples

    Example:
        @register_tool(
            name="calculate_mean",
            category="statistics",
            description="Calculate the mean of a series",
            parameters=["data"],
            returns="Mean value"
        )
        def calculate_mean(data):
            return data.mean()
    """
    def decorator(func):
        registry = get_tool_registry()
        registry.register_tool(
            name=name,
            func=func,
            category=category,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=examples
        )
        # Add metadata to function for introspection
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
