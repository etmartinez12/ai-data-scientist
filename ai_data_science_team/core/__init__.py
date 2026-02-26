"""
Core infrastructure for AI Data Science Team.

This module provides foundational components including:
- LLM provider factory for multi-provider support
- Configuration management
- Base agent classes
"""

from ai_data_science_team.core.llm_provider import create_llm, get_available_providers
from ai_data_science_team.core.config import get_config, LLMConfig

__all__ = [
    "create_llm",
    "get_available_providers",
    "get_config",
    "LLMConfig",
]
