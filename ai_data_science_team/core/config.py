"""
Configuration Management

Handles configuration loading from:
1. Environment variables (.env file)
2. YAML configuration files
3. Default values

Priority: Environment variables > YAML config > Defaults
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class LLMConfig:
    """
    LLM Configuration container.

    Attributes:
        provider: LLM provider name
        model: Model name
        api_key: API key for the provider
        base_url: Base URL for Ollama or custom endpoints
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        additional_params: Provider-specific additional parameters
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **additional_params
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_params = additional_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        config.update(self.additional_params)
        return {k: v for k, v in config.items() if v is not None}

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LLMConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            LLMConfig instance
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "YAML support requires 'pyyaml' package. "
                "Install with: pip install pyyaml"
            )

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        llm_config = config.get('llm', {})
        return cls(
            provider=llm_config.get('provider', 'openai'),
            model=llm_config.get('model', 'gpt-4o-mini'),
            api_key=llm_config.get('api_key'),
            base_url=llm_config.get('base_url'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens'),
            **{k: v for k, v in llm_config.items()
               if k not in ['provider', 'model', 'api_key', 'base_url', 'temperature', 'max_tokens']}
        )

    def __repr__(self) -> str:
        # Mask API key in repr
        api_key_repr = f"{self.api_key[:8]}..." if self.api_key else None
        return (
            f"LLMConfig(provider='{self.provider}', model='{self.model}', "
            f"api_key='{api_key_repr}', base_url='{self.base_url}', "
            f"temperature={self.temperature}, max_tokens={self.max_tokens})"
        )


def load_dotenv_file(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file. If None, searches for .env in:
                  1. Current directory
                  2. Project root (ai_data_science_team package directory)
                  3. User home directory

    Returns:
        True if .env file was loaded, False otherwise
    """
    if not DOTENV_AVAILABLE:
        return False

    if env_file:
        return load_dotenv(env_file)

    # Search for .env file in common locations
    search_paths = [
        Path.cwd() / ".env",  # Current directory
        Path(__file__).parent.parent.parent / ".env",  # Project root
        Path.home() / ".env",  # Home directory
    ]

    for path in search_paths:
        if path.exists():
            load_dotenv(path)
            return True

    return False


def get_config(
    config_file: Optional[str] = None,
    env_file: Optional[str] = None,
    auto_load_env: bool = True
) -> LLMConfig:
    """
    Load configuration with priority: env vars > YAML config > defaults.

    Args:
        config_file: Path to YAML configuration file
        env_file: Path to .env file
        auto_load_env: Automatically search for and load .env file

    Returns:
        LLMConfig instance

    Examples:
        # Auto-load from .env and environment variables
        config = get_config()

        # Load from specific YAML file
        config = get_config(config_file="config.yaml")

        # Load from specific .env file
        config = get_config(env_file="production.env")
    """
    # Load .env file if requested
    if auto_load_env:
        load_dotenv_file(env_file)

    # Start with environment variables (highest priority)
    config = LLMConfig.from_env()

    # Override with YAML config if provided
    if config_file and YAML_AVAILABLE:
        yaml_config = LLMConfig.from_yaml(config_file)

        # Only override if env var was not set
        if not os.getenv("LLM_PROVIDER"):
            config.provider = yaml_config.provider
        if not os.getenv("LLM_MODEL"):
            config.model = yaml_config.model
        if not os.getenv("LLM_API_KEY"):
            config.api_key = yaml_config.api_key
        if not os.getenv("LLM_BASE_URL"):
            config.base_url = yaml_config.base_url
        if not os.getenv("LLM_TEMPERATURE"):
            config.temperature = yaml_config.temperature
        if not os.getenv("LLM_MAX_TOKENS"):
            config.max_tokens = yaml_config.max_tokens

        # Merge additional params
        config.additional_params.update(yaml_config.additional_params)

    return config


def get_provider_models(provider: str) -> Dict[str, Any]:
    """
    Get available models and configuration for a provider.

    Args:
        provider: Provider name

    Returns:
        Dictionary with provider information including available models
    """
    providers_info = {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "requires_api_key": True,
            "default_model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
        "ollama": {
            "name": "Ollama (Local)",
            "models": ["llama2", "llama3", "mistral", "mixtral", "phi", "codellama", "deepseek-coder"],
            "requires_api_key": False,
            "default_model": "llama2",
            "default_base_url": "http://localhost:11434",
        },
        "anthropic": {
            "name": "Anthropic Claude",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "requires_api_key": True,
            "default_model": "claude-3-5-sonnet-20241022",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
        "azure": {
            "name": "Azure OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "requires_api_key": True,
            "default_model": "gpt-4o-mini",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "requires_endpoint": True,
            "endpoint_env": "AZURE_OPENAI_ENDPOINT",
        },
        "google": {
            "name": "Google Generative AI",
            "models": ["gemini-pro", "gemini-pro-vision"],
            "requires_api_key": True,
            "default_model": "gemini-pro",
            "api_key_env": "GOOGLE_API_KEY",
        },
        "custom": {
            "name": "Custom Endpoint (OpenAI-compatible)",
            "models": [],  # User-defined
            "requires_api_key": False,
            "requires_base_url": True,
        },
    }

    return providers_info.get(provider, {})
