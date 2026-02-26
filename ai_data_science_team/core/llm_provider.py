"""
LLM Provider Factory

Provides a unified interface for creating LLM instances from multiple providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Ollama (local models: llama2, mistral, etc.)
- Anthropic (Claude models)
- Azure OpenAI
- Google (Gemini, PaLM)
- Custom endpoints (compatible with OpenAI API)

Usage:
    from ai_data_science_team.core import create_llm

    # Using environment variables
    llm = create_llm()

    # Explicit provider
    llm = create_llm(provider="openai", model="gpt-4o-mini", api_key="sk-...")
    llm = create_llm(provider="ollama", model="llama2", base_url="http://localhost:11434")
    llm = create_llm(provider="custom", model="custom-model", base_url="https://api.internal.com")
"""

import os
from typing import Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel


def get_available_providers() -> Dict[str, bool]:
    """
    Check which LLM providers are available based on installed packages.

    Returns:
        Dict mapping provider names to availability status
    """
    providers = {
        "openai": False,
        "ollama": False,
        "anthropic": False,
        "azure": False,
        "google": False,
        "custom": True,  # Always available (uses OpenAI-compatible API)
    }

    try:
        import langchain_openai
        providers["openai"] = True
    except ImportError:
        pass

    try:
        import langchain_ollama
        providers["ollama"] = True
    except ImportError:
        pass

    try:
        import langchain_anthropic
        providers["anthropic"] = True
    except ImportError:
        pass

    try:
        import langchain_openai
        # Azure uses same package as OpenAI
        providers["azure"] = True
    except ImportError:
        pass

    try:
        import langchain_google_genai
        providers["google"] = True
    except ImportError:
        pass

    return providers


def _create_openai_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLanguageModel:
    """Create OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI provider requires 'langchain-openai' package. "
            "Install with: pip install langchain-openai"
        )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def _create_ollama_llm(
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    **kwargs
) -> BaseLanguageModel:
    """Create Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "Ollama provider requires 'langchain-ollama' package. "
            "Install with: pip install langchain-ollama"
        )

    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )


def _create_anthropic_llm(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLanguageModel:
    """Create Anthropic Claude LLM instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "Anthropic provider requires 'langchain-anthropic' package. "
            "Install with: pip install langchain-anthropic"
        )

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def _create_azure_llm(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: str = "2024-02-15-preview",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLanguageModel:
    """Create Azure OpenAI LLM instance."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError:
        raise ImportError(
            "Azure provider requires 'langchain-openai' package. "
            "Install with: pip install langchain-openai"
        )

    api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not azure_endpoint:
        raise ValueError(
            "Azure OpenAI requires API key and endpoint. Set AZURE_OPENAI_API_KEY "
            "and AZURE_OPENAI_ENDPOINT environment variables."
        )

    return AzureChatOpenAI(
        model=model,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def _create_google_llm(
    model: str = "gemini-pro",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseLanguageModel:
    """Create Google Generative AI LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "Google provider requires 'langchain-google-genai' package. "
            "Install with: pip install langchain-google-genai"
        )

    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key required. Set GOOGLE_API_KEY environment variable "
            "or pass api_key parameter."
        )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        **kwargs
    )


def _create_custom_llm(
    model: str,
    base_url: str,
    api_key: str = "not-needed",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLanguageModel:
    """
    Create custom endpoint LLM instance (OpenAI-compatible API).

    Works with:
    - Self-hosted LLMs (vLLM, text-generation-inference)
    - LiteLLM proxy
    - Custom OpenAI-compatible endpoints
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "Custom provider requires 'langchain-openai' package. "
            "Install with: pip install langchain-openai"
        )

    if not base_url:
        raise ValueError("Custom provider requires base_url parameter")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> BaseLanguageModel:
    """
    Create an LLM instance from the specified provider.

    Reads from environment variables if parameters not provided:
    - LLM_PROVIDER: Provider name (openai, ollama, anthropic, azure, google, custom)
    - LLM_MODEL: Model name
    - LLM_API_KEY: API key for the provider
    - LLM_BASE_URL: Base URL for Ollama or custom endpoints
    - LLM_TEMPERATURE: Temperature setting (default: 0.7)
    - LLM_MAX_TOKENS: Max tokens to generate

    Args:
        provider: LLM provider name. Options: openai, ollama, anthropic, azure, google, custom
        model: Model name (provider-specific)
        api_key: API key for the provider
        base_url: Base URL for Ollama or custom endpoints
        temperature: Temperature for generation (0.0-1.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseLanguageModel instance

    Examples:
        # Using environment variables (LLM_PROVIDER, LLM_MODEL, etc.)
        llm = create_llm()

        # OpenAI
        llm = create_llm(provider="openai", model="gpt-4o-mini", api_key="sk-...")

        # Ollama (local)
        llm = create_llm(provider="ollama", model="llama2")
        llm = create_llm(provider="ollama", model="mistral", base_url="http://192.168.1.100:11434")

        # Anthropic Claude
        llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")

        # Azure OpenAI
        llm = create_llm(
            provider="azure",
            model="gpt-4o-mini",
            api_key="...",
            azure_endpoint="https://your-resource.openai.azure.com"
        )

        # Google Gemini
        llm = create_llm(provider="google", model="gemini-pro", api_key="...")

        # Custom endpoint (OpenAI-compatible)
        llm = create_llm(
            provider="custom",
            model="custom-model",
            base_url="https://api.internal.com/v1",
            api_key="your-key"
        )

    Raises:
        ValueError: If provider is not supported or required parameters missing
        ImportError: If required package for provider is not installed
    """
    # Read from environment if not provided
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL")
    api_key = api_key or os.getenv("LLM_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL")
    temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens = max_tokens or (int(os.getenv("LLM_MAX_TOKENS")) if os.getenv("LLM_MAX_TOKENS") else None)

    provider = provider.lower()

    # Check if provider is available
    available_providers = get_available_providers()
    if provider not in available_providers:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {', '.join(available_providers.keys())}"
        )

    if not available_providers[provider]:
        raise ImportError(
            f"Provider '{provider}' is not available. "
            f"Install the required package for this provider."
        )

    # Create LLM based on provider
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return _create_openai_llm(model, api_key, temperature, max_tokens, **kwargs)

    elif provider == "ollama":
        model = model or "llama2"
        return _create_ollama_llm(model, base_url or "http://localhost:11434", temperature, **kwargs)

    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-20241022"
        return _create_anthropic_llm(model, api_key, temperature, max_tokens, **kwargs)

    elif provider == "azure":
        model = model or "gpt-4o-mini"
        azure_endpoint = kwargs.pop("azure_endpoint", None) or base_url
        return _create_azure_llm(
            model, api_key, azure_endpoint,
            temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    elif provider == "google":
        model = model or "gemini-pro"
        return _create_google_llm(model, api_key, temperature, **kwargs)

    elif provider == "custom":
        if not model:
            raise ValueError("Custom provider requires 'model' parameter")
        if not base_url:
            raise ValueError("Custom provider requires 'base_url' parameter")
        return _create_custom_llm(model, base_url, api_key or "not-needed", temperature, max_tokens, **kwargs)

    else:
        raise ValueError(f"Unsupported provider: {provider}")
