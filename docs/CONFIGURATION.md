# LLM Configuration Guide

This guide explains how to configure different LLM providers for AI Data Science Team.

## Table of Contents

- [Quick Start](#quick-start)
- [Provider-Specific Configuration](#provider-specific-configuration)
  - [OpenAI](#openai)
  - [Ollama (Local Models)](#ollama-local-models)
  - [Anthropic Claude](#anthropic-claude)
  - [Azure OpenAI](#azure-openai)
  - [Google Generative AI](#google-generative-ai)
  - [Custom Endpoints](#custom-endpoints)
- [Configuration Methods](#configuration-methods)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Provider Dependencies

```bash
# OpenAI (default)
pip install "ai-data-science-team[openai]"

# Ollama (local models)
pip install "ai-data-science-team[ollama]"

# Anthropic
pip install "ai-data-science-team[anthropic]"

# All providers
pip install "ai-data-science-team[all_providers]"
```

### 2. Set Environment Variables

Create a `.env` file in your project root:

```bash
# Choose your provider
LLM_PROVIDER=openai  # or ollama, anthropic, azure, google, custom

# Set the model
LLM_MODEL=gpt-4o-mini

# Set your API key (provider-specific)
OPENAI_API_KEY=sk-your-key-here
```

### 3. Use in Code

```python
from ai_data_science_team import create_llm, DataCleaningAgent

# Auto-loads from environment variables
llm = create_llm()

# Use with any agent
agent = DataCleaningAgent(model=llm)
agent.invoke_agent(user_instructions="Clean the data", data_raw=df)
```

## Provider-Specific Configuration

### OpenAI

**Requirements:**
- OpenAI API key from https://platform.openai.com/api-keys
- Package: `langchain-openai`

**Environment Variables:**
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini  # or gpt-4o, gpt-4-turbo, gpt-3.5-turbo
OPENAI_API_KEY=sk-your-key-here
LLM_TEMPERATURE=0.7  # Optional
```

**In Code:**
```python
from ai_data_science_team import create_llm

# Method 1: Environment variables
llm = create_llm()

# Method 2: Explicit parameters
llm = create_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-your-key-here",
    temperature=0.7
)

# Method 3: Direct (backward compatible)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-your-key-here")
```

**Available Models:**
- `gpt-4o` - Latest GPT-4 Omni
- `gpt-4o-mini` - Faster, cheaper GPT-4 Omni
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5

### Ollama (Local Models)

**Requirements:**
- Ollama installed locally: https://ollama.ai/
- Package: `langchain-ollama`
- Models pulled via: `ollama pull llama2`

**Environment Variables:**
```bash
LLM_PROVIDER=ollama
LLM_MODEL=llama2  # or mistral, mixtral, phi, codellama, etc.
LLM_BASE_URL=http://localhost:11434  # Default Ollama URL
LLM_TEMPERATURE=0.7  # Optional
```

**In Code:**
```python
from ai_data_science_team import create_llm

# Local Ollama (default port)
llm = create_llm(provider="ollama", model="llama2")

# Remote Ollama instance
llm = create_llm(
    provider="ollama",
    model="mistral",
    base_url="http://192.168.1.100:11434"
)
```

**Popular Models:**
- `llama2` - Meta's Llama 2 (7B, 13B, 70B)
- `llama3` - Meta's Llama 3
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B
- `phi` - Microsoft Phi-2
- `codellama` - Code-specialized Llama
- `deepseek-coder` - DeepSeek Coder

**Setup Ollama:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2

# Start Ollama service (usually auto-starts)
ollama serve
```

### Anthropic Claude

**Requirements:**
- Anthropic API key from https://console.anthropic.com/
- Package: `langchain-anthropic`

**Environment Variables:**
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022  # or other Claude models
ANTHROPIC_API_KEY=sk-ant-your-key-here
LLM_TEMPERATURE=0.7  # Optional
```

**In Code:**
```python
from ai_data_science_team import create_llm

llm = create_llm(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-your-key-here"
)
```

**Available Models:** 
# update these are more models become available
- `claude-3-5-sonnet-20241022` - Latest Claude 3.5 Sonnet
- `claude-3-opus-20240229` - Most capable Claude 3
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fastest Claude 3

### Azure OpenAI

**Requirements:**
- Azure OpenAI resource and deployment
- Package: `langchain-openai`

**Environment Variables:**
```bash
LLM_PROVIDER=azure
LLM_MODEL=gpt-4o-mini  # Your deployment name
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
```

**In Code:**
```python
from ai_data_science_team import create_llm

llm = create_llm(
    provider="azure",
    model="gpt-4o-mini",
    api_key="your-azure-key",
    azure_endpoint="https://your-resource.openai.azure.com"
)
```

### Google Generative AI

**Requirements:**
- Google API key from https://makersuite.google.com/app/apikey
- Package: `langchain-google-genai`

**Environment Variables:**
```bash
LLM_PROVIDER=google
LLM_MODEL=gemini-pro
GOOGLE_API_KEY=your-google-api-key
```

**In Code:**
```python
from ai_data_science_team import create_llm

llm = create_llm(
    provider="google",
    model="gemini-pro",
    api_key="your-google-api-key"
)
```

**Available Models:**
- `gemini-pro` - Text generation
- `gemini-pro-vision` - Multimodal (text + images)
- `gemini-1.5-pro` - Latest Gemini 1.5

### Custom Endpoints

Use this for self-hosted LLMs or custom OpenAI-compatible APIs (vLLM, LiteLLM, text-generation-inference, etc.).

**Environment Variables:**
```bash
LLM_PROVIDER=custom
LLM_MODEL=your-model-name
LLM_BASE_URL=https://your-api-endpoint.com/v1
LLM_API_KEY=your-api-key  # Optional, use "not-needed" if no auth
```

**In Code:**
```python
from ai_data_science_team import create_llm

llm = create_llm(
    provider="custom",
    model="your-model-name",
    base_url="https://your-api-endpoint.com/v1",
    api_key="your-api-key"  # or "not-needed"
)
```

**Compatible with:**
- vLLM deployments
- Text-generation-inference (TGI)
- LiteLLM proxy
- FastChat
- Any OpenAI-compatible API

## Configuration Methods

### Method 1: Environment Variables (.env file)

**Most Recommended** - Clean separation of config from code.

Create `.env` in your project root:
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key-here
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

Then in code:
```python
from ai_data_science_team import create_llm
llm = create_llm()  # Auto-loads from .env
```

### Method 2: YAML Configuration File

Create `config.yaml`:
```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}  # Can reference env vars
  temperature: 0.7
  max_tokens: 4096
```

Use in code:
```python
from ai_data_science_team import get_config, create_llm

config = get_config(config_file="config.yaml")
llm = create_llm(
    provider=config.provider,
    model=config.model,
    api_key=config.api_key
)
```

### Method 3: Direct Code Configuration

```python
from ai_data_science_team import create_llm

llm = create_llm(
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-your-key-here",
    temperature=0.7,
    max_tokens=4096
)
```

### Method 4: Pass LLM Instance Directly (Backward Compatible)

```python
from langchain_openai import ChatOpenAI
from ai_data_science_team import DataCleaningAgent

# Create LLM directly
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-...")

# Pass to agent
agent = DataCleaningAgent(model=llm)
```

## Troubleshooting

### "ImportError: No module named 'langchain_openai'"

Install the provider package:
```bash
pip install langchain-openai
# or
pip install "ai-data-science-team[openai]"
```

### "OpenAI API key required"

Set your API key:
```bash
export OPENAI_API_KEY=sk-your-key-here
# or add to .env file
```

### "Connection refused" with Ollama

Make sure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

### "Rate limit exceeded" with OpenAI

Try:
1. Reduce request rate
2. Use a different model (e.g., gpt-3.5-turbo)
3. Upgrade your OpenAI plan

### Models not working as expected

Try adjusting temperature:
```python
# More deterministic (lower temperature)
llm = create_llm(provider="openai", model="gpt-4o-mini", temperature=0.1)

# More creative (higher temperature)
llm = create_llm(provider="openai", model="gpt-4o-mini", temperature=0.9)
```

### Azure OpenAI specific issues

Make sure:
1. Deployment name matches your Azure deployment (not the base model name)
2. API version is correct
3. Endpoint URL is complete (includes https://)

## Best Practices

1. **Never commit API keys** - Use `.env` files and add `.env` to `.gitignore`
2. **Use environment variables** - Keep config separate from code
3. **Start with cheaper models** - Use `gpt-3.5-turbo` or `llama2` for testing
4. **Monitor costs** - Set up billing alerts for cloud providers
5. **Test locally first** - Use Ollama for development before deploying with paid APIs
6. **Version your configs** - Keep `example.env` in version control with dummy values
7. **Use model-specific optimizations** - Adjust temperature, max_tokens based on provider

## Example: Multi-Provider Setup

Support multiple providers in the same application:

```python
from ai_data_science_team import create_llm
import os

# Development: Use Ollama (free, local)
if os.getenv("ENVIRONMENT") == "development":
    llm = create_llm(provider="ollama", model="llama2")

# Production: Use OpenAI
elif os.getenv("ENVIRONMENT") == "production":
    llm = create_llm(provider="openai", model="gpt-4o-mini")

# Fallback
else:
    llm = create_llm()  # Uses .env configuration
```
