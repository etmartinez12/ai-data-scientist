from setuptools import find_packages, setup
import os

def parse_requirements(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    with open(filename, "r") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8", errors="ignore") as fh:
    long_description = fh.read()

version = {}
with open("ai_data_science_team/_version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)


setup(
    name="ai-data-science-team",
    version=version["__version__"],
    description="AI-powered data science agents for automated data analysis and ML workflows.",
    author="Elias Tommy Martinez",
    author_email="martinez.elias12@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/etmartinez12/ai-data-scientist",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        # LLM Providers (optional)
        "openai": ["langchain-openai", "openai"],
        "ollama": ["langchain-ollama"],
        "anthropic": ["langchain-anthropic"],
        "all_providers": ["langchain-openai", "openai", "langchain-ollama", "langchain-anthropic"],
        # ML Tools (optional)
        "machine_learning": ["h2o", "mlflow"],
        # Data Science Tools (optional)
        "data_science": ["pytimetk", "missingno", "sweetviz"],
        # Everything
        "all": [
            "h2o", "mlflow",
            "pytimetk", "missingno", "sweetviz",
            "langchain-openai", "openai", "langchain-ollama", "langchain-anthropic"
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
