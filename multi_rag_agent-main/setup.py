from setuptools import setup, find_packages

setup(
    name="arag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv==1.0.1",
        "weaviate-client==3.25.2",
        "sentence-transformers==3.4.1",
        "PyPDF2==3.0.1",
        "tqdm==4.67.1",
        "numpy==2.2.4",
        "google-generativeai==0.8.4",
        "langchain==0.3.21",
        "langchain-core==0.3.47",
        "langchain-text-splitters==0.3.7",
        "pydantic==2.10.6",
        "pydantic-settings==2.8.1",
    ],
)