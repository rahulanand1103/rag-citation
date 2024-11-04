from setuptools import setup, find_packages

# "backports.tarfile==1.2.0",
setup(
    name="rag_citation",
    version="0.0.5",
    author="rahul anand",
    author_email="rahulanand1103@gmail.com",
    packages=find_packages(),
    install_requires=["spacy==3.7.5", "sentence_transformers==3.0.1"],
    description="RAG Citation is an project that combines Retrieval-Augmented Generation (RAG) with automatic citation generation. This tool is designed to enhance the credibility of AI-generated content by providing relevant citations for the information used in generating responses.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rahulanand1103/rag-citation",
    license="MIT",
    python_requires=">=3.8",
    include_package_data=True,
)
