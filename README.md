# RAG Citation: Enhancing Rag Pipeline with Automatic Citations (A Non-LLM Approach)

## Project Overview

RAG Citation is an project that combines Retrieval-Augmented Generation (RAG) with automatic citation generation. This tool is designed to enhance the credibility of RAG-generated content by providing relevant citations for the information used in generating responses.


## Key Features

- **Non-LLM Approach:** Utilizes efficient algorithms and NLP techniques for citation generation, making it fast and lightweight.
- **Semantic Search:** Identifies relevant source documents based on meaning and context rather than just keyword matching.
- **Named Entity Recognition:**  Extracts and returns relevant named entities from LLM-generated answers, such as people, organizations, money and dates.
- **Flexible Integration:** Can be easily integrated into rag pipeline.
- **Hallucination (Beta)** This beta feature identifies instances where the LLM-generated answer contains entities like ["DATE", "MONEY", "CARDINAL", "ORDINAL", "QUANTITY", "TIME"], but these entities cannot be found within the context. If such a mismatch occurs, it flags the result as a potential hallucination.

## Quickstart
* <b>Langchain example</b>: [langchain.ipynb](https://github.com/rahulanand1103/rag-citation/blob/main/docs/examples/3.example-langchain.ipynb)
* <b>Embeddchain example</b>: [embeddchain.ipynb](https://github.com/rahulanand1103/rag-citation/blob/main/docs/examples/2.example-embeddchain.ipynb)
* <b>custom embedding model</b>: [custom_embedding_model.ipynb](https://github.com/rahulanand1103/rag-citation/blob/main/docs/examples/4.example-custom_embedding_model.ipynb)

To get started with `rag-citation`, install it using pip and download the spacy model:

```bash
pip install rag-citation
```
To download the spacy model-sm
```bash
python -m spacy download en_core_web_sm
```

To download the spacy model-md
```bash
python -m spacy download en_core_web_md
```

To download the spacy model-lg
```bash
python -m spacy download en_core_web_lg
```

Here's a basic example demonstrating how to use the library:

```python
from rag_citation import CiteItem, Inference
import uuid

## Sample context from vectorDB or semantic search
documents = [
    "Elon MuskCEO, Tesla$221.6B$439M (0.20%)Real Time Net Worthas of 8/6/24Reflects change since 5 pm ET of prior trading day. 1 in the world todayPhoto by Martin Schoeller for ForbesAbout Elon MuskElon Musk cofounded six companies, including electric car maker Tesla, rocket producer SpaceX and tunneling startup Boring Company.He owns about 12% of Tesla excluding options, but has pledged more than half his shares as collateral for personal loans of up to $3.5 billion.In early 2024, a Delaware judge voided Musk's 2018 deal to receive options equaling an additional 9% of Tesla.",
    "people in the world; as of August 2024[update], Forbes estimates his net worth to be US$241 billion.[3] Musk was born in Pretoria to model Maye and businessman and engineer Errol Musk, and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics and physics."
]

## Example answer generated by an LLM
answer = "Elon Musk's net worth is estimated to be US$241 billion as of August 2024."

## Helper function to generate a UUID
def generate_uuid():
    return str(uuid.uuid4())

## Helper function to create context in the correct format
def format_document(documents):
    context = []
    for document in documents:
        context.append(
            {
                "source_id": generate_uuid(), 
                "document": document, 
                "meta": [{"meta-data": "some-info"}], 
            }
        )
    return context

context = format_document(documents)
cite_item = CiteItem(answer=answer, context=context)

## Initialize the Inference 
inference = Inference(spacy_model="sm", embedding_model="md")

## Get citation and other information
output = inference(cite_item)

print("------ Citation ------")
print(output.citation) 
print("------ Hallucination ------") 
print(output.hallucination) 
print("------ Missing Entities ------")
print(output.missing) 

```

## Output Explanation

### `print(output.citation)`
```
[
  {
    "answer_sentences": "Elon Musk's net worth is estimated to be US$241 billion as of August 2024.",
    "cite_document": [
      {
        "document": "people in the world; as of August 2024[update], Forbes estimates his net worth to be US$241 billion.[3]",
        "source_id": "23d1f1f0-2afa-4749-8639-78ec685fd837",
        "entity": [
          {
            "word": "US$241 billion",
            "entity_name": "MONEY"
          },
          {
            "word": "August 2024",
            "entity_name": "DATE"
          }
        ],
        "meta": [
          {
            "url": "https://www.forbes.com/profile/elon-musk/",
            "chunk_id": "1eab8dd1ffa92906f7fc839862871ca5"
          }
        ]
      }
    ]
  }
]
```
| Key                | Description                                                                                           | Example                                                                                                    |
|--------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `answer_sentences` | Textual information or sentences extracted as answers or relevant information related to the citation.| `"Elon Musk's net worth is estimated to be US$241 billion as of August 2024."`                             |
| `cite_document`    | List of source documents used in the citation. Each document contains:                                |                                                                                                            |
|                    | - `document`: Text from the source document.                                                          | `"people in the world; as of August 2024[update], Forbes estimates his net worth to be US$241 billion.[3]"`|
|                    | - `source_id`: Unique identifier for the source document.                                             | `"6874d990-fedc-42bd-b0be-730bcdd59d26"`                                                                   |
|                    |  - `entity`: List of recognized entities in the document. Each entity contains:                       |                                                                                                            |
|                    |  - `word`: Recognized word or phrase.                                                                 | `"US$241 billion"`                                                                                         |
|                    | - `entity_name` Type of the entity (e.g., `MONEY`, `DATE`).                                           | `"MONEY"`                                                                                                  |
|                    | - `meta`Metadata about the document:                                                                  |  `[]`                                                                                                      |

##### `print(output.hallucination)` 
`False`

| Key              | Description                                                         | Example |
|------------------|---------------------------------------------------------------------|---------|
| `hallucination`  | Indicates if the output contains hallucinated information.          | `false` |

### `print(output.missing)`
` [] `

| Key           | Description                                 | Example             |
|---------------|---------------------------------------------|---------------------|
| `missing`     | List of entities expected but not found.    | `["$100 USD"]` |


## Installation

**From PyPI:**

```bash
pip install rag-citation
```

**From Source:**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag-citation.git 
   cd rag-citation
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt 
   ```

## Configuration

The `Inference` class can be configured with different models and settings:

- **`spacy_model`:** The spaCy model used for named entity recognition (default: `"en_core_web_sm"`). To use different models, pass:
  - `"sm"` for `en_core_web_sm`
  - `"md"` for `en_core_web_md`
  - `"lg"` for `en_core_web_lg`
  You can download and install spaCy models [here](https://spacy.io/models).

- **`embedding_model`:** The sentence embedding model from the SentenceTransformers library used for semantic similarity (default: `"all-mpnet-base-v2"`). To use different models, pass:
  - `"sm"` for `avsolatorio/GIST-small-Embedding-v0`
  - `"md"` for `avsolatorio/GIST-Embedding-v0`
  - `"lg"` for `avsolatorio/GIST-large-Embedding-v0`
  Install SentenceTransformers with: `pip install -U sentence-transformers`  
  You can explore the models on [Hugging Face](https://huggingface.co/avsolatorio/GIST-Embedding-v0).

- **`therhold_value`:** The similarity threshold value for semantic matching (current default: `0.88`). You can adjust this value as needed.


## Contributing

We welcome contributions! Here’s how you can help:

- **Report Bugs:** Submit issues on GitHub.
- **Suggest Features:**  Open an issue with your ideas.
- **Code Contributions:** Fork, make changes, and submit a pull request.
- **Documentation:** Update and enhance our docs.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [SpaCy](https://spacy.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [Huggingface/avsolatorio](https://huggingface.co/avsolatorio)
