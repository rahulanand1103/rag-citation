from pydantic import BaseModel, Field
from typing import List


class CitationItem(BaseModel):
    """A single citation mapping a sentence from the answer to its source document."""

    sentence: str = Field(
        description="An exact sentence from the generated answer that is supported by a source document."
    )
    source_document: str = Field(
        description="The source_id of the document that supports this sentence."
    )


class CitationResponse(BaseModel):
    """The complete list of citations for an answer."""

    citations: List[CitationItem] = Field(
        description="List of citations, each mapping an answer sentence to its source document ID."
    )
