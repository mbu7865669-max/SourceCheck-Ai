"""
Pydantic data schemas and models for SourceCheck AI.

Defines structured models for search results, fetched web pages,
retrieval output, and final agent response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single web search result item."""

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="Target URL of the search result")
    snippet: str = Field(default="", description="Snippet or preview description from search engine")


class FetchedPage(BaseModel):
    """Metadata and extracted text content from a fetched web page."""

    original_url: str = Field(..., description="The requested URL before any redirects")
    final_url: str = Field(..., description="The final destination URL after redirects")
    title: str = Field(default="", description="HTML title or main heading of the page")
    text: str = Field(
        default="",
        description="Clean, readable plain text extracted from HTML. NOTE: Untrusted external content.",
    )
    status_code: Optional[int] = Field(default=None, description="HTTP response status code")
    success: bool = Field(default=True, description="Whether the page was retrieved and parsed successfully")
    error: Optional[str] = Field(default=None, description="Error message if retrieval or parsing failed")


class RetrievalResult(BaseModel):
    """Aggregate output of the web retrieval stage for a user query."""

    query: str = Field(..., description="The original search query")
    search_results: List[SearchResult] = Field(
        default_factory=list, description="Search engine result items"
    )
    pages: List[FetchedPage] = Field(
        default_factory=list, description="Successfully or attempted fetched web pages"
    )


class EvidencePassage(BaseModel):
    """A scored passage extracted from a retrieved web page for evidence verification."""

    text: str = Field(..., description="Verbatim text excerpt of the evidence passage")
    source_url: str = Field(..., description="Canonical source URL from which the passage was extracted")
    page_title: str = Field(default="", description="Title of the source web page")
    score: float = Field(default=0.0, description="Deterministic relevance score against the question")
    passage_index: int = Field(default=0, description="0-indexed position of the passage within the source page")


class ExtractionResult(BaseModel):
    """Aggregate output of the evidence extraction stage."""

    question: str = Field(..., description="The user query or question evaluated")
    passages: List[EvidencePassage] = Field(
        default_factory=list, description="Ranked, deduplicated evidence passages"
    )


class Citation(BaseModel):
    """Citation linking a specific claim or statement to a verified source URL."""

    url: str = Field(..., description="Verified source URL from retrieval")
    title: str = Field(default="", description="Title of the cited source page")
    quote: str = Field(default="", description="Relevant supporting excerpt from the source text")


class Claim(BaseModel):
    """An individual factual claim supported by citations."""

    claim: str = Field(..., description="Factual claim or statement formulated from evidence")
    citations: List[Citation] = Field(
        default_factory=list, description="Supporting citations from retrieved sources"
    )


class AgentResponse(BaseModel):
    """Structured response from the SourceCheck AI agent."""

    question: str = Field(..., description="The original user question")
    answer: str = Field(..., description="Concise factual answer strictly grounded in retrieved evidence")
    sources: List[str] = Field(
        default_factory=list, description="List of unique, verified source URLs supporting the answer"
    )
    claims: List[Claim] = Field(
        default_factory=list, description="Claim-level citations linking facts to sources"
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True if evidence was missing, inadequate, unverified, or contradictory",
    )
    status: str = Field(
        default="success",
        description="Status code: 'success', 'insufficient_evidence', or 'error'",
    )
    errors: List[str] = Field(
        default_factory=list, description="Safe diagnostic messages if any pipeline stage encountered an issue"
    )


