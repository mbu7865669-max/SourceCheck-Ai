"""
Autonomous retrieval agent coordinator for SourceCheck AI.

Orchestrates the end-to-end question-answering workflow:
1. Validates the natural-language question.
2. Retrieves relevant web pages via the configured search provider.
3. Extracts and ranks evidence passages based on topical relevance.
4. Defends against prompt injection via structural context bounding.
5. Invokes language model reasoning to formulate a grounded answer.
6. Strictly validates citations against retrieved URLs to prevent hallucinated sources.
7. Formats and returns a typed, machine-readable AgentResponse.
"""

import logging
from typing import List, Optional, Set

import httpx

from app.config import settings
from app.extractor import extract_evidence
from app.model import BaseLLM, ModelError, ModelTimeoutError, get_model
from app.retrieval import SearchProvider, retrieve
from app.schemas import AgentResponse, Citation, Claim, EvidencePassage, RetrievalResult

logger = logging.getLogger(__name__)


class SourceCheckAgent:
    """
    Core agent orchestrating search, retrieval, evidence scoring, and grounded LLM reasoning.
    """

    def __init__(
        self,
        model: Optional[BaseLLM] = None,
        search_provider: Optional[SearchProvider] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model or get_model()
        self.search_provider = search_provider
        self.http_client = http_client

    def answer_question(
        self,
        question: str,
        max_search_results: Optional[int] = None,
        max_pages_to_fetch: Optional[int] = None,
        max_evidence_passages: Optional[int] = None,
    ) -> AgentResponse:
        """
        Execute the full question-answering pipeline.

        Args:
            question: Plain-English user question.
            max_search_results: Bounded search result count.
            max_pages_to_fetch: Maximum pages to download and clean.
            max_evidence_passages: Maximum evidence passages passed to the model.

        Returns:
            Structured AgentResponse containing the grounded answer and verified citations.
        """
        errors: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Question Validation
        # ----------------------------------------------------------------------
        if not question or not isinstance(question, str) or not question.strip():
            return AgentResponse(
                question=question or "",
                answer="Please provide a valid, non-empty question.",
                sources=[],
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=["Question is empty or whitespace only."],
            )

        clean_question = question.strip()

        # ----------------------------------------------------------------------
        # 2. Web Retrieval
        # ----------------------------------------------------------------------
        max_search = (
            max_search_results if max_search_results is not None else settings.MAX_SEARCH_RESULTS
        )
        max_pages = (
            max_pages_to_fetch if max_pages_to_fetch is not None else settings.MAX_PAGES_TO_FETCH
        )

        try:
            retrieval_res: RetrievalResult = retrieve(
                query=clean_question,
                max_search_results=max_search,
                max_pages_to_fetch=max_pages,
                provider=self.search_provider,
                client=self.http_client,
            )
        except Exception as exc:
            logger.error(f"Retrieval stage failed: {exc}")
            return AgentResponse(
                question=clean_question,
                answer="An error occurred while searching for web information.",
                sources=[],
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=[f"Retrieval error: {exc}"],
            )

        # Collect safe diagnostic warnings for unavailable pages
        for page in retrieval_res.pages:
            if not page.success and page.error:
                errors.append(f"Failed to fetch {page.original_url}: {page.error}")

        # Check if search returned zero results or all page fetches failed
        successful_pages = [p for p in retrieval_res.pages if p.success and p.text]
        if not retrieval_res.search_results or not successful_pages:
            return AgentResponse(
                question=clean_question,
                answer="No accessible web sources could be found to answer this question.",
                sources=[],
                claims=[],
                insufficient_evidence=True,
                status="insufficient_evidence",
                errors=errors,
            )

        # Collect verified URLs from retrieval
        verified_urls: Set[str] = set()
        for p in successful_pages:
            if p.final_url:
                verified_urls.add(p.final_url)
            if p.original_url:
                verified_urls.add(p.original_url)

        # ----------------------------------------------------------------------
        # 3. Evidence Extraction & Ranking
        # ----------------------------------------------------------------------
        max_evidence = (
            max_evidence_passages
            if max_evidence_passages is not None
            else settings.MAX_EVIDENCE_FOR_MODEL
        )

        try:
            passages: List[EvidencePassage] = extract_evidence(
                question=clean_question,
                retrieval_input=retrieval_res,
                max_total_passages=max_evidence,
                min_score=0.1,  # Only pass passages with positive query keyword overlap
            )
        except Exception as exc:
            logger.error(f"Evidence extraction failed: {exc}")
            return AgentResponse(
                question=clean_question,
                answer="An error occurred while analyzing candidate evidence passages.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=errors + [f"Extraction error: {exc}"],
            )

        # If zero passages had topical relevance, abstain
        if not passages:
            return AgentResponse(
                question=clean_question,
                answer="Insufficient factual evidence found in retrieved web sources to answer the question.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="insufficient_evidence",
                errors=errors,
            )

        # ----------------------------------------------------------------------
        # 4. LLM Reasoning over Bounded Evidence
        # ----------------------------------------------------------------------
        try:
            model_output = self.model.generate_answer(
                question=clean_question,
                passages=passages,
            )
        except ModelTimeoutError as t_err:
            logger.warning(f"Model reasoning timed out: {t_err}")
            return AgentResponse(
                question=clean_question,
                answer="The reasoning model timed out before completing the answer.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=errors + [str(t_err)],
            )
        except ModelError as m_err:
            logger.error(f"Model generation error: {m_err}")
            return AgentResponse(
                question=clean_question,
                answer="The reasoning model encountered an error formulating the answer.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=errors + [str(m_err)],
            )
        except Exception as exc:
            logger.error(f"Unexpected model error: {exc}")
            return AgentResponse(
                question=clean_question,
                answer="An unexpected error occurred during evidence reasoning.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="error",
                errors=errors + [str(exc)],
            )

        # ----------------------------------------------------------------------
        # 5. Grounding & Citation Validation
        # ----------------------------------------------------------------------
        raw_answer = model_output.get("answer", "").strip()
        insufficient = model_output.get("insufficient_evidence", False)
        raw_claims = model_output.get("claims", [])

        # Check for contradictory or insufficient evidence signals
        if (
            insufficient
            or not raw_answer
            or "insufficient" in raw_answer.lower()
            or "conflicting" in raw_answer.lower()
            or "contradictory" in raw_answer.lower()
        ):
            return AgentResponse(
                question=clean_question,
                answer=raw_answer or "Insufficient factual evidence found to answer the question.",
                sources=list(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="insufficient_evidence",
                errors=errors,
            )

        # Validate that cited URLs exist in verified_urls (prevent citation hallucination)
        validated_claims: List[Claim] = []
        validated_sources_set: Set[str] = set()

        # Map URLs to titles from successful pages
        url_to_title = {p.final_url: p.title for p in successful_pages}
        url_to_title.update({p.original_url: p.title for p in successful_pages})

        for raw_c in raw_claims:
            claim_text = raw_c.get("claim", "").strip()
            cited_url = raw_c.get("url", "").strip()
            quote_text = raw_c.get("quote", "").strip()

            # Discard citations to invented or unverified URLs
            if cited_url in verified_urls:
                validated_sources_set.add(cited_url)
                citation = Citation(
                    url=cited_url,
                    title=url_to_title.get(cited_url, ""),
                    quote=quote_text,
                )
                validated_claims.append(
                    Claim(claim=claim_text or raw_answer, citations=[citation])
                )

        # If model provided an answer but cited no valid URL, fallback to top passage source
        if not validated_sources_set and passages:
            top_source = passages[0].source_url
            if top_source in verified_urls:
                validated_sources_set.add(top_source)
                validated_claims.append(
                    Claim(
                        claim=raw_answer,
                        citations=[
                            Citation(
                                url=top_source,
                                title=passages[0].page_title,
                                quote=passages[0].text[:200],
                            )
                        ],
                    )
                )

        # Deduplicate and sort validated sources deterministically
        final_sources = sorted(list(validated_sources_set))

        return AgentResponse(
            question=clean_question,
            answer=raw_answer,
            sources=final_sources,
            claims=validated_claims,
            insufficient_evidence=False,
            status="success",
            errors=errors,
        )


# Global agent runner helper
def run_agent(question: str) -> AgentResponse:
    """Convenience function to run the SourceCheck agent with configured defaults."""
    agent = SourceCheckAgent()
    return agent.answer_question(question)
