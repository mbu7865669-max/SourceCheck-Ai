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
import re
from typing import List, Optional, Set

import httpx

from app.config import settings
from app.extractor import STOPWORDS, extract_evidence, normalize_text_for_matching
from app.model import BaseLLM, ModelError, ModelTimeoutError, get_model
from app.retrieval import SearchProvider, is_blocked_status, retrieve
from app.schemas import AgentResponse, Citation, Claim, EvidencePassage, RetrievalResult

logger = logging.getLogger(__name__)



# ==============================================================================
# Evidence verification
# ==============================================================================

_MIN_QUOTE_WORDS = 3
_MIN_CLAIM_OVERLAP = 0.5


def _quote_found_on_page(quote: str, page_norm: str) -> bool:
    """True when every fragment of the quote (split on ellipses) appears verbatim
    on the page, ignoring case, punctuation and whitespace."""
    fragments = [f for f in re.split(r"\.\.\.|\u2026", quote) if f.strip()]
    if not fragments:
        return False
    total_words = 0
    for fragment in fragments:
        norm = normalize_text_for_matching(fragment)
        if not norm:
            continue
        if norm not in page_norm:
            return False
        total_words += len(norm.split())
    return total_words >= _MIN_QUOTE_WORDS


def _token_matches(token: str, pool: Set[str]) -> bool:
    if token in pool:
        return True
    stem = token[:5]
    return len(token) > 5 and any(t.startswith(stem) for t in pool)


def _support_ratio(claim: str, quote: str) -> float:
    """Share of the claim's content words found in the quote (prefix match, so
    "walked"/"walk" agree); -1 when a number in the claim is missing from it."""
    claim_tokens = normalize_text_for_matching(claim).split()
    quote_tokens = set(normalize_text_for_matching(quote).split())
    numbers = [t for t in claim_tokens if t.isdigit()]
    if any(n not in quote_tokens for n in numbers):
        return -1.0
    content = [t for t in claim_tokens if not t.isdigit() and len(t) > 2 and t not in STOPWORDS]
    if not content:
        return 1.0 if numbers else 0.0
    return sum(1 for t in content if _token_matches(t, quote_tokens)) / len(content)


def _quote_supports_claim(claim: str, quote: str) -> bool:
    """Every number in the claim must appear in the quote, and at least half of
    its content words too. Blocks answers grounded in nothing on the page."""
    return _support_ratio(claim, quote) >= _MIN_CLAIM_OVERLAP


_SOURCE_REF = re.compile(r"sources?\s*((?:\d+\s*(?:,|and|&)?\s*)+)", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _cited_source_numbers(text: str) -> Set[int]:
    numbers: Set[int] = set()
    for match in _SOURCE_REF.finditer(text):
        numbers.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    return numbers


def _best_supporting_sentence(claim: str, passages: List[EvidencePassage]):
    """The real source sentence that best supports the claim, as (passage, sentence),
    or None when no sentence clears the support bar."""
    best = None
    best_ratio = _MIN_CLAIM_OVERLAP
    for passage in passages:
        for sentence in _SENTENCE_SPLIT.split(passage.text):
            sentence = sentence.strip()
            if len(normalize_text_for_matching(sentence).split()) < _MIN_QUOTE_WORDS:
                continue
            ratio = _support_ratio(claim, sentence)
            if ratio >= best_ratio and (best is None or ratio > best_ratio):
                best, best_ratio = (passage, sentence), ratio
    return best


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
        # Sources that refused us (403/429/...) land here instead of in `errors`:
        # an expected skip is not a failure, and a red diagnostic for one blocked
        # dictionary site makes a perfectly good answer look broken.
        notices: List[str] = []

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

        # Collect safe diagnostics for unavailable pages: refusals are notices,
        # everything else (timeouts, bad content, oversize bodies) stays an error.
        for page in retrieval_res.pages:
            if page.success or not page.error:
                continue
            if is_blocked_status(page.status_code):
                notices.append(
                    f"Skipped {page.original_url} - refused by the site "
                    f"(HTTP {page.status_code})."
                )
            else:
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
                notices=notices,
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
                notices=notices,
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
                notices=notices,
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
                notices=notices,
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
                notices=notices,
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
                notices=notices,
            )

        # ----------------------------------------------------------------------
        # 5. Grounding & Citation Validation
        # ----------------------------------------------------------------------
        # Free models don't always follow the JSON schema (e.g. claims as plain
        # strings, answer as a list). Coerce the shape here so a malformed reply
        # degrades to "no usable claims" instead of a 500.
        if not isinstance(model_output, dict):
            logger.warning("Model returned %s instead of a JSON object", type(model_output).__name__)
            model_output = {}
        answer_value = model_output.get("answer")
        raw_answer = answer_value.strip() if isinstance(answer_value, str) else (
            "" if answer_value is None else str(answer_value).strip()
        )
        insufficient = model_output.get("insufficient_evidence", False) is True
        claims_value = model_output.get("claims")
        if not isinstance(claims_value, list):
            claims_value = []
        # Plain-string claims are kept as {"claim": text}; verification below
        # recovers a real supporting sentence for them or drops them.
        raw_claims = []
        for c in claims_value:
            if isinstance(c, dict):
                raw_claims.append(c)
            elif isinstance(c, str) and c.strip():
                raw_claims.append({"claim": c.strip()})
        if len(raw_claims) != len(claims_value):
            logger.warning(
                "Dropped %d malformed claim(s) from the model output",
                len(claims_value) - len(raw_claims),
            )

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
                notices=notices,
            )

        # Validate every claim against the pages we actually fetched:
        #   1. the cited URL must be one we retrieved (no invented citations),
        #   2. the quote must appear verbatim on that page (no invented evidence),
        #   3. the quote must support the claim (numbers + content words).
        validated_claims: List[Claim] = []
        validated_sources_set: Set[str] = set()

        url_to_title = {p.final_url: p.title for p in successful_pages}
        url_to_title.update({p.original_url: p.title for p in successful_pages})
        url_to_text: dict = {}
        for page in successful_pages:
            norm = normalize_text_for_matching(page.text)
            for page_url in (page.final_url, page.original_url):
                if page_url:
                    url_to_text[page_url] = norm

        rejected = 0
        for raw_c in raw_claims:
            claim_text = str(raw_c.get("claim") or "").strip() or raw_answer
            cited_url = str(raw_c.get("url") or "").strip()
            quote_text = str(raw_c.get("quote") or "").strip()
            # Source numbers like "(Source 1, 2)" are references, not facts to check.
            source_refs = _cited_source_numbers(claim_text)
            claim_text = re.sub(r"\s*\(\s*\)", "", _SOURCE_REF.sub("", claim_text)).strip()

            verified = (
                cited_url in verified_urls
                and _quote_found_on_page(quote_text, url_to_text.get(cited_url, ""))
                and _quote_supports_claim(claim_text, quote_text)
            )

            if not verified:
                # Recover: find a real sentence in the evidence that supports the
                # claim - in the cited URL, else the "Source N" passages it names,
                # else any passage. The quote shown is always page text.
                pool = [ps for ps in passages if ps.source_url == cited_url]
                if not pool:
                    pool = [ps for i, ps in enumerate(passages, start=1) if i in source_refs]
                if not pool:
                    pool = list(passages)
                found = _best_supporting_sentence(claim_text, pool)
                if found is not None and found[0].source_url in verified_urls:
                    cited_url, quote_text = found[0].source_url, found[1]
                    verified = True

            if not verified:
                rejected += 1
                continue

            validated_sources_set.add(cited_url)
            validated_claims.append(
                Claim(
                    claim=claim_text,
                    citations=[
                        Citation(
                            url=cited_url,
                            title=url_to_title.get(cited_url, ""),
                            quote=quote_text,
                        )
                    ],
                )
            )

        if rejected:
            logger.info("Rejected %d claim(s) not supported by the fetched pages", rejected)

        # Nothing survived verification: abstain instead of presenting the model's
        # answer as verified (it may come from the model's own knowledge or setup).
        if not validated_claims:
            return AgentResponse(
                question=clean_question,
                answer="Insufficient factual evidence found in retrieved web sources to answer the question.",
                sources=sorted(verified_urls),
                claims=[],
                insufficient_evidence=True,
                status="insufficient_evidence",
                errors=errors,
                notices=notices + [
                    "The model's answer could not be matched to text on the retrieved pages, "
                    "so it is not shown as verified."
                ],
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
            notices=notices,
        )


# Global agent runner helper
def run_agent(question: str) -> AgentResponse:
    """Convenience function to run the SourceCheck agent with configured defaults."""
    agent = SourceCheckAgent()
    return agent.answer_question(question)
