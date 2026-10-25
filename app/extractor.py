"""
Evidence extraction and relevance-ranking module for SourceCheck AI.

Provides deterministic, lightweight passage splitting and relevance scoring
to extract supporting context from retrieved web pages for downstream AI reasoning.

SCORING METHODOLOGY & LIMITATIONS:
1. Scoring combines:
   - Content keyword overlap with term frequency scaling.
   - Consecutive multi-word phrase matching (bigrams and trigrams).
   - Exact query phrase matches.
   - Length penalties for overly short snippets.
2. LIMITATION & DISTINCTION:
   High keyword overlap measures topical relevance, NOT factual truth or credibility.
   A passage discussing a topic can be outdated, misleading, or contradictory.
   Factual validation against source claims is deferred to the subsequent AI reasoning layer.
"""

import logging
import re
from typing import List, Optional, Set, Tuple, Union

from app.config import settings
from app.schemas import EvidencePassage, FetchedPage, RetrievalResult

logger = logging.getLogger(__name__)

# Standard English stopwords to filter non-informative query terms
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


# ==============================================================================
# Text Normalization & Tokenization
# ==============================================================================


def normalize_text_for_matching(text: str) -> str:
    """
    Normalize text for scoring by lowercasing, stripping punctuation, and collapsing whitespace.
    The original text is retained separately for citations.
    """
    if not text:
        return ""
    # Lowercase and replace non-alphanumeric characters with spaces (supports unicode \w)
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_keywords_and_phrases(question: str) -> Tuple[List[str], List[str], str]:
    """
    Extract meaningful content keywords, consecutive n-grams (phrases), and full query phrase.

    Returns:
        (keywords, phrases, exact_query_normalized)
    """
    norm_q = normalize_text_for_matching(question)
    tokens = [t for t in norm_q.split() if t]

    # Filter out stopwords and single characters
    content_words = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # If all tokens were stopwords (e.g. "who is who"), fall back to all tokens
    keywords = content_words if content_words else tokens

    # Generate bigrams and trigrams for phrase matching
    phrases: List[str] = []
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            phrases.append(f"{tokens[i]} {tokens[i+1]}")
    if len(tokens) >= 3:
        for i in range(len(tokens) - 2):
            phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    return keywords, phrases, norm_q


# ==============================================================================
# Passage Splitting
# ==============================================================================


def split_text_into_passages(
    text: str,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> List[str]:
    """
    Split clean page text into coherent paragraphs or sentence groups.

    Args:
        text: Raw cleaned page text.
        min_chars: Minimum character length for a standalone passage.
        max_chars: Maximum character length before a paragraph is broken down.

    Returns:
        List of candidate text passages preserving verbatim source content.
    """
    min_chars = min_chars if min_chars is not None else settings.MIN_PASSAGE_CHARS
    max_chars = max_chars if max_chars is not None else settings.MAX_PASSAGE_CHARS

    if not text or not text.strip():
        return []

    # First split by double newlines into natural paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages: List[str] = []
    seen_normalized: Set[str] = set()

    for para in raw_paragraphs:
        # If the paragraph is already within bounds, keep it as a coherent passage
        if min_chars <= len(para) <= max_chars:
            norm_key = normalize_text_for_matching(para)
            if norm_key and norm_key not in seen_normalized:
                seen_normalized.add(norm_key)
                passages.append(para)
            continue

        # If paragraph is longer than max_chars, split into sentence clusters
        if len(para) > max_chars:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            current_chunk: List[str] = []
            current_len = 0

            for s in sentences:
                s_len = len(s)
                if current_chunk and (current_len + s_len + 1 > max_chars):
                    chunk_text = " ".join(current_chunk).strip()
                    if len(chunk_text) >= min_chars:
                        norm_key = normalize_text_for_matching(chunk_text)
                        if norm_key and norm_key not in seen_normalized:
                            seen_normalized.add(norm_key)
                            passages.append(chunk_text)
                    # Sliding overlap: keep last sentence for context continuity
                    current_chunk = [current_chunk[-1], s]
                    current_len = len(current_chunk[0]) + s_len + 1
                else:
                    current_chunk.append(s)
                    current_len += s_len + 1

            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) >= min_chars:
                    norm_key = normalize_text_for_matching(chunk_text)
                    if norm_key and norm_key not in seen_normalized:
                        seen_normalized.add(norm_key)
                        passages.append(chunk_text)

        # Standalone paragraphs < min_chars are discarded as boilerplate noise

    return passages


# ==============================================================================
# Relevance Scoring
# ==============================================================================


def score_passage(
    passage_text: str,
    keywords: List[str],
    phrases: List[str],
    exact_query_norm: str,
) -> float:
    """
    Compute a deterministic relevance score for a passage against query keywords and phrases.

    Scoring components:
    - Unique keyword matches with sublinear term-frequency bonus.
    - Query coverage ratio (proportion of question keywords present).
    - Multi-word phrase matches (bigrams and trigrams).
    - Exact query string match bonus.
    - Short passage penalty.

    Returns:
        Deterministic float score >= 0.0.
    """
    if not passage_text or not keywords:
        return 0.0

    norm_passage = normalize_text_for_matching(passage_text)
    passage_tokens = norm_passage.split()
    if not passage_tokens:
        return 0.0

    token_counts: dict[str, int] = {}
    for t in passage_tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    # 1. Keyword overlap
    matched_keywords = 0
    keyword_score = 0.0

    for kw in keywords:
        count = token_counts.get(kw, 0)
        if count > 0:
            matched_keywords += 1
            # Base point + sublinear term-frequency bonus (capped at +0.6)
            tf_bonus = min(count - 1, 3) * 0.2
            keyword_score += 1.0 + tf_bonus

    if matched_keywords == 0:
        return 0.0

    # Query coverage multiplier (encourages passages that answer multiple parts of the question)
    coverage_ratio = matched_keywords / len(keywords)
    total_score = keyword_score + (coverage_ratio * 2.0)

    # 2. Multi-word phrase matches
    for phrase in phrases:
        if phrase in norm_passage:
            # Differentiate bigrams vs trigrams by word count
            words = phrase.split()
            if len(words) == 2:
                total_score += 1.5
            elif len(words) >= 3:
                total_score += 2.5

    # 3. Exact full query phrase match bonus
    if exact_query_norm and len(exact_query_norm.split()) >= 2:
        if exact_query_norm in norm_passage:
            total_score += 4.0

    # 4. Length penalty for extremely brief passages (< 60 chars)
    if len(passage_text) < 60:
        total_score -= 0.5

    return max(round(total_score, 4), 0.0)


# ==============================================================================
# High-Level Evidence Extraction & Ranking
# ==============================================================================


def extract_evidence_from_page(
    page: FetchedPage,
    question: str,
    max_passages: Optional[int] = None,
    min_score: float = 0.0,
) -> List[EvidencePassage]:
    """
    Extract and rank evidence passages from a single fetched web page.

    Args:
        page: A FetchedPage object containing cleaned text and source metadata.
        question: User query to evaluate relevance against.
        max_passages: Maximum passages to return from this page.
        min_score: Minimum threshold score to be included.

    Returns:
        List of EvidencePassage objects ordered deterministically by score.
    """
    max_passages = (
        max_passages if max_passages is not None else settings.MAX_PASSAGES_PER_PAGE
    )

    if not page.success or not page.text or not question or not question.strip():
        return []

    keywords, phrases, exact_query_norm = extract_keywords_and_phrases(question)
    raw_passages = split_text_into_passages(page.text)

    scored_passages: List[EvidencePassage] = []

    for idx, raw_text in enumerate(raw_passages):
        score = score_passage(raw_text, keywords, phrases, exact_query_norm)
        if score >= min_score:
            scored_passages.append(
                EvidencePassage(
                    text=raw_text,
                    source_url=page.final_url or page.original_url,
                    page_title=page.title or "",
                    score=score,
                    passage_index=idx,
                )
            )

    # Deterministic tie-breaking:
    # 1. Higher score first (-score)
    # 2. Longer passage first (-len)
    # 3. Source URL alphabetical
    # 4. Original passage index on page
    scored_passages.sort(
        key=lambda p: (-p.score, -len(p.text), p.source_url, p.passage_index)
    )

    return scored_passages[:max_passages]


def extract_evidence(
    question: str,
    retrieval_input: Union[RetrievalResult, List[FetchedPage]],
    max_passages_per_page: Optional[int] = None,
    max_total_passages: Optional[int] = None,
    min_score: float = 0.0,
) -> List[EvidencePassage]:
    """
    Extract, score, and rank supporting evidence passages across multiple retrieved web pages.

    Args:
        question: The user query or question to ground evidence against.
        retrieval_input: Either a RetrievalResult object or a list of FetchedPage objects.
        max_passages_per_page: Limit on top passages extracted per source page.
        max_total_passages: Global limit on final returned evidence passages.
        min_score: Minimum relevance score threshold (default 0.0 includes all parsed passages).

    Returns:
        Deterministically sorted and deduplicated list of EvidencePassage items.
    """
    max_passages_per_page = (
        max_passages_per_page
        if max_passages_per_page is not None
        else settings.MAX_PASSAGES_PER_PAGE
    )
    max_total_passages = (
        max_total_passages
        if max_total_passages is not None
        else settings.MAX_TOTAL_PASSAGES
    )

    if not question or not question.strip():
        return []

    # Unpack input pages
    if isinstance(retrieval_input, RetrievalResult):
        pages = retrieval_input.pages
    elif isinstance(retrieval_input, list):
        pages = retrieval_input
    else:
        logger.warning(f"Unsupported retrieval input type: {type(retrieval_input)}")
        return []

    all_candidate_passages: List[EvidencePassage] = []

    for page in pages:
        page_passages = extract_evidence_from_page(
            page=page,
            question=question,
            max_passages=max_passages_per_page,
            min_score=min_score,
        )
        all_candidate_passages.extend(page_passages)

    # Cross-page deduplication (retain highest-scoring unique passage)
    seen_norm: Set[str] = set()
    deduplicated: List[EvidencePassage] = []

    # Sort globally before deduplication so highest score for identical text is retained
    all_candidate_passages.sort(
        key=lambda p: (-p.score, -len(p.text), p.source_url, p.passage_index)
    )

    for passage in all_candidate_passages:
        norm_key = normalize_text_for_matching(passage.text)
        if norm_key and norm_key not in seen_norm:
            seen_norm.add(norm_key)
            deduplicated.append(passage)
            if len(deduplicated) >= max_total_passages:
                break

    return deduplicated
