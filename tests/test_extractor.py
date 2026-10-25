"""
Unit tests for the SourceCheck AI evidence extraction and relevance ranking layer.

Covers:
- Normal passage extraction and sentence clustering.
- Empty and malformed page content handling.
- Within-page and cross-page passage deduplication.
- Keyword overlap, multi-word phrase matching, and exact query phrase bonuses.
- Stable, deterministic tie-breaking for identical scores.
- Configurable per-page and total passage limits.
- Preservation of source URLs and page titles.
- Unicode support and punctuation normalization.
- Queries with zero matching keywords.
"""

import unittest

from app.extractor import (
    extract_evidence,
    extract_evidence_from_page,
    extract_keywords_and_phrases,
    normalize_text_for_matching,
    score_passage,
    split_text_into_passages,
)
from app.schemas import EvidencePassage, FetchedPage, RetrievalResult


class TestTextSplitting(unittest.TestCase):
    """Tests for breaking cleaned page text into coherent evidence passages."""

    def test_split_normal_paragraphs(self) -> None:
        text = (
            "AMD ROCm is an open-source software development platform for GPU computing.\n\n"
            "It supports machine learning frameworks such as PyTorch, TensorFlow, and JAX.\n\n"
            "AMD Instinct accelerators deliver high performance for large language models."
        )
        passages = split_text_into_passages(text, min_chars=30, max_chars=500)
        self.assertEqual(len(passages), 3)
        self.assertIn("AMD ROCm is an open-source software development platform", passages[0])
        self.assertIn("PyTorch, TensorFlow, and JAX", passages[1])
        self.assertIn("AMD Instinct accelerators", passages[2])

    def test_split_long_paragraph_into_clusters(self) -> None:
        # Construct a paragraph with multiple sentences exceeding max_chars
        sentences = [
            f"Sentence number {i} provides detailed factual context about GPU compute architecture."
            for i in range(15)
        ]
        long_paragraph = " ".join(sentences)
        self.assertGreater(len(long_paragraph), 1000)

        passages = split_text_into_passages(long_paragraph, min_chars=50, max_chars=350)
        self.assertGreaterEqual(len(passages), 3)
        for p in passages:
            self.assertLessEqual(len(p), 400)
            self.assertGreaterEqual(len(p), 50)

    def test_discard_short_boilerplate_noise(self) -> None:
        text = "Menu\n\nPrivacy Policy\n\nThis is a legitimate informational passage that exceeds the minimum character count.\n\nHome"
        passages = split_text_into_passages(text, min_chars=40)
        self.assertEqual(len(passages), 1)
        self.assertIn("legitimate informational passage", passages[0])

    def test_empty_and_whitespace_text(self) -> None:
        self.assertEqual(split_text_into_passages(""), [])
        self.assertEqual(split_text_into_passages("   \n\n\t   "), [])

    def test_within_page_deduplication(self) -> None:
        duplicate_text = (
            "AMD ROCm is an open-source software development platform for GPU computing.\n\n"
            "AMD ROCm is an open-source software development platform for GPU computing."
        )
        passages = split_text_into_passages(duplicate_text, min_chars=30)
        self.assertEqual(len(passages), 1)


class TestRelevanceRanking(unittest.TestCase):
    """Tests for deterministic scoring and keyword/phrase matching."""

    def test_keyword_overlap_ranking(self) -> None:
        question = "What architecture does AMD Instinct MI300 use?"
        keywords, phrases, exact_norm = extract_keywords_and_phrases(question)

        # Passage A matches 3 key terms: "AMD", "Instinct", "MI300", "architecture"
        passage_a = "The AMD Instinct MI300 series accelerator is built on the advanced CDNA 3 architecture."
        # Passage B matches only 1 term: "architecture"
        passage_b = "Modern computing systems require efficient power delivery and memory architecture."

        score_a = score_passage(passage_a, keywords, phrases, exact_norm)
        score_b = score_passage(passage_b, keywords, phrases, exact_norm)

        self.assertGreater(score_a, score_b)
        self.assertGreater(score_b, 0.0)

    def test_phrase_match_bonus(self) -> None:
        question = "AMD ROCm compatibility"
        keywords, phrases, exact_norm = extract_keywords_and_phrases(question)

        # Passage A contains consecutive bigram "AMD ROCm"
        passage_a = "Official documentation outlines AMD ROCm compatibility across various Linux distributions."
        # Passage B contains "AMD" and "compatibility" but separated with no bigram
        passage_b = "AMD provides hardware drivers, while software tools ensure application compatibility."

        score_a = score_passage(passage_a, keywords, phrases, exact_norm)
        score_b = score_passage(passage_b, keywords, phrases, exact_norm)

        self.assertGreater(score_a, score_b)

    def test_exact_query_phrase_bonus(self) -> None:
        question = "ROCm PyTorch container"
        keywords, phrases, exact_norm = extract_keywords_and_phrases(question)

        # Passage A has exact question phrase verbatim
        passage_a = "To run deep learning workloads, pull the official ROCm PyTorch container from Docker Hub."
        # Passage B has the words scattered
        passage_b = "Containers provide isolation. ROCm supports deep learning with the PyTorch framework."

        score_a = score_passage(passage_a, keywords, phrases, exact_norm)
        score_b = score_passage(passage_b, keywords, phrases, exact_norm)

        self.assertGreater(score_a, score_b)

    def test_zero_keyword_overlap(self) -> None:
        question = "Quantum computing algorithms"
        keywords, phrases, exact_norm = extract_keywords_and_phrases(question)

        unrelated_passage = "The quick brown fox jumps over the lazy dog in the middle of summer."
        score = score_passage(unrelated_passage, keywords, phrases, exact_norm)

        self.assertEqual(score, 0.0)

    def test_unicode_and_punctuation_handling(self) -> None:
        question = "Café au lait in Paris, France?"
        norm_q = normalize_text_for_matching(question)
        self.assertIn("café", norm_q)
        self.assertIn("lait", norm_q)
        self.assertNotIn("?", norm_q)
        self.assertNotIn(",", norm_q)

        keywords, phrases, exact_norm = extract_keywords_and_phrases(question)
        passage = "Enjoying a traditional café au lait at a Parisian bistro is a quintessential experience."
        score = score_passage(passage, keywords, phrases, exact_norm)
        self.assertGreater(score, 0.0)

    def test_stable_tie_breaking(self) -> None:
        # Two passages with the exact same score from different URLs and lengths
        p1 = EvidencePassage(
            text="Short text about ROCm platform computing.",
            source_url="https://example.com/b",
            page_title="Title B",
            score=3.5,
            passage_index=0,
        )
        p2 = EvidencePassage(
            text="A somewhat longer text about ROCm platform computing with extra words.",
            source_url="https://example.com/a",
            page_title="Title A",
            score=3.5,
            passage_index=0,
        )

        passages = [p1, p2]
        # Sort key: (-score, -len(text), url, index)
        passages.sort(key=lambda p: (-p.score, -len(p.text), p.source_url, p.passage_index))

        # p2 has longer text, so it should rank first among identical scores
        self.assertEqual(passages[0].source_url, "https://example.com/a")
        self.assertEqual(passages[1].source_url, "https://example.com/b")


class TestEvidenceExtractionPipeline(unittest.TestCase):
    """Tests for high-level extract_evidence() and extract_evidence_from_page()."""

    def test_extract_evidence_normal_flow(self) -> None:
        page_1 = FetchedPage(
            original_url="https://example.com/doc1",
            final_url="https://example.com/doc1",
            title="AMD CDNA 3 Architecture",
            text=(
                "AMD CDNA 3 architecture powers the AMD Instinct MI300 accelerators.\n\n"
                "It introduces unified memory architecture and matrix math cores."
            ),
            success=True,
        )
        page_2 = FetchedPage(
            original_url="https://example.com/doc2",
            final_url="https://example.com/doc2",
            title="ROCm Software Stack",
            text=(
                "The ROCm open software platform includes drivers, compilers, and libraries.\n\n"
                "ROCm 6.2 adds enhanced performance for transformer-based inference workloads."
            ),
            success=True,
        )

        retrieval_result = RetrievalResult(
            query="CDNA 3 architecture",
            search_results=[],
            pages=[page_1, page_2],
        )

        evidence = extract_evidence(
            question="What is CDNA 3 architecture?",
            retrieval_input=retrieval_result,
            max_total_passages=5,
        )

        self.assertGreater(len(evidence), 0)
        # Top passage should be from page 1 about CDNA 3
        top = evidence[0]
        self.assertEqual(top.source_url, "https://example.com/doc1")
        self.assertEqual(top.page_title, "AMD CDNA 3 Architecture")
        self.assertIn("CDNA 3", top.text)
        self.assertGreater(top.score, 0.0)

    def test_empty_and_failed_pages_handling(self) -> None:
        failed_page = FetchedPage(
            original_url="https://example.com/404",
            final_url="https://example.com/404",
            title="",
            text="",
            success=False,
            error="HTTP 404",
        )
        evidence = extract_evidence("AMD query", [failed_page])
        self.assertEqual(evidence, [])

        empty_query_evidence = extract_evidence("   ", [failed_page])
        self.assertEqual(empty_query_evidence, [])

    def test_cross_page_deduplication(self) -> None:
        shared_text = "This exact verbatim statement is syndicated across multiple mirror websites."
        page_1 = FetchedPage(
            original_url="https://site1.com/page",
            final_url="https://site1.com/page",
            title="Site 1",
            text=shared_text,
            success=True,
        )
        page_2 = FetchedPage(
            original_url="https://site2.com/page",
            final_url="https://site2.com/page",
            title="Site 2",
            text=shared_text,
            success=True,
        )

        evidence = extract_evidence(
            question="syndicated statement",
            retrieval_input=[page_1, page_2],
            max_total_passages=5,
        )

        # Should only retain one instance of the identical text
        self.assertEqual(len(evidence), 1)

    def test_limits_enforced(self) -> None:
        paragraphs = [
            f"Paragraph {i} contains relevant keywords regarding ROCm compute optimization and libraries."
            for i in range(10)
        ]
        page = FetchedPage(
            original_url="https://example.com/long",
            final_url="https://example.com/long",
            title="Long Page",
            text="\n\n".join(paragraphs),
            success=True,
        )

        # Enforce max 3 per page and max 2 total
        evidence = extract_evidence(
            question="ROCm compute optimization",
            retrieval_input=[page],
            max_passages_per_page=3,
            max_total_passages=2,
        )

        self.assertEqual(len(evidence), 2)

    def test_question_with_no_matches_in_pages(self) -> None:
        page = FetchedPage(
            original_url="https://example.com/plants",
            final_url="https://example.com/plants",
            title="Botany Guide",
            text="Photosynthesis converts light energy into chemical energy in green plants.",
            success=True,
        )

        # When min_score > 0, completely unrelated questions yield empty results
        evidence = extract_evidence(
            question="AMD ROCm GPU benchmarks",
            retrieval_input=[page],
            min_score=0.1,
        )
        self.assertEqual(evidence, [])

        # When min_score == 0.0, it can return the passage with score 0.0
        evidence_zero = extract_evidence(
            question="AMD ROCm GPU benchmarks",
            retrieval_input=[page],
            min_score=0.0,
        )
        self.assertEqual(len(evidence_zero), 1)
        self.assertEqual(evidence_zero[0].score, 0.0)


if __name__ == "__main__":
    unittest.main()
