"""
Unit tests for the SourceCheck AI core web retrieval engine.

Covers:
- HTML title and text extraction (standard, empty, malformed, truncated)
- URL validation and SSRF protection (localhost, private IPs, link-local metadata, decimal notation)
- Redirect validation and redirect-based SSRF defense
- HTTP error statuses, network timeouts, and response body size limits
- Search provider functionality, empty results, and error handling
- End-to-end retrieval with URL deduplication, result limits, and source metadata preservation
"""

import unittest
import httpx

from app.retrieval import (
    DuckDuckGoSearchProvider,
    MockSearchProvider,
    SecurityError,
    SearchError,
    clean_html,
    describe_http_status,
    fetch_page,
    get_search_provider,
    is_blocked_status,
    retrieve,
    search_web,
    validate_url,
)
from app.schemas import SearchResult


class TestHtmlCleaning(unittest.TestCase):
    """Tests for HTML parsing, title extraction, and text cleaning."""

    def test_clean_html_standard(self) -> None:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AMD ROCm Documentation</title>
            <script>console.log("malicious code");</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/home">Home</a> | <a href="/docs">Docs</a></nav>
            <h1>ROCm Platform Overview</h1>
            <p>AMD ROCm is an open-source software stack for GPU computing.</p>
            <p>It supports PyTorch and TensorFlow workloads on AMD Instinct accelerators.</p>
            <footer>Copyright 2026 AMD Inc. All rights reserved.</footer>
        </body>
        </html>
        """
        title, text = clean_html(html)

        self.assertEqual(title, "AMD ROCm Documentation")
        self.assertIn("ROCm Platform Overview", text)
        self.assertIn("AMD ROCm is an open-source software stack for GPU computing.", text)
        self.assertIn("It supports PyTorch and TensorFlow workloads on AMD Instinct accelerators.", text)

        # Ensure boilerplate and executable tags were removed
        self.assertNotIn("console.log", text)
        self.assertNotIn("color: red", text)
        self.assertNotIn("Home | Docs", text)
        self.assertNotIn("Copyright 2026", text)

    def test_clean_html_drops_menus_and_keeps_paragraphs(self) -> None:
        html = """
        <html><head><title>hi | Cambridge Dictionary</title></head><body>
          <div class="hdr-menu"><ul><li>Translator tool</li><li>Browse</li><li>HFE</li></ul></div>
          <div id="sidebar"><p>Word of the Day: petrified, extremely frightened. Read More</p></div>
          <div role="navigation"><a href="/">Home</a></div>
          <div aria-hidden="true">Log out English (UK) Change</div>
          <main>
            <h1>hi</h1>
            <p>Used as an <b>informal</b> greeting, usually to people who you know.</p>
            <p>Written abbreviation for Hawaii, used in addresses.</p>
          </main>
        </body></html>
        """
        _, text = clean_html(html)
        self.assertIn("Used as an informal greeting, usually to people who you know.", text)
        self.assertNotIn(" ,", text)
        self.assertIn("Written abbreviation for Hawaii, used in addresses.", text)
        for junk in ("Translator tool", "Browse", "Word of the Day", "Home", "Log out"):
            self.assertNotIn(junk, text)
        # Paragraphs stay separate so the passage splitter can rank them individually.
        self.assertIn("people who you know.\n\nWritten abbreviation", text)

    def test_clean_html_without_main_still_separates_blocks(self) -> None:
        html = "<html><body><div><p>First paragraph here.</p><p>Second paragraph here.</p></div></body></html>"
        _, text = clean_html(html)
        self.assertEqual(text, "First paragraph here.\n\nSecond paragraph here.")

    def test_clean_html_fallback_to_h1(self) -> None:
        html = "<html><body><h1>Fallback Heading</h1><p>Body text here.</p></body></html>"
        title, text = clean_html(html)
        self.assertEqual(title, "Fallback Heading")
        self.assertIn("Body text here.", text)

    def test_clean_html_empty_and_whitespace(self) -> None:
        title, text = clean_html("")
        self.assertEqual(title, "")
        self.assertEqual(text, "")

        title, text = clean_html("   \n\t  ")
        self.assertEqual(title, "")
        self.assertEqual(text, "")

    def test_clean_html_malformed(self) -> None:
        malformed = "<p>Unclosed paragraph <b>Bold without end <i>Italic<br><div>Block</div>"
        title, text = clean_html(malformed)
        self.assertEqual(title, "")
        self.assertIn("Unclosed paragraph", text)
        self.assertIn("Bold without end", text)
        self.assertIn("Block", text)

    def test_clean_html_truncation(self) -> None:
        long_html = "<html><body><p>" + ("A" * 1000) + "</p></body></html>"
        _, text = clean_html(long_html, max_text_length=100)
        self.assertTrue(text.endswith("... [truncated]"))
        self.assertLessEqual(len(text), 120)


class TestUrlValidationAndSSRF(unittest.TestCase):
    """Tests for URL safety, protocol restrictions, and SSRF prevention."""

    def test_valid_public_urls(self) -> None:
        self.assertEqual(validate_url("https://example.com"), "https://example.com")
        self.assertEqual(validate_url("http://amd.com/en/products"), "http://amd.com/en/products")
        self.assertEqual(
            validate_url("https://docs.python.org/3/library/"),
            "https://docs.python.org/3/library/",
        )

    def test_invalid_schemes(self) -> None:
        forbidden = [
            "ftp://files.example.com",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<html></html>",
            "gopher://gopher.floodgap.com",
        ]
        for url in forbidden:
            with self.subTest(url=url):
                with self.assertRaises(SecurityError):
                    validate_url(url)

    def test_ssrf_blocked_localhost(self) -> None:
        localhost_targets = [
            "http://localhost",
            "http://localhost:8000/admin",
            "http://127.0.0.1",
            "http://127.0.0.1:5000/metrics",
            "http://[::1]:8080",
            "http://app.localhost",
            "http://internal.service.local",
            "http://router.home",
        ]
        for url in localhost_targets:
            with self.subTest(url=url):
                with self.assertRaises(SecurityError):
                    validate_url(url)

    def test_ssrf_blocked_private_networks(self) -> None:
        private_ips = [
            "http://10.0.0.1",
            "http://10.254.0.1:8080",
            "http://172.16.0.1",
            "http://192.168.1.1/router",
            "http://0.0.0.0:8000",
        ]
        for url in private_ips:
            with self.subTest(url=url):
                with self.assertRaises(SecurityError):
                    validate_url(url)

    def test_ssrf_blocked_cloud_metadata(self) -> None:
        # 169.254.169.254 is the standard AWS, GCP, Azure, OpenStack instance metadata service
        metadata_url = "http://169.254.169.254/latest/meta-data/"
        with self.assertRaises(SecurityError):
            validate_url(metadata_url)

    def test_ssrf_blocked_decimal_ip(self) -> None:
        # 2130706433 is the decimal representation of 127.0.0.1
        decimal_url = "http://2130706433/"
        with self.assertRaises(SecurityError):
            validate_url(decimal_url)


class TestPageFetching(unittest.TestCase):
    """Tests for fetch_page with mocked HTTP transports."""

    def test_successful_page_fetch(self) -> None:
        target_url = "https://example.com/article"

        def mock_handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), target_url)
            html_content = (
                "<html><head><title>Test Article</title></head>"
                "<body><p>Fact: AMD ROCm 6.2 was released in 2024.</p></body></html>"
            )
            return httpx.Response(200, text=html_content)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page(target_url, client=client)

        self.assertTrue(page.success)
        self.assertEqual(page.original_url, target_url)
        self.assertEqual(page.final_url, target_url)
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.title, "Test Article")
        self.assertIn("AMD ROCm 6.2 was released in 2024.", page.text)
        self.assertIsNone(page.error)

    def test_fetch_ssrf_url_blocked_before_request(self) -> None:
        # Verify that SSRF destination returns a failed FetchedPage without sending a request
        request_made = False

        def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_made
            request_made = True
            return httpx.Response(200, text="internal data")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("http://127.0.0.1:8000/secret", client=client)

        self.assertFalse(page.success)
        self.assertFalse(request_made)
        self.assertIn("blocked", page.error.lower())

    def test_safe_redirect_followed(self) -> None:
        start_url = "https://example.com/short"
        dest_url = "https://example.com/long-canonical-article"

        def mock_handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == start_url:
                return httpx.Response(301, headers={"Location": dest_url})
            elif str(request.url) == dest_url:
                return httpx.Response(200, text="<html><head><title>Destination</title></head><body>Done</body></html>")
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page(start_url, client=client)

        self.assertTrue(page.success)
        self.assertEqual(page.original_url, start_url)
        self.assertEqual(page.final_url, dest_url)
        self.assertEqual(page.title, "Destination")
        self.assertEqual(page.text, "Done")

    def test_redirect_to_internal_ip_blocked(self) -> None:
        # Critical SSRF test: public website attempts to redirect to local internal service
        start_url = "https://example.com/redirect-to-internal"
        malicious_dest = "http://169.254.169.254/latest/meta-data/"

        def mock_handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == start_url:
                return httpx.Response(302, headers={"Location": malicious_dest})
            return httpx.Response(200, text="leaked credentials")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page(start_url, client=client)

        self.assertFalse(page.success)
        self.assertIn("redirect blocked by security check", page.error.lower())
        self.assertEqual(page.final_url, malicious_dest)

    def test_max_redirects_exceeded(self) -> None:
        def mock_handler(request: httpx.Request) -> httpx.Response:
            # Continual loop redirect
            return httpx.Response(302, headers={"Location": "https://example.com/loop"})

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/loop", client=client, max_redirects=3)

        self.assertFalse(page.success)
        self.assertIn("exceeded maximum redirect limit", page.error.lower())

    def test_http_404_error(self) -> None:
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/missing", client=client)

        self.assertFalse(page.success)
        self.assertEqual(page.status_code, 404)
        self.assertIn("404", page.error)

    def test_http_timeout_error(self) -> None:
        def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Timed out reading data")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/slow", client=client, max_retries=0)

        self.assertFalse(page.success)
        self.assertIn("timed out", page.error.lower())

    def test_response_size_limit_exceeded(self) -> None:
        # Verify that response body larger than max_bytes is rejected
        def mock_handler(request: httpx.Request) -> httpx.Response:
            huge_content = "X" * 5000
            return httpx.Response(200, text=huge_content)

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/huge", client=client, max_bytes=1000)

        self.assertFalse(page.success)
        self.assertIn("exceeds limit", page.error.lower())

    def test_unsupported_content_type(self) -> None:
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Type": "application/zip"}, content=b"PK...")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/file.zip", client=client)

        self.assertFalse(page.success)
        self.assertIn("unsupported content type", page.error.lower())


class TestBlockedStatusHandling(unittest.TestCase):
    """401/403/429/451 mean "you are not welcome here", not "this fetch broke"."""

    def test_blocked_status_classification(self) -> None:
        for code in (401, 403, 429, 451):
            self.assertTrue(is_blocked_status(code), code)
        for code in (200, 404, 418, 500, 503, None):
            self.assertFalse(is_blocked_status(code), code)

    def test_refusal_wording_for_blocked_statuses(self) -> None:
        message = describe_http_status(403)
        self.assertIn("403", message)
        self.assertIn("refused", message.lower())
        self.assertNotIn("error", message.lower())

    def test_403_page_is_worded_as_a_refusal(self) -> None:
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="Forbidden")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://www.merriam-webster.com/dictionary/hi", client=client)

        self.assertFalse(page.success)
        self.assertEqual(page.status_code, 403)
        self.assertIn("refused", page.error.lower())
        self.assertIn("403", page.error)

    def test_500_page_keeps_failure_wording(self) -> None:
        # Ordinary failures must keep sounding like failures - only refusals soften.
        def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = httpx.Client(transport=httpx.MockTransport(mock_handler))
        page = fetch_page("https://example.com/broken", client=client)

        self.assertFalse(page.success)
        self.assertEqual(page.status_code, 500)
        self.assertIn("returned status 500", page.error)


class TestSearchProviders(unittest.TestCase):
    """Tests for SearchProvider implementations, empty results, and search error handling."""

    def test_mock_search_provider(self) -> None:
        mock_provider = MockSearchProvider({
            "AMD ROCm": [
                SearchResult(
                    title="ROCm Ecosystem",
                    url="https://rocm.docs.amd.com",
                    snippet="Official ROCm docs.",
                )
            ]
        })
        results = mock_provider.search("AMD ROCm")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "ROCm Ecosystem")
        self.assertEqual(results[0].url, "https://rocm.docs.amd.com")

        # Fallback query
        unknown_results = mock_provider.search("custom query", max_results=1)
        self.assertEqual(len(unknown_results), 1)
        self.assertIn("custom query", unknown_results[0].title)

    def test_empty_search_results(self) -> None:
        mock_provider = MockSearchProvider(predefined_results={"empty query": []})
        results = mock_provider.search("empty query")
        self.assertEqual(results, [])

        # Empty string query
        empty_str_results = mock_provider.search("   ")
        self.assertEqual(empty_str_results, [])

    def test_duckduckgo_search_success_mocked(self) -> None:
        sample_ddg_html = """
        <html><body>
            <div class="result__body">
                <h2 class="result__title">
                    <a href="https://example.com/target-article">ROCm 6 Overview</a>
                </h2>
                <div class="result__snippet">ROCm is an open computing platform.</div>
            </div>
        </body></html>
        """

        def mock_ddg(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://html.duckduckgo.com/html/")
            return httpx.Response(200, text=sample_ddg_html)

        client = httpx.Client(transport=httpx.MockTransport(mock_ddg))
        provider = DuckDuckGoSearchProvider(client=client)

        results = provider.search("ROCm 6", max_results=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "ROCm 6 Overview")
        self.assertEqual(results[0].url, "https://example.com/target-article")
        self.assertEqual(results[0].snippet, "ROCm is an open computing platform.")

    def test_duckduckgo_search_error(self) -> None:
        def mock_ddg_error(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = httpx.Client(transport=httpx.MockTransport(mock_ddg_error))
        provider = DuckDuckGoSearchProvider(client=client)

        with self.assertRaises(SearchError):
            provider.search("failing query")

    def test_get_search_provider_factory(self) -> None:
        mock_p = get_search_provider("mock")
        self.assertIsInstance(mock_p, MockSearchProvider)

        ddg_p = get_search_provider("duckduckgo")
        self.assertIsInstance(ddg_p, DuckDuckGoSearchProvider)

        with self.assertRaises(ValueError):
            get_search_provider("non_existent_engine")


class TestEndToEndRetrieval(unittest.TestCase):
    """Tests for the high-level retrieve() workflow."""

    def test_retrieve_pipeline_with_mock_provider(self) -> None:
        url_1 = "https://example.com/page1"
        url_2 = "https://example.com/page2"

        provider = MockSearchProvider({
            "AMD GPU Architecture": [
                SearchResult(title="Page 1", url=url_1, snippet="Snippet 1"),
                SearchResult(title="Page 2", url=url_2, snippet="Snippet 2"),
            ]
        })

        def mock_http(request: httpx.Request) -> httpx.Response:
            if str(request.url) == url_1:
                return httpx.Response(
                    200,
                    text="<html><head><title>Page 1 Title</title></head><body>CDNA 3 architecture details.</body></html>",
                )
            elif str(request.url) == url_2:
                return httpx.Response(
                    200,
                    text="<html><head><title>Page 2 Title</title></head><body>MI300 accelerator specifications.</body></html>",
                )
            return httpx.Response(404)

        client = httpx.Client(transport=httpx.MockTransport(mock_http))

        retrieval_res = retrieve(
            query="AMD GPU Architecture",
            max_search_results=2,
            max_pages_to_fetch=2,
            provider=provider,
            client=client,
        )

        self.assertEqual(retrieval_res.query, "AMD GPU Architecture")
        self.assertEqual(len(retrieval_res.search_results), 2)
        self.assertEqual(len(retrieval_res.pages), 2)

        # Verify page 1 metadata and content
        p1 = retrieval_res.pages[0]
        self.assertTrue(p1.success)
        self.assertEqual(p1.original_url, url_1)
        self.assertEqual(p1.final_url, url_1)
        self.assertEqual(p1.title, "Page 1 Title")
        self.assertIn("CDNA 3 architecture details.", p1.text)

        # Verify page 2 metadata and content
        p2 = retrieval_res.pages[1]
        self.assertTrue(p2.success)
        self.assertEqual(p2.original_url, url_2)
        self.assertEqual(p2.final_url, url_2)
        self.assertEqual(p2.title, "Page 2 Title")
        self.assertIn("MI300 accelerator specifications.", p2.text)

    def test_retrieve_pipeline_deduplicates_urls(self) -> None:
        # If search returns duplicate URLs, the pipeline must only fetch once
        dup_url = "https://example.com/same-page"
        provider = MockSearchProvider({
            "duplicate query": [
                SearchResult(title="Result 1", url=dup_url, snippet="Snippet 1"),
                SearchResult(title="Result 2 (Duplicate)", url=dup_url, snippet="Snippet 2"),
            ]
        })

        fetch_count = 0

        def mock_http(request: httpx.Request) -> httpx.Response:
            nonlocal fetch_count
            fetch_count += 1
            return httpx.Response(200, text="<html><body>Deduplicated content.</body></html>")

        client = httpx.Client(transport=httpx.MockTransport(mock_http))

        res = retrieve("duplicate query", provider=provider, client=client)

        self.assertEqual(len(res.search_results), 2)
        self.assertEqual(len(res.pages), 1)  # Only 1 unique page fetched
        self.assertEqual(fetch_count, 1)

    def test_retrieve_pipeline_limits_and_empty_query(self) -> None:
        # Empty query handling
        empty_res = retrieve("   ")
        self.assertEqual(empty_res.query, "   ")
        self.assertEqual(len(empty_res.search_results), 0)
        self.assertEqual(len(empty_res.pages), 0)

        # Max pages limit enforcement
        provider = MockSearchProvider({
            "many results": [
                SearchResult(title=f"Page {i}", url=f"https://example.com/p{i}")
                for i in range(5)
            ]
        })

        def mock_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html><body>Sample body.</body></html>")

        client = httpx.Client(transport=httpx.MockTransport(mock_http))
        limited_res = retrieve(
            "many results",
            max_search_results=5,
            max_pages_to_fetch=2,
            provider=provider,
            client=client,
        )

        self.assertEqual(len(limited_res.search_results), 5)
        self.assertEqual(len(limited_res.pages), 2)  # Capped at max_pages_to_fetch


if __name__ == "__main__":
    unittest.main()
