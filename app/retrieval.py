"""
Web search and content retrieval module for SourceCheck AI.

Handles:
- Configurable web search providers (DuckDuckGo, Mock).
- Safe HTTP page fetching with SSRF protection, explicit timeouts, bounded retries, and redirect validation.
- Clean text and title extraction using BeautifulSoup.

SECURITY NOTICE & RESIDUAL LIMITATIONS:
This module implements robust pre-flight SSRF defenses by validating URL protocols,
blocking loopback, private, link-local, and reserved IP ranges (including decimal/numeric representations),
enforcing blocked internal domain names, and resolving DNS records before connecting.
However, because standard user-space HTTP clients (such as HTTPX or Requests) perform their own internal
DNS lookup during socket establishment, a Time-of-Check to Time-of-Use (TOCTOU) DNS rebinding window exists
if an attacker controls an authoritative nameserver with 0-second TTL.
Complete network-level isolation against DNS rebinding requires infrastructure-level controls such as an egress
proxy, network firewall, or custom socket-level IP pinning.
All extracted content from web pages is treated as UNTRUSTED external data and must never be interpreted
as trusted system prompt instructions.
"""

import abc
import ipaddress
import logging
import re
import socket
import time
import urllib.parse
from typing import List, Optional, Set, Tuple

import bs4
import httpx

from app.config import settings
from app.schemas import FetchedPage, RetrievalResult, SearchResult

logger = logging.getLogger(__name__)


# ==============================================================================
# Exceptions
# ==============================================================================


class RetrievalError(Exception):
    """Base exception for web retrieval errors."""


class SecurityError(RetrievalError):
    """Raised when a URL violates security policies (e.g., SSRF or forbidden schemes)."""


class SearchError(RetrievalError):
    """Raised when web search fails."""


class FetchError(RetrievalError):
    """Raised when web page fetching fails."""


# ==============================================================================
# Security & URL Validation
# ==============================================================================


def is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address belongs to private, loopback, link-local, or reserved ranges."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url(url: str, block_ssrf: bool = True) -> str:
    """
    Validate that a URL uses safe protocols and does not target internal network destinations.

    Args:
        url: The candidate URL string to validate.
        block_ssrf: Whether to enforce anti-SSRF checks against internal/private destinations.

    Returns:
        The validated URL string.

    Raises:
        SecurityError: If the scheme is unsupported or destination points to an internal/private host.
    """
    if not url or not isinstance(url, str):
        raise SecurityError("URL must be a non-empty string.")

    parsed = urllib.parse.urlsplit(url.strip())

    # Only permit HTTP and HTTPS schemes
    if parsed.scheme.lower() not in ("http", "https"):
        raise SecurityError(
            f"Unsupported URL scheme: '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SecurityError("URL is missing a valid hostname.")

    hostname = hostname.lower().strip()

    if block_ssrf:
        # 1. Block well-known internal hostnames directly
        blocked_names = {"localhost", "localhost.localdomain"}
        if (
            hostname in blocked_names
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
            or hostname.endswith(".lan")
            or hostname.endswith(".home")
        ):
            raise SecurityError(f"Access to internal hostname '{hostname}' is blocked.")

        # 2. Check for integer / decimal IP representation (e.g., 2130706433 for 127.0.0.1)
        if hostname.isdigit():
            try:
                dec_ip = ipaddress.ip_address(int(hostname))
                if is_blocked_ip(dec_ip):
                    raise SecurityError(
                        f"Direct connection to private/local IP address '{hostname}' ({dec_ip}) is blocked."
                    )
                return url.strip()
            except ValueError:
                pass

        # 3. Check if hostname is an IP literal (IPv4 or IPv6)
        try:
            ip_obj = ipaddress.ip_address(hostname)
            if is_blocked_ip(ip_obj):
                raise SecurityError(
                    f"Direct connection to private/local IP address '{hostname}' is blocked."
                )
            return url.strip()
        except ValueError:
            # Not an IP literal; proceed to DNS resolution check
            pass

        # 4. Resolve hostname via DNS to verify target IP addresses
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip_str)
                if is_blocked_ip(ip_obj):
                    raise SecurityError(
                        f"Hostname '{hostname}' resolves to blocked private/local IP '{ip_str}'."
                    )
        except socket.gaierror:
            # Domain could not be resolved at validation time (e.g., offline test or non-existent host).
            # Live requests will fail safely during connection.
            pass

    return url.strip()


# ==============================================================================
# HTML Cleaning & Text Extraction
# ==============================================================================


_UNWANTED_TAGS = [
    "script", "style", "noscript", "template", "nav", "footer", "header", "aside",
    "form", "svg", "iframe", "button", "select", "img", "picture", "video",
    "audio", "source", "canvas", "dialog", "menu",
]

_BLOCK_TAGS = [
    "p", "div", "section", "article", "main", "li", "ul", "ol", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "tr", "td",
    "th", "figure", "figcaption", "hr", "address",
]

_BOILERPLATE_ROLES = frozenset(
    {"navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar", "dialog"}
)

# Whole class/id tokens (split on - and _) that mark site chrome, not content.
_BOILERPLATE_TOKEN = re.compile(
    r"(?:^|[-_])(?:nav|navbar|navigation|menu|sidebar|breadcrumbs?|cookies?|"
    r"footer|social|share|sharing|newsletter|popup|modal|advert|advertisement|"
    r"ads?|promo|skip|toolbar)(?:[-_]|$)",
    re.IGNORECASE,
)

_KEEP_TAGS = frozenset({"html", "body", "main", "article"})


def _is_boilerplate(tag: "bs4.element.Tag") -> bool:
    """True for elements that are hidden or clearly navigation / site chrome."""
    if not isinstance(tag, bs4.element.Tag) or tag.name in _KEEP_TAGS or tag.attrs is None:
        return False
    if tag.get("role", "").strip().lower() in _BOILERPLATE_ROLES:
        return True
    if tag.has_attr("hidden") or tag.get("aria-hidden", "").strip().lower() == "true":
        return True
    tokens = list(tag.get("class") or [])
    if tag.get("id"):
        tokens.append(tag.get("id"))
    return any(_BOILERPLATE_TOKEN.search(str(token)) for token in tokens)


def _main_content_root(soup: "bs4.BeautifulSoup") -> "bs4.element.Tag":
    """The largest <main> / [role=main] / <article> with real text, else <body>."""
    candidates = soup.find_all(["main", "article"]) + soup.find_all(attrs={"role": "main"})
    best, best_len = None, 0
    for candidate in candidates:
        length = len(candidate.get_text(" ", strip=True))
        if length > best_len:
            best, best_len = candidate, length
    if best is not None and best_len >= 200:
        return best
    return soup.body or soup


def clean_html(html_content: str, max_text_length: int = 50000) -> Tuple[str, str]:
    """
    Extract page title and readable plain text from raw HTML content using BeautifulSoup.

    Args:
        html_content: Raw HTML text.
        max_text_length: Maximum allowed characters in the returned text.

    Returns:
        A tuple of (page_title, cleaned_text).
    """
    if not html_content or not html_content.strip():
        return "", ""

    soup = bs4.BeautifulSoup(html_content, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        h1 = soup.find("h1")
        if h1 and h1.get_text():
            title = h1.get_text(strip=True)

    # Remove head metadata tag so it is not duplicated in body text
    if soup.head:
        soup.head.decompose()
    elif soup.title:
        soup.title.decompose()

    # Strip script, style, navigation, media, and boilerplate tags
    for tag in soup.find_all(_UNWANTED_TAGS):
        tag.decompose()

    # Strip landmark / hidden / site-chrome containers (menus, sidebars, cookie bars).
    for tag in soup.find_all(_is_boilerplate):
        tag.decompose()

    # Prefer the page's main content region when it has one with real text,
    # so headers, mega-menus and "word of the day" sidebars never reach scoring.
    root = _main_content_root(soup)

    # Paragraph breaks between block elements, line breaks for <br>, so the
    # passage splitter sees real paragraphs instead of one page-long blob.
    for br in root.find_all("br"):
        br.replace_with("\n")
    for block in root.find_all(_BLOCK_TAGS):
        block.insert_before("\n\n")
        block.insert_after("\n\n")

    raw_text = root.get_text(separator=" ")
    blocks = []
    for chunk in re.split(r"\n\s*\n", raw_text):
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in chunk.split("\n")]
        # Inline tags are joined with a space; drop the one that lands before punctuation.
        lines = [re.sub(r" +([,.;:!?)\]])", r"\1", line) for line in lines]
        text_block = "\n".join(line for line in lines if line)
        if text_block:
            blocks.append(text_block)
    cleaned_text = "\n\n".join(blocks).strip()

    # Bound maximum text length
    if len(cleaned_text) > max_text_length:
        cleaned_text = cleaned_text[:max_text_length] + "... [truncated]"

    return title, cleaned_text


# ==============================================================================
# Web Page Fetching
# ==============================================================================

# Statuses that mean "this host refuses *us*" - bot / IP / rate-limit / legal
# protection - rather than "this fetch broke". Expected during bulk retrieval, so
# the agent reports them as informational notices: a single blocked source must
# not read as a failure when other sources answered the question.
BLOCKED_STATUSES = frozenset({401, 403, 429, 451})


def is_blocked_status(status_code: Optional[int]) -> bool:
    """True when a non-200 means the site refused the request, not that it failed."""
    return status_code in BLOCKED_STATUSES


def describe_http_status(status_code: int) -> str:
    """Human-readable, non-alarming wording for a non-200 response."""
    if status_code in BLOCKED_STATUSES:
        return (
            f"HTTP {status_code} - the site refused the request "
            "(bot/IP protection); page skipped."
        )
    return f"HTTP request returned status {status_code}."


def fetch_page(
    url: str,
    client: Optional[httpx.Client] = None,
    timeout: Optional[float] = None,
    connect_timeout: Optional[float] = None,
    read_timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    max_redirects: Optional[int] = None,
    max_bytes: Optional[int] = None,
    max_text_length: Optional[int] = None,
    block_ssrf: Optional[bool] = None,
) -> FetchedPage:
    """
    Safely retrieve a single web page, following redirects with continuous SSRF revalidation.

    Args:
        url: The target web URL to fetch.
        client: Optional reusable httpx.Client (e.g. for testing with mock transport).
        timeout: Total HTTP timeout in seconds.
        connect_timeout: Specific socket connection timeout in seconds.
        read_timeout: Specific response read timeout in seconds.
        max_retries: Number of retries for transient HTTP errors (502/503/504, read timeouts).
        max_redirects: Maximum allowed redirect hops.
        max_bytes: Maximum response body size in bytes.
        max_text_length: Maximum extracted text length.
        block_ssrf: Whether to enforce SSRF validation.

    Returns:
        FetchedPage containing metadata, status, title, and cleaned text or error details.
    """
    timeout = timeout if timeout is not None else settings.HTTP_TIMEOUT_SECONDS
    connect_timeout = (
        connect_timeout if connect_timeout is not None else settings.HTTP_CONNECT_TIMEOUT_SECONDS
    )
    read_timeout = read_timeout if read_timeout is not None else settings.HTTP_READ_TIMEOUT_SECONDS
    max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
    max_redirects = max_redirects if max_redirects is not None else settings.MAX_REDIRECTS
    max_bytes = max_bytes if max_bytes is not None else settings.MAX_PAGE_BYTES
    max_text_length = max_text_length if max_text_length is not None else settings.MAX_TEXT_LENGTH
    block_ssrf = block_ssrf if block_ssrf is not None else settings.BLOCK_SSRF

    # Initial URL validation
    try:
        current_url = validate_url(url, block_ssrf=block_ssrf)
    except SecurityError as err:
        return FetchedPage(
            original_url=url,
            final_url=url,
            success=False,
            error=str(err),
        )

    headers = {"User-Agent": settings.USER_AGENT}
    should_close_client = False

    if client is None:
        client_timeout = httpx.Timeout(timeout, connect=connect_timeout, read=read_timeout)
        client = httpx.Client(timeout=client_timeout, headers=headers)
        should_close_client = True

    redirect_count = 0

    try:
        while True:
            # Revalidate target URL on each hop (protecting against redirect-based SSRF)
            try:
                current_url = validate_url(current_url, block_ssrf=block_ssrf)
            except SecurityError as sec_err:
                return FetchedPage(
                    original_url=url,
                    final_url=current_url,
                    success=False,
                    error=f"Redirect blocked by security check: {sec_err}",
                )

            # Attempt GET with bounded retries for transient issues
            response = None
            last_request_err = None
            retries_left = max_retries

            while retries_left >= 0:
                try:
                    response = client.get(current_url, follow_redirects=False)
                    # Retry only on transient gateway/server errors
                    if response.status_code in (502, 503, 504) and retries_left > 0:
                        retries_left -= 1
                        time.sleep(0.2)
                        continue
                    break
                except (httpx.ReadTimeout, httpx.ConnectTimeout) as t_err:
                    last_request_err = t_err
                    if retries_left > 0:
                        retries_left -= 1
                        time.sleep(0.2)
                        continue
                    raise t_err

            if response is None:
                raise last_request_err or httpx.RequestError("No response received")

            if response.is_redirect:
                redirect_count += 1
                if redirect_count > max_redirects:
                    return FetchedPage(
                        original_url=url,
                        final_url=current_url,
                        status_code=response.status_code,
                        success=False,
                        error=f"Exceeded maximum redirect limit of {max_redirects} hops.",
                    )

                location = response.headers.get("location")
                if not location:
                    return FetchedPage(
                        original_url=url,
                        final_url=current_url,
                        status_code=response.status_code,
                        success=False,
                        error="Redirect response missing Location header.",
                    )

                # Resolve relative redirects against current URL
                current_url = urllib.parse.urljoin(current_url, location)
                continue

            # Non-redirect response reached
            break

        # Check HTTP response status
        if response.status_code != 200:
            return FetchedPage(
                original_url=url,
                final_url=current_url,
                status_code=response.status_code,
                success=False,
                error=describe_http_status(response.status_code),
            )

        # Check Content-Type header
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(
            allowed in content_type for allowed in ("text/", "html", "xhtml", "xml", "json")
        ):
            return FetchedPage(
                original_url=url,
                final_url=current_url,
                status_code=response.status_code,
                success=False,
                error=f"Unsupported content type: {content_type}",
            )

        # Check response body size
        content_bytes = response.content
        if len(content_bytes) > max_bytes:
            return FetchedPage(
                original_url=url,
                final_url=current_url,
                status_code=response.status_code,
                success=False,
                error=f"Response size ({len(content_bytes)} bytes) exceeds limit of {max_bytes} bytes.",
            )

        # Parse and clean HTML content
        title, text = clean_html(response.text, max_text_length=max_text_length)

        return FetchedPage(
            original_url=url,
            final_url=current_url,
            title=title,
            text=text,
            status_code=response.status_code,
            success=True,
            error=None,
        )

    except httpx.TimeoutException:
        return FetchedPage(
            original_url=url,
            final_url=current_url,
            success=False,
            error=f"Connection timed out after {timeout} seconds.",
        )
    except httpx.RequestError as req_err:
        return FetchedPage(
            original_url=url,
            final_url=current_url,
            success=False,
            error=f"HTTP network error: {type(req_err).__name__}",
        )
    except Exception as exc:
        return FetchedPage(
            original_url=url,
            final_url=current_url,
            success=False,
            error=f"Unexpected retrieval error: {type(exc).__name__}",
        )
    finally:
        if should_close_client:
            client.close()


# ==============================================================================
# Search Provider Abstraction
# ==============================================================================


class SearchProvider(abc.ABC):
    """Abstract base class for search providers."""

    @abc.abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Execute a search query and return a list of SearchResult items.

        Args:
            query: Plain-text search query.
            max_results: Maximum number of search results to return.

        Returns:
            List of SearchResult objects.
        """


class MockSearchProvider(SearchProvider):
    """
    Mock search provider for automated evaluation, offline testing, and unit tests.
    Does not make any live network requests.
    """

    def __init__(self, predefined_results: Optional[dict[str, List[SearchResult]]] = None) -> None:
        self.predefined_results = predefined_results or {}

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query or not query.strip() or max_results <= 0:
            return []

        clean_query = query.strip()
        if clean_query in self.predefined_results:
            return self.predefined_results[clean_query][:max_results]

        # Deterministic mock results when query not in predefined dictionary
        return [
            SearchResult(
                title=f"Mock Result 1 for {clean_query}",
                url=f"https://example.com/mock-1?q={urllib.parse.quote(clean_query)}",
                snippet=f"Verified mock excerpt providing facts for: {clean_query}.",
            ),
            SearchResult(
                title=f"Mock Result 2 for {clean_query}",
                url=f"https://example.com/mock-2?q={urllib.parse.quote(clean_query)}",
                snippet=f"Secondary supporting source for: {clean_query}.",
            ),
        ][:max_results]


class DuckDuckGoSearchProvider(SearchProvider):
    """
    Zero-API-key search provider using DuckDuckGo HTML endpoint with automatic Lite fallback.
    Retrieves web results without paid subscription or token credentials.
    """

    def __init__(self, client: Optional[httpx.Client] = None, timeout: float = 10.0) -> None:
        self.client = client
        self.timeout = timeout

    def _extract_target_url(self, raw_url: str) -> str:
        """Extract the destination URL if DuckDuckGo wraps it in a redirect query."""
        if not raw_url:
            return ""
        parsed = urllib.parse.urlsplit(raw_url)
        if "duckduckgo.com" in parsed.netloc and "/l/" in parsed.path:
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs:
                return urllib.parse.unquote(qs["uddg"][0])
        return raw_url

    def _parse_html_results(self, html_text: str, max_results: int) -> List[SearchResult]:
        """Parse search results from html.duckduckgo.com output."""
        soup = bs4.BeautifulSoup(html_text, "html.parser")
        result_nodes = soup.select(".result__body")
        results: List[SearchResult] = []

        for node in result_nodes:
            if len(results) >= max_results:
                break

            title_elem = node.select_one(".result__title a")
            snippet_elem = node.select_one(".result__snippet")

            if not title_elem or not title_elem.get("href"):
                continue

            title = title_elem.get_text(strip=True)
            raw_href = title_elem["href"]
            target_url = self._extract_target_url(raw_href)
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            if not target_url.startswith(("http://", "https://")):
                continue

            results.append(SearchResult(title=title, url=target_url, snippet=snippet))

        return results

    def _parse_lite_results(self, html_text: str, max_results: int) -> List[SearchResult]:
        """Parse search results from lite.duckduckgo.com output."""
        soup = bs4.BeautifulSoup(html_text, "html.parser")
        links = soup.select(".result-link")
        snippets = soup.select(".result-snippet")
        results: List[SearchResult] = []

        for i, link_elem in enumerate(links):
            if len(results) >= max_results:
                break

            if not link_elem.get("href"):
                continue

            title = link_elem.get_text(strip=True)
            raw_href = link_elem["href"]
            target_url = self._extract_target_url(raw_href)
            snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""

            if not target_url.startswith(("http://", "https://")):
                continue

            results.append(SearchResult(title=title, url=target_url, snippet=snippet))

        return results

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        if not query or not query.strip() or max_results <= 0:
            return []

        clean_query = query.strip()
        headers = {"User-Agent": settings.USER_AGENT}

        should_close = False
        client = self.client
        if client is None:
            client = httpx.Client(timeout=self.timeout, headers=headers)
            should_close = True

        try:
            # 1. Primary endpoint: Try standard HTML search
            resp_html = client.post("https://html.duckduckgo.com/html/", data={"q": clean_query})
            if resp_html.status_code == 200:
                results = self._parse_html_results(resp_html.text, max_results)
                if results:
                    return results

            # 2. Fallback endpoint: Try Lite search
            resp_lite = client.post("https://lite.duckduckgo.com/lite/", data={"q": clean_query})
            if resp_lite.status_code == 200:
                results = self._parse_lite_results(resp_lite.text, max_results)
                return results

            # If both endpoints returned HTTP errors, raise SearchError
            raise SearchError(
                f"DuckDuckGo search endpoints failed (HTML status: {resp_html.status_code}, Lite status: {resp_lite.status_code})."
            )

        except httpx.TimeoutException:
            raise SearchError(f"Search request to DuckDuckGo timed out after {self.timeout}s.")
        except httpx.RequestError as exc:
            raise SearchError(f"Network error communicating with DuckDuckGo: {type(exc).__name__}")
        except Exception as exc:
            if isinstance(exc, SearchError):
                raise
            raise SearchError(f"Unexpected error during DuckDuckGo search: {exc}")
        finally:
            if should_close:
                client.close()


def get_search_provider(name: Optional[str] = None) -> SearchProvider:
    """
    Factory function returning the configured search provider instance.

    Args:
        name: Provider name ('duckduckgo' or 'mock'). Defaults to settings.SEARCH_PROVIDER.

    Returns:
        An instance of SearchProvider.
    """
    provider_name = (name or settings.SEARCH_PROVIDER).lower().strip()

    if provider_name == "duckduckgo":
        return DuckDuckGoSearchProvider()
    elif provider_name == "mock":
        return MockSearchProvider()
    else:
        raise ValueError(
            f"Unsupported search provider: '{provider_name}'. Supported options: 'duckduckgo', 'mock'."
        )


# ==============================================================================
# High-Level Retrieval Workflow
# ==============================================================================


def search_web(
    query: str,
    max_results: Optional[int] = None,
    provider: Optional[SearchProvider] = None,
) -> List[SearchResult]:
    """
    Search the web for a query using the configured search provider.

    Args:
        query: User search query.
        max_results: Maximum results to return.
        provider: Optional explicit SearchProvider instance.

    Returns:
        List of SearchResult objects.
    """
    if not query or not query.strip():
        return []

    max_results = max_results if max_results is not None else settings.MAX_SEARCH_RESULTS
    search_engine = provider or get_search_provider()
    return search_engine.search(query, max_results=max_results)


def retrieve(
    query: str,
    max_search_results: Optional[int] = None,
    max_pages_to_fetch: Optional[int] = None,
    provider: Optional[SearchProvider] = None,
    client: Optional[httpx.Client] = None,
) -> RetrievalResult:
    """
    Execute end-to-end web retrieval: search the query, deduplicate URLs, and fetch candidate pages.

    Args:
        query: The natural-language or keyword search query.
        max_search_results: Number of search items to request.
        max_pages_to_fetch: Number of candidate pages to download and parse.
        provider: Optional explicit SearchProvider instance.
        client: Optional reusable HTTP client.

    Returns:
        RetrievalResult containing search results and fetched page contents with metadata.
    """
    if not query or not query.strip():
        return RetrievalResult(query=query or "", search_results=[], pages=[])

    max_search_results = (
        max_search_results if max_search_results is not None else settings.MAX_SEARCH_RESULTS
    )
    max_pages_to_fetch = (
        max_pages_to_fetch if max_pages_to_fetch is not None else settings.MAX_PAGES_TO_FETCH
    )

    search_results = search_web(query, max_results=max_search_results, provider=provider)

    pages: List[FetchedPage] = []

    # Deduplicate URLs while preserving order to avoid redundant downloads
    seen_urls: Set[str] = set()
    unique_urls: List[str] = []
    for res in search_results:
        clean_url = res.url.strip()
        if clean_url and clean_url not in seen_urls:
            seen_urls.add(clean_url)
            unique_urls.append(clean_url)
            if len(unique_urls) >= max_pages_to_fetch:
                break

    for page_url in unique_urls:
        page = fetch_page(page_url, client=client)
        pages.append(page)

    return RetrievalResult(
        query=query,
        search_results=search_results,
        pages=pages,
    )
