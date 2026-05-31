#!/usr/bin/env python3
"""Small stdio MCP server for reading arXiv papers.

The server is dependency-light by design. It prefers arXiv source bundles,
because this environment may not have PDF text extraction tools installed.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "codex-arxiv-mcp/1.0 (mailto:codex@example.invalid)"
DEFAULT_MAX_CHARS = 60_000


@dataclass
class Paper:
    arxiv_id: str
    title: str = ""
    authors: list[str] | None = None
    summary: str = ""
    published: str = ""
    updated: str = ""
    pdf_url: str = ""


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("arxiv.org"):
        path = parsed.path.strip("/")
        if path.startswith(("abs/", "pdf/", "e-print/")):
            value = path.split("/", 1)[1]
    value = value.removesuffix(".pdf")
    match = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?|([a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", value)
    if not match:
        raise ValueError(f"Could not find an arXiv id in: {value}")
    return match.group(0)


def fetch_url(url: str, *, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def get_metadata(arxiv_id: str) -> Paper:
    query = urllib.parse.urlencode({"id_list": arxiv_id})
    data = fetch_url(f"{ARXIV_API}?{query}")
    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return Paper(arxiv_id=arxiv_id)

    def text(name: str) -> str:
        node = entry.find(f"atom:{name}", ns)
        return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""

    authors = [
        re.sub(r"\s+", " ", node.text or "").strip()
        for node in entry.findall("atom:author/atom:name", ns)
    ]
    pdf_url = ""
    for link in entry.findall("atom:link", ns):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "")
            break
    return Paper(
        arxiv_id=arxiv_id,
        title=text("title"),
        authors=authors,
        summary=text("summary"),
        published=text("published"),
        updated=text("updated"),
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
    )


def strip_tex(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\(begin|end)\{[^}]+\}", "\n", tex)
    tex = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n\1\n", tex)
    tex = re.sub(r"\\(?:title|author|date|caption)\{([^}]*)\}", r"\n\1\n", tex)
    tex = re.sub(r"\\cite(?:\[[^\]]*\])?\{[^}]*\}", "[citation]", tex)
    tex = re.sub(r"\\ref\{[^}]*\}", "[ref]", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda m: m.group(1) or " ", tex)
    tex = re.sub(r"\\.", " ", tex)
    tex = tex.replace("{", "").replace("}", "")
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    tex = re.sub(r"[ \t]{2,}", " ", tex)
    return tex.strip()


def extract_source_text(arxiv_id: str) -> str:
    data = fetch_url(f"https://arxiv.org/e-print/{arxiv_id}")
    files: list[tuple[str, str]] = []

    def add_file(name: str, content: bytes) -> None:
        if name.lower().endswith((".tex", ".bbl", ".txt")):
            files.append((name, content.decode("utf-8", errors="replace")))

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    add_file(member.name, extracted.read())
    except tarfile.TarError:
        try:
            content = gzip.decompress(data)
        except OSError:
            content = data
        add_file(f"{arxiv_id}.tex", content)

    if not files:
        raise RuntimeError("arXiv source bundle did not contain TeX or text files")

    files.sort(
        key=lambda item: (
            0 if item[0].lower().endswith(".tex") else 1,
            0 if re.search(r"(main|paper|article)\.tex$", item[0], re.I) else 1,
            item[0],
        )
    )
    chunks = [f"## {name}\n\n{strip_tex(content)}" for name, content in files]
    return "\n\n".join(chunks)


def extract_pdf_text(arxiv_id: str) -> str:
    pdf = fetch_url(f"https://arxiv.org/pdf/{arxiv_id}")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, f"{arxiv_id}.pdf")
        txt_path = os.path.join(tmp, f"{arxiv_id}.txt")
        with open(pdf_path, "wb") as f:
            f.write(pdf)
        if shutil.which("pdftotext"):
            subprocess.run(["pdftotext", "-layout", pdf_path, txt_path], check=True, timeout=60)
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
    raise RuntimeError("No PDF text extractor found; install poppler-utils or use arXiv source")


def read_paper(value: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    arxiv_id = normalize_arxiv_id(value)
    paper = get_metadata(arxiv_id)
    header = [
        f"# {paper.title or arxiv_id}",
        f"arXiv: {arxiv_id}",
        f"Authors: {', '.join(paper.authors or [])}",
        f"Published: {paper.published}",
        f"Updated: {paper.updated}",
        f"PDF: {paper.pdf_url}",
        "",
        "## Abstract",
        paper.summary,
        "",
        "## Body",
    ]
    try:
        body = extract_source_text(arxiv_id)
        source_note = "Extracted from arXiv source files."
    except Exception as source_error:
        try:
            body = extract_pdf_text(arxiv_id)
            source_note = "Extracted from PDF."
        except Exception as pdf_error:
            body = f"Could not extract body text.\nSource error: {source_error}\nPDF error: {pdf_error}"
            source_note = "Extraction failed."
    text = "\n".join(header + [source_note, "", body])
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + f"\n\n[Truncated to {max_chars} characters.]"
    return text


def metadata_text(value: str) -> str:
    paper = get_metadata(normalize_arxiv_id(value))
    return "\n".join(
        [
            f"# {paper.title or paper.arxiv_id}",
            f"arXiv: {paper.arxiv_id}",
            f"Authors: {', '.join(paper.authors or [])}",
            f"Published: {paper.published}",
            f"Updated: {paper.updated}",
            f"PDF: {paper.pdf_url}",
            "",
            paper.summary,
        ]
    ).strip()


TOOLS = [
    {
        "name": "read_arxiv_paper",
        "description": "Fetch an arXiv paper by URL or id and return readable text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url_or_id": {"type": "string", "description": "arXiv URL or id, e.g. 1603.02754"},
                "max_chars": {"type": "integer", "description": "Maximum response characters", "default": DEFAULT_MAX_CHARS},
            },
            "required": ["url_or_id"],
        },
    },
    {
        "name": "get_arxiv_metadata",
        "description": "Fetch title, authors, dates, PDF URL, and abstract for an arXiv paper.",
        "inputSchema": {
            "type": "object",
            "properties": {"url_or_id": {"type": "string", "description": "arXiv URL or id"}},
            "required": ["url_or_id"],
        },
    },
]


def handle(method: str, params: dict[str, Any] | None) -> Any:
    params = params or {}
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion", "2024-11-05"),
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "arxiv-reader", "version": "1.0.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "read_arxiv_paper":
            text = read_paper(args["url_or_id"], int(args.get("max_chars", DEFAULT_MAX_CHARS)))
        elif name == "get_arxiv_metadata":
            text = metadata_text(args["url_or_id"])
        else:
            raise ValueError(f"Unknown tool: {name}")
        return {"content": [{"type": "text", "text": text}]}
    if method == "resources/templates/list":
        return {
            "resourceTemplates": [
                {
                    "uriTemplate": "arxiv://{id}",
                    "name": "arXiv paper",
                    "description": "Read an arXiv paper by id, for example arxiv://1603.02754",
                    "mimeType": "text/plain",
                }
            ]
        }
    if method == "resources/read":
        uri = params.get("uri", "")
        if not uri.startswith("arxiv://"):
            raise ValueError(f"Unsupported resource URI: {uri}")
        arxiv_id = urllib.parse.unquote(uri.removeprefix("arxiv://"))
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": read_paper(arxiv_id)}]}
    if method in {"notifications/initialized", "ping"}:
        return {} if method == "ping" else None
    raise ValueError(f"Unsupported method: {method}")


def read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("ascii", errors="replace").partition(":")
        headers[key.lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def write_message(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
    sys.stdout.buffer.flush()


def run_stdio() -> None:
    while True:
        message = read_message()
        if message is None:
            return
        if "id" not in message:
            try:
                handle(message.get("method", ""), message.get("params"))
            except Exception:
                pass
            continue
        try:
            result = handle(message.get("method", ""), message.get("params"))
            write_message({"jsonrpc": "2.0", "id": message["id"], "result": result})
        except Exception as exc:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32000, "message": str(exc)},
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP server for reading arXiv papers")
    parser.add_argument("--stdio", action="store_true", help="Run as an MCP stdio server")
    parser.add_argument("--read", help="Read a paper directly for smoke tests")
    args = parser.parse_args()
    if args.read:
        print(read_paper(args.read))
        return
    run_stdio()


if __name__ == "__main__":
    main()
