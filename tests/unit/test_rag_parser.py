"""
Unit tests for RAG Document Parsers (Text, Markdown, JSON, HTML).
"""

import pytest

from rag.parser import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


def test_parse_plain_text(parser):
    content = "OmniForge is an enterprise AI/ML platform. It supports multimodal intelligence."
    doc = parser.parse_text(content, title="OmniForge Overview", metadata={"author": "OmniForge Team"})

    assert doc.title == "OmniForge Overview"
    assert doc.source_type == "text"
    assert "OmniForge is an enterprise" in doc.content
    assert doc.metadata["author"] == "OmniForge Team"
    assert doc.doc_id.startswith("doc_")


def test_parse_markdown_extracts_headers_and_tables(parser):
    md_content = """# Architecture Overview
OmniForge unifies Classical ML, Vision, NLP, and RAG.

## Features Table
| Module | Capability |
| :--- | :--- |
| RAG | Cross-Encoder Reranking |
| Vision | ByteTrack and YOLO |
"""
    doc = parser.parse_markdown(md_content, title="Arch Doc")

    assert doc.title == "Architecture Overview"
    assert doc.source_type == "markdown"
    assert len(doc.metadata["sections"]) >= 1
    assert "Features Table" in [s["heading"] for s in doc.metadata["sections"]]
    assert len(doc.metadata["tables"]) >= 1
    assert "Module" in doc.metadata["tables"][0]["headers"]


def test_parse_json(parser):
    json_str = '{"service": "OmniForge RAG", "version": "1.0", "active": true, "tags": ["vector", "nlp"]}'
    doc = parser.parse_json(json_str, title="Service Config")

    assert doc.source_type == "json"
    assert "service: OmniForge RAG" in doc.content
    assert doc.metadata["keys"] == ["service", "version", "active", "tags"]


def test_parse_html(parser):
    html_str = """
    <html>
        <head><title>OmniForge Docs</title></head>
        <body>
            <h1>Welcome to OmniForge</h1>
            <p>High performance multimodal platform.</p>
        </body>
    </html>
    """
    doc = parser.parse_html(html_str)

    assert doc.title == "OmniForge Docs"
    assert "Welcome to OmniForge" in doc.content
    assert "High performance multimodal platform." in doc.content
