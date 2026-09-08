"""
Multi-format Document Parser for Enterprise RAG.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional
from rag.base import BaseDocumentParser, Document


class DocumentParser(BaseDocumentParser):
    """
    Production-grade document parser supporting Markdown, plain text, structured JSON, and HTML.
    Extracts metadata, cleans whitespace, and generates deterministic document identifiers.
    """

    def parse(
        self,
        raw_content: str,
        title: str,
        source_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """
        Standard parser interface delegating to format-specific parsing routines.
        """
        st = source_type.lower()
        if st == "markdown" or st == "md":
            return self.parse_markdown(raw_content, title=title, metadata=metadata, doc_id=doc_id)
        elif st == "json":
            return self.parse_json(raw_content, title=title, metadata=metadata, doc_id=doc_id)
        elif st == "html" or st == "htm":
            return self.parse_html(raw_content, title=title, metadata=metadata, doc_id=doc_id)
        else:
            return self.parse_text(raw_content, title=title, metadata=metadata, doc_id=doc_id)

    def parse_text(
        self,
        text: str,
        title: str = "Plain Text Document",
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parse unformatted plain text document."""
        cleaned_content = text.strip()
        doc_metadata = metadata.copy() if metadata else {}

        if not doc_id:
            content_hash = hashlib.sha256(f"{title}_{cleaned_content}".encode("utf-8")).hexdigest()[:16]
            doc_id = f"doc_{content_hash}"

        doc_metadata.update({
            "word_count": len(cleaned_content.split()),
            "char_count": len(cleaned_content),
            "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_type": "text",
        })

        return Document(
            doc_id=doc_id,
            title=title,
            content=cleaned_content,
            source_type="text",
            metadata=doc_metadata,
        )

    def parse_markdown(
        self,
        markdown_text: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parse Markdown document, extracting headings and structured tables into metadata."""
        cleaned_content = markdown_text.strip()
        doc_metadata = metadata.copy() if metadata else {}

        # Extract top-level heading as title
        extracted_title = None
        sections: List[Dict[str, Any]] = []
        for line in cleaned_content.splitlines():
            line_str = line.strip()
            if line_str.startswith("# ") and not extracted_title:
                extracted_title = line_str.lstrip("# ").strip()
            elif line_str.startswith("#"):
                level = len(line_str) - len(line_str.lstrip("#"))
                heading_text = line_str.lstrip("# ").strip()
                sections.append({"level": level, "heading": heading_text})

        if not extracted_title:
            extracted_title = title if title else "Markdown Document"

        # Extract Markdown tables
        tables: List[Dict[str, Any]] = []
        lines = cleaned_content.splitlines()
        for idx, line in enumerate(lines):
            if "|" in line and idx + 1 < len(lines) and ("|---" in lines[idx + 1] or "|:--" in lines[idx + 1] or "| :--" in lines[idx + 1]):
                headers = [col.strip() for col in line.split("|") if col.strip()]
                tables.append({"line_index": idx, "headers": headers})

        doc_metadata.update({
            "sections": sections,
            "tables": tables,
            "word_count": len(cleaned_content.split()),
            "char_count": len(cleaned_content),
            "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_type": "markdown",
        })

        if not doc_id:
            content_hash = hashlib.sha256(f"{extracted_title}_{cleaned_content}".encode("utf-8")).hexdigest()[:16]
            doc_id = f"doc_{content_hash}"

        return Document(
            doc_id=doc_id,
            title=extracted_title,
            content=cleaned_content,
            source_type="markdown",
            metadata=doc_metadata,
        )

    def parse_json(
        self,
        json_content: str,
        title: str = "JSON Document",
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parse JSON document, serializing key-value pairs into readable sentences."""
        cleaned_content = json_content.strip()
        doc_metadata = metadata.copy() if metadata else {}
        extracted_keys: List[str] = []

        try:
            parsed = json.loads(cleaned_content)
            if isinstance(parsed, dict):
                extracted_keys = list(parsed.keys())
                lines = [f"{k}: {v}" for k, v in parsed.items()]
                cleaned_content = "\n".join(lines)
            elif isinstance(parsed, list):
                cleaned_content = "\n".join([str(item) for item in parsed])
        except Exception:
            pass

        doc_metadata.update({
            "keys": extracted_keys,
            "word_count": len(cleaned_content.split()),
            "char_count": len(cleaned_content),
            "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_type": "json",
        })

        if not doc_id:
            content_hash = hashlib.sha256(f"{title}_{cleaned_content}".encode("utf-8")).hexdigest()[:16]
            doc_id = f"doc_{content_hash}"

        return Document(
            doc_id=doc_id,
            title=title,
            content=cleaned_content,
            source_type="json",
            metadata=doc_metadata,
        )

    def parse_html(
        self,
        html_content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parse HTML string by stripping tags and extracting page title."""
        doc_metadata = metadata.copy() if metadata else {}

        # Extract title
        extracted_title = title
        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if title_match and not extracted_title:
            extracted_title = title_match.group(1).strip()
        if not extracted_title:
            extracted_title = "HTML Document"

        # Strip tags and normalize whitespace
        text_content = re.sub(r"<[^>]+>", " ", html_content)
        text_content = re.sub(r"\s+", " ", text_content).strip()

        doc_metadata.update({
            "word_count": len(text_content.split()),
            "char_count": len(text_content),
            "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_type": "html",
        })

        if not doc_id:
            content_hash = hashlib.sha256(f"{extracted_title}_{text_content}".encode("utf-8")).hexdigest()[:16]
            doc_id = f"doc_{content_hash}"

        return Document(
            doc_id=doc_id,
            title=extracted_title,
            content=text_content,
            source_type="html",
            metadata=doc_metadata,
        )
