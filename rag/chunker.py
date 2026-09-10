"""
Recursive Character and Semantic Chunker for Enterprise RAG.
"""

from __future__ import annotations

from typing import List, Optional

from rag.base import BaseChunker, Document, DocumentChunk


class RecursiveSemanticChunker(BaseChunker):
    """
    Structure-aware recursive text chunker.
    Splits documents along natural semantic boundaries (paragraphs, lines, sentences)
    while enforcing strict size limits and sliding token overlap.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Overlap in characters between adjacent chunks.
            separators: Hierarchy of split delimiters.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by delimiter hierarchy."""
        final_chunks: List[str] = []
        if not separators:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            splits = list(text)
        elif separator == ". ":
            splits = [s + "." if not s.endswith(".") else s for s in text.split(". ") if s.strip()]
        else:
            splits = text.split(separator)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge short split components into chunks respecting chunk_size and chunk_overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        sep = separator if separator != ". " else " "

        for s in splits:
            s_len = len(s)
            if total_len + s_len + (len(sep) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = sep.join(current_doc).strip()
                    if doc:
                        docs.append(doc)
                    # Create overlap by keeping trailing splits
                    while current_doc and total_len > self.chunk_overlap:
                        removed = current_doc.pop(0)
                        total_len -= len(removed) + len(sep)
                current_doc.append(s)
                total_len += s_len
            else:
                current_doc.append(s)
                total_len += s_len + (len(sep) if len(current_doc) > 1 else 0)

        if current_doc:
            doc = sep.join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """
        Decompose a Document into structured DocumentChunk instances with exact character offsets.
        """
        raw_text = document.content
        if not raw_text.strip():
            return []

        text_chunks = self._split_text(raw_text, self.separators)
        chunks: List[DocumentChunk] = []

        current_search_pos = 0
        for idx, text in enumerate(text_chunks):
            text_clean = text.strip()
            if not text_clean:
                continue

            # Locate start position in raw_text
            start_pos = raw_text.find(text_clean, current_search_pos)
            if start_pos == -1:
                start_pos = raw_text.find(text_clean)
            if start_pos == -1:
                start_pos = current_search_pos

            end_pos = start_pos + len(text_clean)
            # Ensure text matches exact raw slice
            exact_text = raw_text[start_pos:end_pos]
            current_search_pos = max(0, start_pos + 1)

            chunk_id = f"{document.doc_id}_chunk_{idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=exact_text,
                    chunk_index=idx,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata=document.metadata.copy(),
                )
            )

        return chunks
