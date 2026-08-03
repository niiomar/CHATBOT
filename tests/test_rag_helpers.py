from dataclasses import dataclass, field

from rag import format_context, format_history, source_label


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_format_history_empty():
    assert format_history([]) == "(none)"
    assert format_history(None) == "(none)"


def test_format_history_renders_roles():
    history = [
        {"role": "user", "content": "What is Act 1040?"},
        {"role": "assistant", "content": "It governs..."},
    ]
    rendered = format_history(history)
    assert "User: What is Act 1040?" in rendered
    assert "Assistant: It governs..." in rendered


def test_source_label_with_page():
    doc = FakeDoc("text", {"source_file": "policy.pdf", "page": 2})
    assert source_label(doc) == "policy.pdf, p.3"


def test_source_label_without_page():
    doc = FakeDoc("text", {"source_file": "notes.txt"})
    assert source_label(doc) == "notes.txt"


def test_source_label_unknown_source():
    doc = FakeDoc("text", {})
    assert source_label(doc) == "unknown"


def test_format_context_numbers_chunks_and_dedupes_sources():
    docs = [
        FakeDoc("first chunk", {"source_file": "a.pdf", "page": 0}),
        FakeDoc("second chunk", {"source_file": "a.pdf", "page": 0}),
        FakeDoc("third chunk", {"source_file": "b.txt"}),
    ]
    context, sources = format_context(docs)

    assert "[1] (Source: a.pdf, p.1)\nfirst chunk" in context
    assert "[2] (Source: a.pdf, p.1)\nsecond chunk" in context
    assert "[3] (Source: b.txt)\nthird chunk" in context
    assert sources == ["a.pdf, p.1", "b.txt"]


def test_format_context_empty():
    context, sources = format_context([])
    assert context == ""
    assert sources == []
