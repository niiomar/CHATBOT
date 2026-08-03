import ingest


def test_load_documents_empty_dir(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ingest, "DOCS_PATH", str(tmp_path))
    with caplog.at_level("WARNING"):
        docs = ingest.load_documents()
    assert docs == []
    assert "no documents found" in caplog.text


def test_load_documents_loads_txt_and_tags_source(tmp_path, monkeypatch):
    (tmp_path / "policy.txt").write_text("Section 1: hello world", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("should be skipped", encoding="utf-8")
    monkeypatch.setattr(ingest, "DOCS_PATH", str(tmp_path))

    docs = ingest.load_documents()

    assert len(docs) == 1
    assert docs[0].metadata["source_file"] == "policy.txt"
    assert "hello world" in docs[0].page_content


def test_load_documents_skips_unreadable_file(tmp_path, monkeypatch, caplog):
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a real pdf")
    monkeypatch.setattr(ingest, "DOCS_PATH", str(tmp_path))

    with caplog.at_level("ERROR"):
        docs = ingest.load_documents()

    assert docs == []
    assert "corrupt.pdf" in caplog.text
