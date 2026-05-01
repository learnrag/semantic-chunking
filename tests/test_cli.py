"""CLI smoke tests with MockEmbedder."""

from pathlib import Path

from chunker.cli import main

FIXTURES = Path(__file__).parent / "fixtures"
GOLD = FIXTURES / "gold.json"


def test_cli_chunk_mock(tmp_path: Path):
    doc = tmp_path / "doc.txt"
    doc.write_text("First sentence here. Second sentence there. Third one ends.", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert main(["chunk", str(doc), "--mock", "-o", str(out), "--percentile", "90"]) == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip()


def test_cli_inspect_mock(tmp_path: Path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text("Alpha. Beta. Gamma. Delta.", encoding="utf-8")
    assert main(["inspect", str(doc), "--mock", "--show-similarity", "--threshold", "0.1"]) == 0
    captured = capsys.readouterr().out
    assert "similarity" in captured
    assert "Sentence" in captured


def test_cli_benchmark_mock(capsys):
    assert main(["benchmark", str(GOLD), "--mock", "--percentile", "90"]) == 0
    out = capsys.readouterr().out
    assert "macro:" in out
    assert "F1=" in out


def test_cli_compare_mock(tmp_path: Path, capsys):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Topic A sentence one. Topic A sentence two.\n\n"
        "Topic B sentence one. Topic B sentence two.",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "compare",
                str(doc),
                "--mock",
                "--threshold",
                "0.2",
                "--percentile",
                "90",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "semantic_absolute" in out
    assert "fixed_token" in out
