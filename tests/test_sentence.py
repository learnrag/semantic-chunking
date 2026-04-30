"""Tests for sentence segmentation."""

from chunker.sentence import RegexSentenceSegmenter


def test_sentence_offsets_round_trip():
    text = "Hello world. How are you? Fine!"
    spans = RegexSentenceSegmenter().split(text)
    assert len(spans) == 3
    for span in spans:
        assert text[span.start_char : span.end_char] == span.text


def test_empty_and_whitespace():
    assert RegexSentenceSegmenter().split("") == []
    spans = RegexSentenceSegmenter().split("   ")
    assert len(spans) == 1


def test_single_sentence_no_terminator():
    text = "No period here"
    spans = RegexSentenceSegmenter().split(text)
    assert len(spans) == 1
    assert spans[0].text == text
