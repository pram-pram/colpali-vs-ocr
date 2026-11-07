from cvo.chunking import chunk_text


def test_empty_text(): assert chunk_text("   ") == []
def test_short_text_single_chunk(): assert chunk_text("a b c", size=4, overlap=1) == ["a b c"]
def test_overlap_windows():
    toks = " ".join(str(i) for i in range(10))
    assert chunk_text(toks, size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
def test_no_redundant_tail():
    assert chunk_text("0 1 2 3 4", size=4, overlap=1) == ["0 1 2 3", "3 4"]
