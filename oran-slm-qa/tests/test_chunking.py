from oran_qa.data.chunking import HEADING_RE


def test_heading_regex():
    assert HEADING_RE.match("1 Introduction")
    assert HEADING_RE.match("7.3.2 Near-RT RIC functions")
    assert not HEADING_RE.match("Introduction without numbering")
