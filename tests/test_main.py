def test_build_prompt():
    from main import build_prompt
    result = build_prompt("evening dinner")
    assert "evening dinner" in result
    assert "MAKEUP:" in result