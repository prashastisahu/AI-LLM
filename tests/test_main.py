def test_build_analysis_prompt():
    from app.vision import build_analysis_prompt
    result = build_analysis_prompt("evening dinner")
    assert "evening dinner" in result
    assert '"needs"' in result
    assert "Do not suggest makeup" in result
