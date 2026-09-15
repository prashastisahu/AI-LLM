def test_build_prompt():
    from app.vision import build_prompt
    result = build_prompt("evening dinner")
    assert "evening dinner" in result
    assert "recommendations" in result
    assert "Do not suggest makeup" in result
