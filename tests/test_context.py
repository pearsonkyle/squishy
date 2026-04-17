from __future__ import annotations
 
import json
 
from squishy.context import build_system_prompt, detect_project, trim_history
 
 
def test_detect_node_nextjs(tmp_path):
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"next": "14", "react": "18"}}))
    info = detect_project(str(tmp_path))
    assert info.language == "javascript"
    assert info.framework == "nextjs"
 
 
def test_detect_python_fastapi(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["fastapi"]\n')
    info = detect_project(str(tmp_path))
    assert info.language == "python"
    assert info.framework == "fastapi"
 
 
def test_detect_rust(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = \"x\"\n")
    info = detect_project(str(tmp_path))
    assert info.language == "rust"
 
 
def test_detect_empty(tmp_path):
    info = detect_project(str(tmp_path))
    assert info.language == "unknown"
 
 
def test_build_system_prompt_includes_project_info(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["flask"]\n')
    info = detect_project(str(tmp_path))
    prompt = build_system_prompt(str(tmp_path), info)
    assert "python" in prompt.lower()
    assert "flask" in prompt.lower()
    assert "read files before editing" in prompt.lower()
 
 
def test_trim_history_preserves_system_and_first_user():
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "first user"})
    for i in range(20):
        msgs.append({"role": "assistant", "content": f"a{i}"})
 
    trimmed = trim_history(msgs, max_messages=10)
    assert trimmed[0]["content"] == "sys"
    assert trimmed[1]["content"] == "first user"
    assert len(trimmed) == 10
    # The tail should include the most recent assistant messages
    assert trimmed[-1]["content"] == "a19"
 
 
def test_trim_history_noop_when_short():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert trim_history(msgs) == msgs
