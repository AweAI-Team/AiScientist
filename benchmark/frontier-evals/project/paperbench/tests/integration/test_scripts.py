import subprocess

import pytest

scripts = [
    "paperbench/scripts/run_judge_eval.py",
    "paperbench/scripts/run_judge.py",
    "paperbench/scripts/run_monitor.py",
]


@pytest.mark.parametrize("script", scripts)
def test_script_help(script: str) -> None:
    """Test that each script runs with --help without errors."""
    try:
        # Run the script with --help and check it doesn't throw an error
        result = subprocess.run(
            ["uv", "run", "python", script, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        pytest.fail(f"Exception occurred while running {script}: {e}")

    # chz-based entrypoints exit with status 1 after printing help, so we
    # detect success via the output rather than the return code.
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    combined_output = "\n".join(part for part in (stdout, stderr) if part)
    combined_lower = combined_output.lower()

    assert combined_output, f"Script {script} produced no output."

    if "traceback" in combined_lower and "entrypointhelpexception" not in combined_lower:
        pytest.fail(f"Script {script} produced an unexpected traceback:\n{combined_output}")

    help_indicators = (
        "usage",
        "options:",
        "entry point",
        "entrypointhelpexception",
    )

    assert any(indicator in combined_lower for indicator in help_indicators), (
        f"Script {script} did not appear to print help text:\n{combined_output}"
    )
