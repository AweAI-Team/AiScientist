"""
Linter tool — synchronous Ruff-based lint pass.

Runs ``ruff check`` on Python files/directories and optionally auto-fixes safe
issues. All commands execute through ShellInterface (synchronous subprocess).
"""

from __future__ import annotations

import json
import shlex

from tools.base import Tool


class LinterTool(Tool):
    """
    Run Ruff linter on Python code with optional auto-fix.

    Runs ruff via ShellInterface.
    """

    def __init__(self, timeout: int = 60):
        self._timeout = timeout

    def name(self) -> str:
        return "linter"

    def execute(
        self,
        shell,
        path: str = "/home/code",
        auto_fix: bool = False,
        **kwargs,
    ) -> str:
        # Validate path exists
        check = shell.send_command(
            f'test -e {shlex.quote(path)} && echo "ok" || echo "missing"',
            timeout=5,
        )
        if "missing" in check.output:
            return f"Error: path does not exist: {path}"

        # Check it's Python
        safe_path = shlex.quote(path)
        is_py = shell.send_command(
            f'test -f {safe_path} && echo "file" || '
            f'(find {safe_path} -name "*.py" -type f 2>/dev/null | head -1)',
            timeout=10,
        )
        output = is_py.output.strip()
        if not output and not path.endswith(".py"):
            return f"Warning: {path} does not appear to contain Python files."

        # Ensure ruff is available
        ruff_check = shell.send_command("which ruff 2>/dev/null || echo 'missing'", timeout=5)
        if "missing" in ruff_check.output:
            shell.send_command("pip install ruff -q 2>&1", timeout=120)
            verify = shell.send_command("which ruff 2>/dev/null || echo 'missing'", timeout=5)
            if "missing" in verify.output:
                return "Error: failed to install ruff. Try `pip install ruff` manually."

        # Run ruff check (JSON output)
        result_before = shell.send_command(
            f"ruff check {safe_path} --output-format json 2>&1",
            timeout=self._timeout,
        )
        issues_before = self._parse(result_before.output)

        # Auto-fix if requested and there are issues
        issues_after = None
        if auto_fix and issues_before["total_issues"] > 0:
            shell.send_command(
                f"ruff check {safe_path} --fix --output-format json 2>&1",
                timeout=self._timeout,
            )
            result_after = shell.send_command(
                f"ruff check {safe_path} --output-format json 2>&1",
                timeout=self._timeout,
            )
            issues_after = self._parse(result_after.output)

        return self._format(issues_before, issues_after, auto_fix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> dict:
        issues: dict = {"total_issues": 0, "errors": 0, "warnings": 0, "details": []}
        try:
            data = json.loads(raw.strip()) if raw.strip() else []
            if not isinstance(data, list):
                return issues
            for item in data:
                code = item.get("code", "")
                severity = "error" if code.startswith(("E", "F")) else "warning"
                issues["details"].append({
                    "file": item.get("filename", ""),
                    "line": item.get("location", {}).get("row", 0),
                    "column": item.get("location", {}).get("column", 0),
                    "severity": severity,
                    "code": code,
                    "message": item.get("message", ""),
                })
                issues["total_issues"] += 1
                if severity == "error":
                    issues["errors"] += 1
                else:
                    issues["warnings"] += 1
        except (json.JSONDecodeError, Exception):
            if raw.strip():
                issues["details"].append({"message": "Parse error", "raw": raw[:500]})
        return issues

    @staticmethod
    def _format(before: dict, after: dict | None, auto_fix: bool) -> str:
        lines = ["=== Linter Report ===\n"]
        lines.append(f"Total issues found: {before['total_issues']}")
        lines.append(f"  - Errors: {before['errors']}")
        lines.append(f"  - Warnings: {before['warnings']}")

        if auto_fix and after is not None:
            fixed = before["total_issues"] - after["total_issues"]
            lines.append(f"\nIssues fixed: {fixed}")
            lines.append(f"Remaining issues: {after['total_issues']}")

        show = after if (auto_fix and after is not None) else before
        if show["details"]:
            lines.append("\n=== Issue Details ===")
            for i, d in enumerate(show["details"][:30], 1):
                if "file" in d:
                    lines.append(
                        f"\n{i}. [{d['severity'].upper()}] {d['file']}:{d['line']}:{d['column']}"
                    )
                    lines.append(f"   {d['code']}: {d['message']}")
                elif "message" in d:
                    lines.append(f"\n{i}. {d['message']}")
            if len(show["details"]) > 30:
                lines.append(f"\n... and {len(show['details']) - 30} more issues")
        else:
            lines.append("\nNo issues found. Code looks good!")

        return "\n".join(lines)

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "linter",
                "description": (
                    "Run Ruff linter on Python files or directories. "
                    "Reports code quality issues (style, potential bugs, imports) "
                    "and optionally auto-fixes safe issues. "
                    "Use before committing code to catch common mistakes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to lint (file or directory, default: /home/code)",
                        },
                        "auto_fix": {
                            "type": "boolean",
                            "description": "Auto-fix safe issues (default: false). Review first, then fix.",
                        },
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
