"""
Linter tool for Python code quality checks with auto-fix capabilities.

This tool runs Ruff linter on Python files/directories and automatically fixes
issues when possible. Only supports Python code, if not, return warning message.

All commands are executed through ComputerInterface to support sandbox environments.

Usage:
    from paperbench.solvers.cus_tools.aweai_mcp.linter import LinterTool

    # Add to your solver config:
    solver.basicagent_tools = [LinterTool()]

    # Execute:
    result = await tool.execute(
        computer,
        path="/workspace/src/module.py",
        auto_fix=False
    )
"""

import asyncio
import json
import shlex
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.utils import send_shell_command_with_timeout


# Sensitive paths that should not be accessed
SENSITIVE_PATTERNS = [
    "/etc/passwd",
    "/etc/shadow",
    "~/.ssh",
    "/root",
]


class LinterTool(Tool):
    """
    Tool for running Ruff linter on Python code and automatically fixing issues.

    All file operations and commands are executed through ComputerInterface
    to support sandbox/remote environments.

    Features:
    - Python code quality checking using Ruff
    - Automatic issue fixing when possible (safe_fix only)
    - Detailed issue reporting with file, line, and column information
    - Path validation and safety checks
    """

    timeout: int = 60

    def name(self) -> str:
        return "linter"

    @staticmethod
    def _get_output(result: Any) -> str:
        """Extract string output from either an ExecutionResult or a plain string."""
        if isinstance(result, str):
            return result
        # ExecutionResult has .output (bytes) and .exit_code
        if hasattr(result, "output"):
            if isinstance(result.output, bytes):
                return result.output.decode("utf-8", errors="ignore")
            return str(result.output)
        return str(result)

    def _validate_path(self, path: str) -> tuple[bool, str | None]:
        """
        Validate path for security issues (local check only).

        Args:
            path: Input path

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Security check: prevent accessing sensitive paths
        for pattern in SENSITIVE_PATTERNS:
            if pattern in path:
                return False, f"Access to sensitive path '{pattern}' is not allowed"
        return True, None

    async def _check_path_exists(
        self, computer: ComputerInterface, path: str
    ) -> tuple[bool, bool, str]:
        """
        Check if path exists and whether it's a directory (via ComputerInterface).

        Args:
            computer: ComputerInterface for command execution
            path: Path to check

        Returns:
            Tuple of (exists, is_directory, error_message)
        """
        safe_path = shlex.quote(path)
        result = await computer.send_shell_command(
            f'test -e {safe_path} && (test -d {safe_path} && echo "dir" || echo "file") || echo "not_found"',
        )
        output = self._get_output(result)

        if "not_found" in output:
            return False, False, f"Path does not exist: {path}"

        is_dir = "dir" in output
        return True, is_dir, ""

    async def _is_python_path(
        self, computer: ComputerInterface, path: str, is_directory: bool
    ) -> bool:
        """
        Check if the path is a Python file or directory containing Python files.

        Args:
            computer: ComputerInterface for command execution
            path: File or directory path
            is_directory: Whether path is a directory

        Returns:
            True if it's a Python path, False otherwise
        """
        safe_path = shlex.quote(path)

        if not is_directory:
            # Check file extension
            return path.endswith(".py")

        # For directories, check if it contains Python files
        result = await computer.send_shell_command(
            f'find {safe_path} -name "*.py" -type f 2>/dev/null | head -1',
        )
        return bool(self._get_output(result).strip())

    def _bin_prefix(self, venv_path: str | None) -> str:
        """Return the bin directory prefix for the given venv, or empty string for system."""
        if venv_path:
            # Normalise: strip trailing slash so join is clean
            return venv_path.rstrip("/") + "/bin/"
        return ""

    async def _ensure_ruff_installed(
        self, computer: ComputerInterface, venv_path: str | None = None,
    ) -> tuple[bool, str]:
        """
        Check if Ruff is installed, install if not.

        When *venv_path* is given the tool looks for (and installs into) that
        venv.  Otherwise it falls back to the system Python with
        ``--break-system-packages`` for PEP 668 compatibility.
        """
        prefix = self._bin_prefix(venv_path)

        # Check if ruff is already available
        result = await computer.send_shell_command(
            f"which {prefix}ruff 2>/dev/null || command -v {prefix}ruff 2>/dev/null",
        )
        if self._get_output(result).strip():
            return True, ""

        # Build pip install command
        if venv_path:
            pip_cmd = f"{prefix}pip install ruff -q"
        else:
            pip_cmd = "pip install --break-system-packages ruff -q"

        # Try to install ruff (may take a while, use timeout)
        install_result = await send_shell_command_with_timeout(
            computer, pip_cmd, timeout=120,
        )

        # Verify installation
        verify_result = await computer.send_shell_command(
            f"which {prefix}ruff 2>/dev/null || command -v {prefix}ruff 2>/dev/null",
        )
        if self._get_output(verify_result).strip():
            return True, ""

        return False, f"Failed to install Ruff: {self._get_output(install_result)}"

    def _get_severity(self, code: str) -> str:
        """
        Determine severity based on Ruff error code.

        Args:
            code: Ruff error code (e.g., 'E501', 'F821', 'W293')

        Returns:
            'error' or 'warning'
        """
        # E (pycodestyle errors), F (pyflakes) are errors
        # W (warnings), I (isort), etc. are warnings
        if code.startswith(("E", "F")):
            return "error"
        return "warning"

    def _parse_ruff_output(self, output: str, stderr: str = "") -> dict[str, Any]:
        """
        Parse Ruff linter output into structured format.

        Args:
            output: Ruff stdout (JSON format)
            stderr: Ruff stderr

        Returns:
            Parsed issues dict
        """
        issues: dict[str, Any] = {
            "total_issues": 0,
            "errors": 0,
            "warnings": 0,
            "fixed": 0,
            "details": []
        }

        try:
            # Parse Ruff JSON output
            if output.strip():
                data = json.loads(output)

                # Ruff returns a list of issues
                if isinstance(data, list):
                    for item in data:
                        code = item.get("code", "")
                        severity = self._get_severity(code)
                        issue = {
                            "file": item.get("filename", ""),
                            "line": item.get("location", {}).get("row", 0),
                            "column": item.get("location", {}).get("column", 0),
                            "severity": severity,
                            "code": code,
                            "message": item.get("message", ""),
                        }
                        issues["details"].append(issue)
                        issues["total_issues"] += 1
                        if severity == "error":
                            issues["errors"] += 1
                        else:
                            issues["warnings"] += 1

        except json.JSONDecodeError:
            # Fallback: parse plain text output or show error
            if output.strip() or stderr.strip():
                issues["details"].append({
                    "message": "Failed to parse Ruff output as JSON",
                    "raw_output": output[:500],
                    "raw_stderr": stderr[:500]
                })

        return issues

    async def _run_ruff(
        self,
        computer: ComputerInterface,
        path: str,
        fix: bool = False,
        venv_path: str | None = None,
    ) -> tuple[str, str]:
        """
        Run Ruff command via ComputerInterface.

        Args:
            computer: ComputerInterface for command execution
            path: Path to lint
            fix: Whether to apply fixes
            venv_path: Optional venv path to use for ruff binary

        Returns:
            Tuple of (stdout, stderr)
        """
        prefix = self._bin_prefix(venv_path)
        safe_path = shlex.quote(path)
        fix_flag = "--fix" if fix else ""

        result = await send_shell_command_with_timeout(
            computer,
            f"{prefix}ruff check {safe_path} {fix_flag} --output-format json",
            timeout=self.timeout,
        )

        return self._get_output(result), ""

    def _format_report(
        self,
        issues_before: dict[str, Any],
        issues_after: dict[str, Any] | None,
        auto_fix: bool
    ) -> str:
        """
        Format linter report for output.

        Args:
            issues_before: Issues found before fixing
            issues_after: Issues found after fixing (None if no fix)
            auto_fix: Whether auto-fix was enabled

        Returns:
            Formatted report string
        """
        report_lines = ["=== Linter Report ===\n"]

        # Summary
        report_lines.append(f"Total issues found: {issues_before['total_issues']}")
        report_lines.append(f"  - Errors: {issues_before['errors']}")
        report_lines.append(f"  - Warnings: {issues_before['warnings']}")

        if auto_fix and issues_after is not None:
            fixed_count = issues_before['total_issues'] - issues_after['total_issues']
            report_lines.append(f"\nIssues fixed: {fixed_count}")
            report_lines.append(f"Remaining issues: {issues_after['total_issues']}")

        # Detail issues - show remaining issues if auto_fix applied, otherwise show all
        issues_to_show = issues_after if (auto_fix and issues_after is not None) else issues_before

        if issues_to_show['details']:
            report_lines.append("\n=== Issue Details ===")
            for i, issue in enumerate(issues_to_show['details'], 1):
                if isinstance(issue, dict) and 'file' in issue:
                    report_lines.append(
                        f"\n{i}. [{issue['severity'].upper()}] {issue['file']}:{issue['line']}:{issue['column']}"
                    )
                    report_lines.append(f"   {issue['code']}: {issue['message']}")
        else:
            report_lines.append("\nNo issues found. Code looks good!")

        return "\n".join(report_lines)

    async def execute(
        self,
        computer: ComputerInterface,
        path: str,
        auto_fix: bool = False,
        venv_path: str | None = None,
    ) -> str:
        """
        Execute Ruff linter check and optional auto-fix on Python code.

        All operations are performed through ComputerInterface to support
        sandbox/remote environments.

        Args:
            computer: ComputerInterface for command execution
            path: Path to the Python file or directory to lint
            auto_fix: Whether to automatically fix issues (default: False)
            venv_path: Optional path to a Python venv (e.g. "/home/submission/venv").
                       When given, ruff is looked up / installed inside this venv.

        Returns:
            Linter report as a string
        """
        # 1. Validate path (security check)
        is_valid, error_msg = self._validate_path(path)
        if not is_valid:
            return f"Error: {error_msg}"

        # 2. Check if path exists (via ComputerInterface)
        exists, is_directory, error_msg = await self._check_path_exists(computer, path)
        if not exists:
            return f"Error: {error_msg}"

        # 3. Verify it's a Python path
        is_python = await self._is_python_path(computer, path, is_directory)
        if not is_python:
            return f"Warning: The path '{path}' does not appear to be a Python file or directory containing Python files."

        # 4. Ensure Ruff is installed
        ruff_ok, ruff_error = await self._ensure_ruff_installed(computer, venv_path)
        if not ruff_ok:
            return f"Error: {ruff_error}"

        # 5. Run Ruff check
        stdout, stderr = await self._run_ruff(computer, path, fix=False, venv_path=venv_path)

        # Check for command execution error
        if "error" in stderr.lower() and "ruff" in stderr.lower():
            return f"Error running linter: {stderr}"

        issues_before = self._parse_ruff_output(stdout, stderr)

        # 6. Run fix if requested and there are issues
        issues_after = None
        if auto_fix and issues_before['total_issues'] > 0:
            await self._run_ruff(computer, path, fix=True, venv_path=venv_path)

            # Run check again to see what's left
            stdout_after, stderr_after = await self._run_ruff(computer, path, fix=False, venv_path=venv_path)
            issues_after = self._parse_ruff_output(stdout_after, stderr_after)

        # 7. Format and return report
        report = self._format_report(issues_before, issues_after, auto_fix)
        return report

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Run Ruff linter on Python files or directories to check for code quality issues and optionally fix them automatically. "
                "Returns a detailed report of issues found, including file location, error codes, and messages. "
                "Use this tool to identify and fix code style issues, potential bugs, and enforce coding standards. "
                "Supports only Python code; returns a warning for non-Python files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the Python file or directory to lint. "
                            "Can be an absolute path or relative to the workspace root. "
                            "For directories, all Python files will be checked recursively."
                        )
                    },
                    "auto_fix": {
                        "type": "boolean",
                        "description": (
                            "Whether to automatically fix issues that can be safely fixed. "
                            "If False, only reports issues without modifying files. "
                            "Recommended to set False first to review issues, then True to fix."
                        ),
                        "default": False
                    },
                    "venv_path": {
                        "type": "string",
                        "description": (
                            "Path to a Python virtual environment (e.g. '/home/submission/venv'). "
                            "When provided, ruff is installed and run from this venv. "
                            "Recommended to set this to your project's venv to avoid system-level conflicts."
                        ),
                    }
                },
                "required": ["path", "venv_path"],
                "additionalProperties": False,
            },
            strict=False,
        )


# ==============================================================================
# Local Computer Interface for Testing
# ==============================================================================


class LocalComputerInterface:
    """
    A simple local implementation of ComputerInterface for testing.

    This allows running the linter tool locally without a full sandbox setup.
    """

    async def send_shell_command(self, command: str, timeout: int = 60) -> str:
        """Execute a shell command locally."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            # Return only stdout for commands that output structured data (like JSON)
            # stderr is typically for warnings/errors that shouldn't be mixed with output
            return stdout.decode('utf-8', errors='ignore')

        except asyncio.TimeoutError:
            if process:
                process.kill()
                await process.wait()
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Command execution failed: {str(e)}"


# ==============================================================================
# Debug / Test Entry Point
# ==============================================================================


async def _run_local_test(path: str, auto_fix: bool = False):
    """Run linter on a local path for testing."""
    print(f"Linting: {path}")
    print(f"Auto-fix: {auto_fix}")
    print("-" * 60)

    tool = LinterTool()
    computer = LocalComputerInterface()

    result = await tool.execute(
        computer,
        path=path,
        auto_fix=auto_fix
    )
    print(result)


async def main():
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test LinterTool locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lint a single file:
  python linter.py --path /path/to/file.py

  # Lint a directory:
  python linter.py --path /path/to/project/

  # Lint with auto-fix:
  python linter.py --path /path/to/file.py --fix

  # Lint current directory:
  python linter.py --path .
"""
    )
    parser.add_argument(
        "--path",
        type=str,
        # required=True,
        default=".",
        help="Path to the Python file or directory to lint"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Enable auto-fix for safe issues"
    )

    args = parser.parse_args()

    await _run_local_test(args.path, args.fix)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
