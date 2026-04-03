from __future__ import annotations

import unittest

from aisci_domain_mle.prompts.templates import main_agent_system_prompt_for_run


class PromptParityTests(unittest.TestCase):
    def test_main_prompt_keeps_log_references_when_file_bus_enabled(self) -> None:
        prompt = main_agent_system_prompt_for_run(True)
        self.assertIn("exp_log.md", prompt)
        self.assertIn("impl_log.md", prompt)

    def test_main_prompt_strips_log_references_when_file_bus_disabled(self) -> None:
        prompt = main_agent_system_prompt_for_run(False)
        self.assertNotIn("exp_log.md", prompt)
        self.assertNotIn("impl_log.md", prompt)
        self.assertIn(
            "latest experiment subagent results in this conversation",
            prompt,
        )


if __name__ == "__main__":
    unittest.main()
