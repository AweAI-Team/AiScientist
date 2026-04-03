from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from aisci_domain_mle.local_runtime_stubs import install_optional_dependency_stubs


class LocalRuntimeStubsTests(unittest.TestCase):
    def test_openai_stub_is_not_injected_when_real_module_is_available(self) -> None:
        saved = {
            name: sys.modules.get(name)
            for name in (
                "openai",
                "openai.types",
                "openai.types.shared_params",
                "openai.types.shared_params.reasoning",
            )
        }
        for name in list(saved):
            sys.modules.pop(name, None)
        try:
            real_openai = types.ModuleType("openai")
            real_openai.__file__ = "/tmp/real_openai.py"
            sys.modules["openai"] = real_openai

            with mock.patch("aisci_domain_mle.local_runtime_stubs.importlib.util.find_spec", return_value=object()):
                install_optional_dependency_stubs()

            self.assertIs(sys.modules["openai"], real_openai)
        finally:
            for name in (
                "openai",
                "openai.types",
                "openai.types.shared_params",
                "openai.types.shared_params.reasoning",
            ):
                sys.modules.pop(name, None)
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
