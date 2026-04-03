from __future__ import annotations

from aisci_domain_mle.vendored_lite import import_mlebench_module, vendored_lite_repo_root


def main() -> None:
    module = import_mlebench_module(
        "mlebench.cli",
        repo_root=vendored_lite_repo_root(),
    )
    module.main()


if __name__ == "__main__":
    main()
