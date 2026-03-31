from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "Paper mode no longer runs a container-side orchestrator. "
        "The host worker owns the agent loop and LLM client, and Docker is only the code-execution sandbox. "
        "Launch paper jobs with `aisci paper run ...` instead of `python -m aisci_domain_paper.orchestrator`."
    )


if __name__ == "__main__":
    main()
