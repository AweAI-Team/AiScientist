from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aisci_domain_mle.input_resolver import build_dry_run_report, build_phase1_job_spec
from aisci_domain_mle.orchestrator import run as run_orchestrator
from aisci_domain_mle.orchestrator_runtime import build_smoke_runtime_config
from aisci_domain_mle.runtime_orchestration import build_runtime_plan


def _add_common_mle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name")
    parser.add_argument("--zip", dest="zip_path")
    parser.add_argument("--mlebench-data-dir")
    parser.add_argument("--workspace-zip")
    parser.add_argument("--competition-bundle-zip")
    parser.add_argument("--data-dir")
    parser.add_argument("--code-repo-zip")
    parser.add_argument("--description-path")
    parser.add_argument("--sample-submission-path")
    parser.add_argument("--validation-command")
    parser.add_argument("--grading-config-path")
    parser.add_argument("--metric-direction")
    parser.add_argument("--llm-profile", default="gpt-5.4")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--time-limit", default="24h")
    parser.add_argument("--dockerfile")
    parser.add_argument("--objective", default="mle optimization job")
    parser.add_argument("--run-final-validation", action="store_true")
    parser.add_argument("--detach", dest="detach", action="store_true")
    parser.add_argument("--wait", dest="detach", action="store_false")
    parser.set_defaults(detach=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain-local MLE migration tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="normalize inputs and optionally dry-run")
    _add_common_mle_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true")

    runtime_parser = subparsers.add_parser(
        "runtime-plan",
        help="preview the shared MLE runtime/session commands without executing them",
    )
    _add_common_mle_args(runtime_parser)
    runtime_parser.add_argument("--run-root", required=True)
    runtime_parser.add_argument("--build-policy", default="auto", choices=("auto", "force", "never"))
    runtime_parser.add_argument("--docker-binary", default="docker")
    runtime_parser.add_argument("--image-tag")
    runtime_parser.add_argument("--container-name")

    smoke_parser = subparsers.add_parser(
        "orchestrator-smoke",
        help="run the Phase 4 orchestrator against a local stub LLM and temp workspace",
    )
    smoke_parser.add_argument("--run-root", required=True)
    smoke_parser.add_argument("--scenario", default="submit_sample")
    smoke_parser.add_argument("--max-steps", type=int, default=3)
    smoke_parser.add_argument("--time-limit-secs", type=int, default=300)
    smoke_parser.add_argument("--file-as-bus", dest="file_as_bus", action="store_true")
    smoke_parser.add_argument("--no-file-as-bus", dest="file_as_bus", action="store_false")
    smoke_parser.set_defaults(file_as_bus=True)
    return parser


def _build_job_spec(args: argparse.Namespace, *, dry_run: bool) -> object:
    return build_phase1_job_spec(
        competition_name=args.name,
        competition_zip_path=args.zip_path,
        mlebench_data_dir=args.mlebench_data_dir,
        workspace_bundle_zip=args.workspace_zip,
        competition_bundle_zip=args.competition_bundle_zip,
        data_dir=args.data_dir,
        code_repo_zip=args.code_repo_zip,
        description_path=args.description_path,
        sample_submission_path=args.sample_submission_path,
        validation_command=args.validation_command,
        grading_config_path=args.grading_config_path,
        metric_direction=args.metric_direction,
        llm_profile=args.llm_profile,
        gpus=args.gpus,
        time_limit=args.time_limit,
        dockerfile=args.dockerfile,
        run_final_validation=args.run_final_validation,
        dry_run=dry_run,
        objective=args.objective,
    )


def run_mle_phase1(args: argparse.Namespace) -> int:
    try:
        job_spec = _build_job_spec(args, dry_run=args.dry_run)
        if args.dry_run:
            report = build_dry_run_report(job_spec, wait_requested=not args.detach)
            print(json.dumps(report.to_dict(), indent=2, default=str))
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        "Phase 1 domain-local CLI only supports --dry-run. "
        "Main aisci CLI execution requires follow-up integration in src/aisci_app and src/aisci_core.",
        file=sys.stderr,
    )
    return 2


def runtime_plan(args: argparse.Namespace) -> int:
    try:
        job_spec = _build_job_spec(args, dry_run=True)
        dry_run_report = build_dry_run_report(job_spec, wait_requested=not args.detach)
        resolved_inputs = dry_run_report.resolved_inputs
        plan = build_runtime_plan(
            job_spec=job_spec,
            resolved_inputs=resolved_inputs,
            run_root=args.run_root,
            build_policy=args.build_policy,
            dockerfile_path=args.dockerfile,
            docker_binary=args.docker_binary,
            image_tag=args.image_tag,
            container_name=args.container_name,
        )
        print(
            json.dumps(
                {
                    "dry_run_report": dry_run_report.to_dict(),
                    "runtime_plan": plan.to_dict(),
                },
                indent=2,
                default=str,
            )
        )
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def orchestrator_smoke(args: argparse.Namespace) -> int:
    runtime = build_smoke_runtime_config(
        args.run_root,
        scenario=args.scenario,
        max_steps=args.max_steps,
        time_limit=args.time_limit_secs,
        file_as_bus=args.file_as_bus,
    )
    run_orchestrator(runtime)

    summary_path = Path(runtime.paths.agent_summary_path)
    env_path = Path(runtime.paths.agent_env_path)
    payload = {
        "run_root": str(Path(args.run_root).resolve()),
        "home_root": runtime.paths.home_root,
        "logs_dir": runtime.paths.logs_dir,
        "summary_path": str(summary_path),
        "env_path": str(env_path),
        "submission_path": runtime.paths.submission_csv_path,
        "summary_exists": summary_path.exists(),
        "submission_exists": Path(runtime.paths.submission_csv_path).exists(),
    }
    if summary_path.exists():
        payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        raise SystemExit(run_mle_phase1(args))
    if args.command == "runtime-plan":
        raise SystemExit(runtime_plan(args))
    if args.command == "orchestrator-smoke":
        raise SystemExit(orchestrator_smoke(args))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
