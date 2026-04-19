import asyncio
import json
from pathlib import Path
from typing import Literal

import structlog.stdlib
from preparedness_turn_completer.turn_completer import TurnCompleter

import chz
from paperbench.grade import JudgeOutput, run_judge
from paperbench.judge.token_usage import get_total_token_usage
from paperbench.utils import get_timestamp

logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class RunJudgeCliArgs:
    submission_path: Path = chz.field(doc="Path to the submission directory.")
    paper_id: str = chz.field(doc="Identifier for the paper.")
    judge: Literal["dummy", "random", "simple"] = chz.field(
        default="dummy",
        doc="Specify the judge to use.",
    )
    max_depth: int = chz.field(default=999, doc="Maximum depth to grade.")
    out_dir: Path = chz.field(doc="Path to store the judge's outputs.")
    code_only: bool = chz.field(
        default=False,
        doc="Set to True to only grade 'Code Development' nodes.",
    )
    resources_provided: bool = chz.field(
        default=False,
        doc="Set to True to weight 'Dataset and Model Acquisition' nodes to 0.",
    )
    completer_config: TurnCompleter.Config | None = chz.field(
        default=None,
        doc=(
            "TurnCompleter configuration to use for judges that require a completer. "
            "Example: completer_config=preparedness_turn_completer.oai_completions_turn_completer:"
            "OpenAICompletionsTurnCompleter.Config"
        ),
    )


async def main(
    submission_path: Path,
    paper_id: str,
    judge_type: str,
    max_depth: int,
    out_dir: Path,
    code_only: bool,
    completer_config: TurnCompleter.Config | None = None,
    resources_provided: bool = False,
) -> None:
    # Judge the submission
    graded_task_tree = await run_judge(
        submission_path=submission_path,
        paper_id=paper_id,
        judge_type=judge_type,
        completer_config=completer_config,
        out_dir=out_dir,
        max_depth=max_depth,
        code_only=code_only,
        resources_provided=resources_provided,
    )

    token_usage = None

    if judge_type == "simple":
        token_usage = get_total_token_usage(graded_task_tree)

    # Save judging outputs
    path_to_judge_output = out_dir / "grader_output.json"
    judge_output = JudgeOutput(
        judge_type=judge_type,
        completer_config=completer_config,
        score=graded_task_tree.score,
        num_leaf_nodes=len(graded_task_tree.get_leaf_nodes()),
        num_invalid_leaf_nodes=len(
            [node for node in graded_task_tree.get_leaf_nodes() if not node.valid_score]
        ),
        graded_at=get_timestamp(),
        graded_task_tree=graded_task_tree,
        token_usage=token_usage,
    )

    with open(path_to_judge_output, "w") as f:
        json.dump(judge_output.to_dict(), f, indent=4)


def _resolve_completer_config(args: RunJudgeCliArgs) -> TurnCompleter.Config | None:
    completer_config = args.completer_config
    if completer_config is not None:
        return completer_config

    if args.judge == "simple":
        raise ValueError("When using the simple judge you must provide `completer_config`.")

    return None


async def _main_from_cli(args: RunJudgeCliArgs) -> None:
    completer_config = _resolve_completer_config(args)
    await main(
        submission_path=args.submission_path,
        paper_id=args.paper_id,
        judge_type=args.judge,
        max_depth=args.max_depth,
        out_dir=args.out_dir,
        code_only=args.code_only,
        completer_config=completer_config,
        resources_provided=args.resources_provided,
    )


if __name__ == "__main__":
    cli_args = chz.entrypoint(RunJudgeCliArgs, allow_hyphens=True)
    asyncio.run(_main_from_cli(cli_args))
