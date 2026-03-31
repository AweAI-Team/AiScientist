from __future__ import annotations

from pathlib import Path

from aisci_core.models import JobPaths

MAIN_AGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content |
|------|---------|
| `/home/paper/paper.md` | Research paper in markdown form |
| `/home/paper/paper.pdf` | Original paper PDF |
| `/home/paper/rubric.json` | Optional rubric or scope hints |
| `/home/paper/addendum.md` | Scope clarifications and constraints |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used |
| `/home/submission/` | Output git repository |
| `/home/submission/reproduce.sh` | Reproduction entry point created during implementation |
| `/home/agent/paper_analysis/` | Structured paper analysis created by `read_paper` |
| `/home/agent/prioritized_tasks.md` | Ranked implementation plan created by `prioritize_tasks` |
| `/home/agent/plan.md` | Planning note created by planning work |
| `/home/agent/impl_log.md` | Implementation changelog created on first implementation write |
| `/home/agent/exp_log.md` | Experiment log created on first experiment write |
| `/home/agent/experiments/` | Experiment command logs created by experiment runs |
| `/home/logs/subagent_logs/` | Subagent trace directories created when subagents run |
"""


IMPLEMENTATION_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog written by you | After implementation work |
| `/home/agent/exp_log.md` | Experiment log written by experiment runs | After experiment work |
| `/home/agent/experiments/` | Experiment command logs | After `exec_command` runs |
"""


EXPERIMENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog | After implementation work |
| `/home/agent/exp_log.md` | Experiment log written by you | After experiment work |
| `/home/agent/experiments/` | Experiment command logs | Created by `exec_command` |
"""


SUBAGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper in markdown form | Always |
| `/home/paper/paper.pdf` | Original paper PDF | Always |
| `/home/paper/rubric.json` | Optional rubric or scope hints | Always |
| `/home/paper/addendum.md` | Scope clarifications and constraints | Always |
| `/home/paper/blacklist.txt` | Blocked resources that must not be used | Always |
| `/home/submission/` | Code repository and working tree | Always |
| `/home/submission/reproduce.sh` | Reproduction entry point | May not exist yet |
| `/home/agent/paper_analysis/` | Detailed paper analysis | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After planning work runs |
| `/home/agent/impl_log.md` | Implementation changelog | After implementation work |
| `/home/agent/exp_log.md` | Experiment log | After experiment work |
"""


def workspace_paths(job_paths: JobPaths) -> dict[str, Path]:
    return {
        "paper": job_paths.workspace_dir / "paper",
        "submission": job_paths.workspace_dir / "submission",
        "agent": job_paths.workspace_dir / "agent",
        "logs": job_paths.logs_dir,
        "analysis": job_paths.workspace_dir / "agent" / "paper_analysis",
        "subagent_logs": job_paths.logs_dir / "subagent_logs",
    }
