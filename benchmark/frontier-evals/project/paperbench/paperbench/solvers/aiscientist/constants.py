"""
Shared constants for AI Scientist solver.

Centralized workspace reference tables used across main agent and all subagent prompts.
Each variant is tailored to its audience's role and needs.
"""

# =============================================================================
# Main Agent Workspace Reference
# - 2-column (no Availability) — main agent controls the workflow and knows what exists
# - Personalized descriptions ("Your output git repository")
# - Used in: prompts/templates.py
# =============================================================================

MAIN_AGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content |
|------|---------|
| `/home/paper/paper.md` | Research paper (markdown) |
| `/home/paper/paper.pdf` | Research paper (PDF) |
| `/home/paper/rubric.json` | Grading rubric (may be partial) |
| `/home/paper/addendum.md` | Scope clarifications, what's in/out |
| `/home/paper/blacklist.txt` | Blocked resources — do NOT access |
| `/home/submission/` | Your output git repository |
| `/home/submission/reproduce.sh` | Reproduction script (REQUIRED for grading) |
| `/home/agent/paper_analysis/` | Created by `read_paper`: `summary.md`, `algorithm.md`, `experiments.md`, `baseline.md`, `structure.md` |
| `/home/agent/prioritized_tasks.md` | Created by `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan (created by plan subagent) |
| `/home/agent/impl_log.md` | Implementation changelog (appended by implement subagent) |
| `/home/agent/exp_log.md` | Experiment log (appended by experiment subagent) |
| `/home/agent/experiments/` | Experiment output logs: `[task_id]/[run_id].log` |"""


# =============================================================================
# Experiment Agent Workspace Reference
# - 3-column with Availability — experiment agent may be spawned at any stage
# - reproduce.sh prominently listed — validating it is a key experiment task
# - exp_log.md says "by you" — experiment agent owns this file
# - Used in: subagents/experiment.py
# =============================================================================

EXPERIMENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper (markdown) | Always |
| `/home/paper/paper.pdf` | Research paper (PDF) | Always |
| `/home/paper/rubric.json` | Grading rubric with scoring weights | Always |
| `/home/paper/addendum.md` | Scope clarifications, what's in/out | Always |
| `/home/paper/blacklist.txt` | Blocked resources — do NOT access | Always |
| `/home/submission/` | Code repository (git) | Always |
| `/home/submission/reproduce.sh` | Reproduction script (REQUIRED for grading) | May not exist yet |
| `/home/agent/paper_analysis/` | Paper analysis: `summary.md`, `algorithm.md`, `experiments.md`, `baseline.md`, `structure.md` | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list (P0-P3) | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After plan subagent runs |
| `/home/agent/impl_log.md` | Implementation changelog | Appended by implement subagent |
| `/home/agent/exp_log.md` | Experiment results log | Appended by you via `add_exp_log` |
| `/home/agent/experiments/` | Experiment output logs: `[task_id]/[run_id].log` | Created by `exec_command` |

**Tip**: Use `ls` or `test -f <path>` to verify a file exists before reading it."""


# =============================================================================
# Implementation Agent Workspace Reference
# - 3-column with Availability — impl agent may be spawned at any stage
# - /home/submission/ says "your workspace" — impl agent owns the code
# - impl_log.md says "by you" — impl agent appends to it
# - Used in: subagents/implementation.py
# =============================================================================

IMPLEMENTATION_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper (markdown) | Always |
| `/home/paper/paper.pdf` | Research paper (PDF) | Always |
| `/home/paper/rubric.json` | Grading rubric with scoring weights | Always |
| `/home/paper/addendum.md` | Scope clarifications, what's in/out | Always |
| `/home/paper/blacklist.txt` | Blocked resources — do NOT access | Always |
| `/home/submission/` | Code repository (git) — your workspace | Always |
| `/home/submission/reproduce.sh` | Reproduction script (REQUIRED for grading) | May not exist yet |
| `/home/agent/paper_analysis/` | Paper analysis: `summary.md`, `algorithm.md`, `experiments.md`, `baseline.md`, `structure.md` | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list (P0-P3) | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After plan subagent runs |
| `/home/agent/impl_log.md` | Implementation changelog | Appended by you via `add_impl_log` |
| `/home/agent/exp_log.md` | Experiment results log | Appended by experiment subagent |
| `/home/agent/experiments/` | Experiment output logs: `[task_id]/[run_id].log` | Created by experiment subagent |

**Tip**: Use `ls` or `test -f <path>` to verify a file exists before reading it."""


# =============================================================================
# Generic Subagent Workspace Reference
# - 3-column with Availability — subagents may be spawned at any stage
# - plan.md included — plan subagent writes it, explore/general may read it
# - reproduce.sh included — general subagent may create/modify it
# - Used in: subagents/generic.py (explore, plan, general)
# =============================================================================

SUBAGENT_WORKSPACE_REFERENCE = """## Workspace Reference

| Path | Content | Availability |
|------|---------|--------------|
| `/home/paper/paper.md` | Research paper (markdown) | Always |
| `/home/paper/paper.pdf` | Research paper (PDF) | Always |
| `/home/paper/rubric.json` | Grading rubric with scoring weights | Always |
| `/home/paper/addendum.md` | Scope clarifications, what's in/out | Always |
| `/home/paper/blacklist.txt` | Blocked resources — do NOT access | Always |
| `/home/submission/` | Code repository (git) | Always (may be empty initially) |
| `/home/submission/reproduce.sh` | Reproduction script (REQUIRED for grading) | May not exist yet |
| `/home/agent/paper_analysis/` | Paper analysis: `summary.md`, `algorithm.md`, `experiments.md`, `baseline.md`, `structure.md` | After `read_paper` |
| `/home/agent/prioritized_tasks.md` | Priority-ranked task list (P0-P3) | After `prioritize_tasks` |
| `/home/agent/plan.md` | Implementation plan | After plan subagent runs |
| `/home/agent/impl_log.md` | Implementation changelog | Appended by implement subagent |
| `/home/agent/exp_log.md` | Experiment results log | Appended by experiment subagent |

**Tip**: Use `ls` or `test -f <path>` to verify a file exists before reading it."""
