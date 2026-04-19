import asyncio
import hashlib
import time

import blobfile as bf
import structlog
import tiktoken
from openai import BadRequestError, LengthFinishReasonError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.responses.web_search_tool_param import WebSearchToolParam
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
)
from paperbench.constants import AGENT_DIR_CONFIG
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    OpenAIResponsesTurnCompleterConfig,
)
from paperbench.solvers.basicagent.logging import LoggableMessages
from paperbench.solvers.basicagent.log_utils import (
    log_model_response_event,
    log_tool_result_event,
)
from paperbench.solvers.basicagent.prompts.templates import (
    DEFAULT_CONTINUE_MESSAGE,
    get_system_message,
)
from paperbench.solvers.basicagent.tools import (
    BashTool,
    PythonTool,
    ReadFileChunk,
    SearchFile,
    SubmitTool,
)
from paperbench.solvers.basicagent.tools.base import Tool
from paperbench.solvers.basicagent.utils import (
    cancel_async_task,
    format_progress_time,
    get_instructions,
    handle_sonnet_limits,
    handle_tool_call,
    optionally_upload_heavy_logs,
    prune_messages,
)
from paperbench.solvers.upload import (
    start_periodic_light_log_upload,
    upload_heavy_logs,
)
from paperbench.solvers.utils import (
    check_for_existing_run,
    log_messages_to_file,
    sanity_check_docker,
)
from paperbench.paper_registry import paper_registry
from paperbench.solvers.cus_tools.aweai_mcp.utils import build_blocked_patterns_from_blacklist

logger = structlog.stdlib.get_logger(component=__name__)


def build_safety_identifier(scope: str, paper_id: str, run_id: str) -> str:
    raw = f"paperbench:{scope}:{paper_id}:{run_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"paperbench-{scope}-{digest}"


@chz.chz
class BasicAgentSolver(BasePBSolver):
    completer_config: BasicAgentTurnCompleterConfig = chz.field(
        default_factory=lambda: OpenAIResponsesTurnCompleterConfig(
            model="gpt-4.1-mini",
            tools=[WebSearchToolParam(type="web_search_preview")],
        )
    )
    max_steps: int | None = chz.field(default=None)
    time_limit: int | None = chz.field(default=5 * 60, doc="Time limit in seconds (5min default)")
    use_submit_tool: bool = chz.field(default=True)
    use_real_time_limit: bool = chz.field(
        default=True,
        doc="If True, don't count API retrying time towards time limit",
    )
    iterative_agent: bool = chz.field(default=False)
    max_context_tokens: int | None = chz.field(
        default=None,
        doc="Proactively prune messages when estimated token count exceeds this limit. "
        "If None, only prune reactively on API context-length errors.",
    )

    upload_interval_messages: int | None = chz.field(default=None)
    upload_interval_seconds: int | None = chz.field(default=1800)

    reminder_freq: int = chz.field(default=5, doc="Every how many steps to remind agent of state")

    @override
    def shortname(self) -> str:
        if self.iterative_agent:
            return "iterativeagent"
        else:
            return "basicagent"

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """
        Solver-specific setup, run after task setup and before agent starts.
        Add any pre-commands here that should run before the agent begins working.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )
        ctx_logger.info("Running pre-commands before agent starts...", destinations=["run"])

        # ========== Add your pre-commands here ==========
        # Example: Install Python packages
        # await computer.check_shell_command("pip install torch transformers datasets")

        # Example: Set environment variables
        # await computer.check_shell_command("export CUDA_VISIBLE_DEVICES=0")

        # Example: Check system info
        # run_result = await computer.check_shell_command("nvidia-smi")
        # ctx_logger.info(f"nvidia-smi output: {run_result.output.decode('utf-8')}", destinations=["run"])
        # ================================================

        ctx_logger.info("Pre-commands completed.", destinations=["run"])

    def _get_tools(self) -> list[Tool]:
        if self.iterative_agent:
            tools = [BashTool(), ReadFileChunk()]
        else:
            tools = [BashTool(), PythonTool(), SearchFile(), ReadFileChunk()]
        if self.use_submit_tool:
            tools.append(SubmitTool())
        return tools

    def _build_tool_constraints(self, paper_id: str) -> dict | None:
        """
        Build tool constraints from the paper's blacklist.
        
        This loads the blacklist for the given paper and converts it to
        search pattern constraints that will be applied to search tools.
        
        Args:
            paper_id: The paper ID to load blacklist for
            
        Returns:
            dict with blocked_search_patterns or None if no blacklist
        """
        try:
            paper = paper_registry.get_paper(paper_id)
            if not paper.blacklist.exists():
                return None
            
            blacklist_content = paper.blacklist.read_text().splitlines()
            # Filter out empty lines and comments, and handle "none" marker
            blacklist = [
                line.strip() for line in blacklist_content
                if line.strip() and not line.startswith("#")
            ]
            if blacklist == ["none"] or not blacklist:
                return None
            
            blocked_patterns = build_blocked_patterns_from_blacklist(blacklist)
            if not blocked_patterns:
                return None
            
            return {"blocked_search_patterns": blocked_patterns}
        except Exception as e:
            # Log error but don't fail - constraints are optional
            logger.warning(
                f"Failed to build tool constraints from blacklist: {e}",
                paper_id=paper_id,
            )
            return None

    async def _execute_agent(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
        start_time: int,
    ) -> int:
        """
        Executes the agent on the given prompt:
        - Stops when time limit is reached, messgae limit is reached, or submit tool is used
        - The message history is logged to task.run_dir / agent.log on every change
            to the message history
        - Periodically uploads the "heavy" logs
        - Prunes context window
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        tools = self._get_tools()
        completer_config = self.completer_config.model_copy(deep=True)
        completer_config.basicagent_tools = tools
        if hasattr(completer_config, "safety_identifier"):
            completer_config.safety_identifier = build_safety_identifier(
                "solver", task.paper_id, task.run_id
            )
        completer = completer_config.build()
        messages = LoggableMessages(
            init=prompt, log_path=bf.join(task.run_dir, "agent.log"), logger=log_messages_to_file
        )

        # Load blacklist and build constraints for search tools
        constraints = self._build_tool_constraints(task.paper_id)
        ctx_logger.info(
            f"Built tool constraints from blacklist",
            constraints=constraints,
            destinations=["run"],
        )

        num_steps = 0
        total_retry_time = 0.0
        last_time_uploaded = time.time()

        # Structured conversation event stream (JSON Lines).
        # We intentionally log *incremental events* (model response + tool results).
        convo_path = bf.join(task.run_dir, "conversation.jsonl")

        upload_task = None
        try:
            while not self._has_agent_finished(
                num_steps=num_steps, start_time=start_time, total_retry_time=total_retry_time
            ):
                num_steps += 1
                optional_upload_outcome = await optionally_upload_heavy_logs(
                    computer=computer,
                    task=task,
                    run_dir=task.run_dir,
                    num_steps=num_steps,
                    start_time=start_time,
                    last_time_uploaded=last_time_uploaded,
                    upload_interval_messages=self.upload_interval_messages,
                    upload_interval_seconds=self.upload_interval_seconds,
                )
                last_time_uploaded, upload_task = (
                    optional_upload_outcome.last_time_uploaded,
                    optional_upload_outcome.upload_task,
                )

                # sonnet has a 1000 message limit we have to special case
                messages = handle_sonnet_limits(self.completer_config, messages)

                # Proactive context pruning: count tokens and prune before hitting API limit
                if self.max_context_tokens is not None:
                    enc = tiktoken.get_encoding("o200k_base")
                    token_count = sum(
                        len(enc.encode(str(m.get("content", "")), disallowed_special=()))
                        for m in messages.data
                    )
                    while token_count > self.max_context_tokens and len(messages.data) > 3:
                        ctx_logger.info(
                            f"Proactive prune: {token_count} tokens exceeds limit {self.max_context_tokens}",
                            destinations=["run"],
                        )
                        messages = prune_messages(messages)
                        token_count = sum(
                            len(enc.encode(str(m.get("content", "")), disallowed_special=()))
                            for m in messages.data
                        )

                # Send reminder every self.reminder_freq steps
                if num_steps % self.reminder_freq == 0:
                    messages.append(self._construct_periodic_reminder(start_time, total_retry_time))

                # generate response
                try:
                    n_input_messages = len(messages.data)
                    res = await make_completer_request(completer=completer, messages=messages.data)
                    (response_messages, tool_calls, response_retry_time) = (
                        res.response_messages,
                        res.tool_calls,
                        res.time_spent_retrying,
                    )
                    total_retry_time += response_retry_time

                    # ---- usage logging (per call) ----
                    usage_obj = res.usage
                    usage_dict: dict[str, object] | None = None
                    in_tokens = out_tokens = total_tokens = None
                    if usage_obj is not None:
                        try:
                            # pydantic models (openai types) support model_dump()
                            usage_dict = usage_obj.model_dump(exclude_none=True)  # type: ignore[attr-defined]
                        except Exception:
                            try:
                                usage_dict = dict(usage_obj)  # type: ignore[arg-type]
                            except Exception:
                                usage_dict = {"repr": repr(usage_obj)}

                        # Normalize across Responses vs Completions field names
                        usage_dict_for_norm: dict[str, object] = usage_dict or {}
                        in_tokens = usage_dict_for_norm.get("prompt_tokens") or usage_dict_for_norm.get(
                            "input_tokens"
                        )
                        out_tokens = (
                            usage_dict_for_norm.get("completion_tokens")
                            or usage_dict_for_norm.get("output_tokens")
                        )
                        total_tokens = usage_dict_for_norm.get("total_tokens")
                        if total_tokens is None and isinstance(in_tokens, int) and isinstance(
                            out_tokens, int
                        ):
                            total_tokens = in_tokens + out_tokens

                    # ---- conversation logging (per model response) ----
                    try:
                        log_model_response_event(
                            convo_path=convo_path,
                            run_id=task.run_id,
                            step=num_steps,
                            n_input_messages=n_input_messages,
                            response_messages=response_messages,
                            tool_calls=tool_calls,
                            usage=usage_dict,
                            normalized={
                                "in_tokens": in_tokens,
                                "out_tokens": out_tokens,
                                "total_tokens": total_tokens,
                            },
                        )
                    except Exception as e:
                        ctx_logger.warning(
                            "Failed to write conversation.jsonl (model_response)",
                            convo_path=convo_path,
                            error=str(e),
                            destinations=["run"],
                        )

                except (LengthFinishReasonError, IndexError) as e:
                    ctx_logger.info(
                        "Context length exceeded. Pruning messages...", destinations=["run"]
                    )
                    prune_individual = "PRUNE_INDIVIDUAL_MESSAGES" in str(e)
                    messages = prune_messages(messages, prune_individual=prune_individual)
                    continue
                except BadRequestError as e:
                    err_str = str(e).lower()
                    if "context length" in err_str or "maximum context" in err_str or "exceeds max length" in err_str or "input token count exceeds" in err_str:
                        ctx_logger.info(
                            "Input exceeds context length (BadRequestError). Pruning messages...",
                            destinations=["run"],
                        )
                        messages = prune_messages(messages, prune_individual=True)
                        continue
                    raise
                messages += response_messages

                if tool_calls:
                    for tool_call in tool_calls:
                        handled = await handle_tool_call(
                            tool_call, tools, task, computer, constraints=constraints
                        )
                        if handled is None:  # signal that submit was called; exit.
                            return num_steps
                        messages.append(handled)

                        # ---- conversation logging (per tool result) ----
                        try:
                            log_tool_result_event(
                                convo_path=convo_path,
                                run_id=task.run_id,
                                step=num_steps,
                                tool_call=tool_call,
                                tool_message=dict(handled),
                            )
                        except Exception as e:
                            ctx_logger.warning(
                                "Failed to write conversation.jsonl (tool_result)",
                                convo_path=convo_path,
                                error=str(e),
                                destinations=["run"],
                            )
                else:
                    messages.append({"role": "user", "content": DEFAULT_CONTINUE_MESSAGE})
        except asyncio.TimeoutError as e:
            # gracefully handle upload_task, then re-raise
            if upload_task:
                await asyncio.shield(upload_task)
            raise e

        return num_steps

    def _has_agent_finished(
        self, num_steps: int, start_time: float, total_retry_time: float
    ) -> bool:
        """
        Returns True if the agent has reached the step limit or the time limit.
        """
        over_step_limit = self.max_steps is not None and num_steps > self.max_steps

        if self.use_real_time_limit and self.time_limit:
            over_time_limit = time.time() - start_time - total_retry_time > self.time_limit
        elif self.time_limit:
            over_time_limit = time.time() - start_time > self.time_limit
        else:
            over_time_limit = False

        return over_step_limit or over_time_limit

    def _construct_periodic_reminder(
        self, start_time: float, total_retry_time: float
    ) -> ChatCompletionUserMessageParam:
        """Constructs a message periodically reminding the agent of various tips."""
        if self.use_real_time_limit and self.time_limit:
            elapsed_time = time.time() - start_time - total_retry_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}. Remember, you only have to stop working when the time limit has been reached."
        if self.time_limit:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(self.time_limit)}. Remember, you only have to stop working when the time limit has been reached."
        else:
            elapsed_time = time.time() - start_time
            periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"
        periodic_msg += "\n\nNote: Don't forget to git commit regularly!"
        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": periodic_msg,
        }
        return message

    async def _execute_agent_and_periodically_upload_logs(
        self,
        computer: ComputerInterface,
        task: PBTask,
        prompt: list[ChatCompletionMessageParam],
    ) -> None:
        """
        Starts agent execution and periodically uploads the "light" logs. When agent finishes, it
        cancels periodic uploading, then performs one final upload of the "heavy" logs.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
            destinations=["run"],
        )

        num_steps = -1
        start_time = int(time.time())
        agent_task: asyncio.Task[int] | None = None
        light_periodic_upload_task: asyncio.Task[None] | None = None

        try:
            async with asyncio.timeout(self.time_limit):
                light_periodic_upload_task = await start_periodic_light_log_upload(
                    agent_start_time=start_time,
                    run_dir=task.run_dir,
                    run_group_id=task.run_group_id,
                    runs_dir=task.runs_dir,
                    run_id=task.run_id,
                )

                agent_task = asyncio.create_task(
                    self._execute_agent(
                        computer=computer,
                        task=task,
                        prompt=prompt,
                        start_time=start_time,
                    )
                )

                while not agent_task.done():
                    ctx_logger.info("Waiting for agent to finish...")
                    if light_periodic_upload_task.done():
                        exc = light_periodic_upload_task.exception()
                        if exc:
                            raise exc
                    await asyncio.sleep(60)
                num_steps = await agent_task
        except asyncio.TimeoutError as e:
            ctx_logger.info(
                f"Agent run timed out after {time.time() - start_time} second (timeout: {self.time_limit}): {e}",
            )
        except asyncio.CancelledError as e:
            ctx_logger.info(
                f"Agent run cancelled after {time.time() - start_time} seconds: {e}",
            )
        finally:
            await cancel_async_task(async_task=agent_task, pb_task=task)
            await cancel_async_task(async_task=light_periodic_upload_task, pb_task=task)

        # one last upload
        await upload_heavy_logs(
            computer=computer,
            agent_start_time=int(start_time),
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=task.run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
            num_messages=num_steps,
        )

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Prepares agent, runs agent, then returns agent output. Skips agent run if we already have agent logs
        for this task.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        # Don't run the agent if we already have agent logs
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        await sanity_check_docker(computer)

        start_time = time.time()
        instructions = await get_instructions(computer, task, self.iterative_agent, self.time_limit)
        system_message = get_system_message(self.iterative_agent, task.judge.code_only)
        await self._execute_agent_and_periodically_upload_logs(
            computer=computer,
            task=task,
            prompt=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": instructions},
            ],
        )

        ctx_logger.info(
            f"Agent `{self.shortname()}` finished running for `{task.question_id}.{task.attempt_id}`!",
            destinations=["group", "run"],
            _print=True,
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None,
            runtime_in_seconds=time.time() - start_time,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )
