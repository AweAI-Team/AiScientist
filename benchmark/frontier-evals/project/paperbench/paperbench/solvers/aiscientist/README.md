# AI Scientist Solver

基于子代理（Subagent）架构的 PaperBench 求解器，用于论文复现任务。

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AiScientistSolver (主代理)                │
│                                                             │
│  Tools: bash, python, read_paper, submit                    │
└────────────────────────┬────────────────────────────────────┘
                         │ 调用工具
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SubagentCoordinator                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Level 0 (串行): StructureSubagent                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Level 1 (并行): Algorithm | Experiments | Baseline  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
paperbench/solvers/aiscientist/
├── solver.py                # 主求解器
├── subagents/
│   ├── base.py              # ⭐ Subagent 基类（核心）
│   ├── coordinator.py       # ⭐ 子代理协调器（核心）
│   ├── generic.py           # 通用子代理实现
│   └── paper_reader.py      # 论文阅读相关子代理
├── tools/
│   ├── paper_reader_tool.py # read_paper 工具
│   └── spawn_subagent_tool.py # 动态创建子代理工具
└── prompts/
    └── templates.py         # 系统提示词模板
```

---

## 核心组件 1: `base.py` - Subagent 基类

`base.py` 定义了子代理的核心抽象，是整个子代理系统的基础。

### 关键类

#### 1. `SubagentCompleteSignal` (完成信号)

```python
class SubagentCompleteSignal(Exception):
    """通过异常机制优雅地处理 subagent 完成"""
    def __init__(self, output: str, artifacts: dict[str, Any] | None = None):
        self.output = output
        self.artifacts = artifacts or {}
```

**设计理由**：使用异常而非返回值，避免在执行循环中硬编码检查 tool name，代码更清晰。

#### 2. `SubagentStatus` (状态枚举)

```python
class SubagentStatus(str, Enum):
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    COMPLETED = "completed"  # 成功完成
    FAILED = "failed"        # 执行失败
    TIMEOUT = "timeout"      # 超时
```

#### 3. `SubagentOutput` (输出结构)

```python
@dataclass
class SubagentOutput:
    subagent_name: str                    # 子代理名称
    status: SubagentStatus                # 执行状态
    content: str                          # 主要输出内容
    artifacts: dict[str, Any]             # 结构化数据（如提取的超参数）
    error_message: str | None             # 错误信息
    num_steps: int                        # 执行步数
    runtime_seconds: float                # 运行时间
    token_usage: dict[str, int]           # Token 使用量
    log_path: str | None                  # 对话日志路径
```

#### 4. `SubagentConfig` (配置)

```python
class SubagentConfig(BaseModel):
    max_steps: int = 50       # 最大步数
    time_limit: int = 300     # 时间限制（秒）
    reminder_freq: int = 10   # 提醒频率（每N步）
    log_dir: str | None       # 日志目录
    output_dir: str           # 输出目录
```

#### 5. `SubagentCompleteTool` (完成工具)

子代理调用此工具来提交结果：

```python
class SubagentCompleteTool(Tool):
    async def execute(self, computer, output, artifacts=None) -> str:
        # 抛出信号，由执行循环捕获
        raise SubagentCompleteSignal(output=output, artifacts=artifacts)
```

#### 6. `Subagent` (抽象基类) ⭐

这是最重要的类，所有子代理都必须继承它：

```python
class Subagent(ABC):
    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,  # LLM 配置
        config: SubagentConfig | None = None,              # 子代理配置
        run_dir: str | None = None,                        # 日志目录
    ):
        ...

    # ============ 必须实现的抽象方法 ============

    @property
    @abstractmethod
    def name(self) -> str:
        """返回子代理的唯一名称"""
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """返回系统提示词"""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """返回可用工具列表"""
        ...

    # ============ 可选覆盖的方法 ============

    def _post_process_output(self, raw_output: str, artifacts: dict) -> tuple[str, dict]:
        """后处理输出，可用于格式化或提取额外信息"""
        return raw_output, artifacts

    # ============ 核心执行方法 ============

    async def run(
        self,
        computer: ComputerInterface,   # 执行环境
        task_description: str,         # 任务描述
        constraints: dict | None,      # 约束（如黑名单）
        context: dict | None,          # 上下文（来自其他子代理）
        run_id: str | None,            # 运行ID（用于日志）
    ) -> SubagentOutput:
        """执行子代理任务"""
        ...
```

### `run()` 方法执行流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 初始化                                                   │
│     - 合并 context 到 task_description                      │
│     - 设置日志路径                                           │
│     - 初始化 LoggableMessages                               │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 主循环 (while num_steps < max_steps)                    │
│     ┌─────────────────────────────────────────────────┐     │
│     │  检查时间限制 → 超时则返回 TIMEOUT              │     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│     ┌─────────────────────────────────────────────────┐     │
│     │  定期发送提醒 (每 reminder_freq 步)              │     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│     ┌─────────────────────────────────────────────────┐     │
│     │  调用 LLM 获取响应                               │     │
│     │  - 处理 LengthFinishReasonError (裁剪消息)      │     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│     ┌─────────────────────────────────────────────────┐     │
│     │  执行工具调用                                    │     │
│     │  - 捕获 SubagentCompleteSignal → 返回 COMPLETED │     │
│     │  - 无工具调用 → 发送 "Continue" 消息            │     │
│     └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 返回 SubagentOutput                                     │
│     - 调用 _post_process_output() 后处理                    │
│     - 包含 status, content, artifacts, token_usage 等      │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件 2: `coordinator.py` - 子代理协调器

`coordinator.py` 负责协调多个子代理的执行，支持依赖管理和并行执行。

### 关键类

#### 1. `SubagentTask` (任务定义)

```python
@dataclass
class SubagentTask:
    subagent: Subagent              # 子代理实例
    task_description: str           # 任务描述
    dependencies: list[str] = []    # 依赖的子代理名称
    context_keys: list[str] = []    # 需要传递的上下文键
```

#### 2. `CoordinatorResult` (协调结果)

```python
@dataclass
class CoordinatorResult:
    outputs: dict[str, SubagentOutput]  # 所有子代理的输出
    synthesized_output: str              # 合成后的输出
    total_runtime_seconds: float         # 总运行时间
    total_tokens: dict[str, int]         # 总 Token 使用量
    all_success: bool                    # 是否全部成功
    failed_subagents: list[str]          # 失败的子代理列表
```

#### 3. `SubagentCoordinator` (主协调器) ⭐

```python
class SubagentCoordinator:
    def __init__(
        self,
        completer_config: BasicAgentTurnCompleterConfig,
        synthesize_fn: Callable[[dict[str, SubagentOutput]], str] | None = None,
    ):
        """
        Args:
            completer_config: LLM 配置
            synthesize_fn: 自定义输出合成函数（默认拼接所有输出）
        """
        ...

    async def run(
        self,
        computer: ComputerInterface,
        tasks: list[SubagentTask],
        constraints: dict | None = None,
    ) -> CoordinatorResult:
        """执行所有任务"""
        ...
```

### 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│  1. 拓扑排序 (_topological_sort)                            │
│     - 基于依赖关系确定执行顺序                              │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 按层级分组 (_group_by_level)                            │
│     - Level 0: 无依赖的任务                                 │
│     - Level 1: 依赖 Level 0 的任务                          │
│     - ...                                                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 按层级执行                                               │
│     ┌─────────────────────────────────────────────────┐     │
│     │  同一层级的任务并行执行 (asyncio.gather)         │     │
│     │  不同层级串行执行                                │     │
│     └─────────────────────────────────────────────────┘     │
│                         │                                   │
│                         ▼                                   │
│     ┌─────────────────────────────────────────────────┐     │
│     │  为每个任务构建 context                          │     │
│     │  - 从 context_keys 指定的已完成任务获取输出     │     │
│     └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 合成输出 (synthesize_fn)                                │
│     - 默认: 拼接所有输出，带状态标记 ✓/✗                    │
└─────────────────────────────────────────────────────────────┘
```

### 依赖分组示例

```python
tasks = [
    SubagentTask(StructureSubagent(), "提取结构", dependencies=[]),
    SubagentTask(AlgorithmSubagent(), "提取算法", dependencies=["structure"], context_keys=["structure"]),
    SubagentTask(ExperimentsSubagent(), "提取实验", dependencies=["structure"], context_keys=["structure"]),
    SubagentTask(BaselineSubagent(), "提取基线", dependencies=["structure"], context_keys=["structure"]),
]

# _group_by_level 结果:
# Level 0: [StructureSubagent]           # 先执行
# Level 1: [Algorithm, Experiments, Baseline]  # 并行执行
```

#### 4. `SequentialCoordinator` (顺序协调器)

简化版协调器，严格串行执行所有任务：

```python
class SequentialCoordinator(SubagentCoordinator):
    """用于调试或资源受限场景"""
    async def run(...) -> CoordinatorResult:
        # 按顺序逐个执行任务
        ...
```

---

## 如何创建新的子代理

### 步骤 1: 继承 `Subagent` 基类

```python
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentCompleteTool,
    SubagentConfig,
)
from paperbench.solvers.basicagent.tools import BashTool, PythonTool

class CodeWriterSubagent(Subagent):
    """代码编写子代理"""

    @property
    def name(self) -> str:
        return "code_writer"

    def system_prompt(self) -> str:
        return """You are a code writing specialist.
Your task is to implement algorithms based on paper descriptions.
When done, use subagent_complete to submit your code."""

    def get_tools(self) -> list[Tool]:
        return [
            BashTool(),
            PythonTool(),
            SubagentCompleteTool(),  # 必须包含，用于提交结果
        ]

    def _post_process_output(self, raw_output: str, artifacts: dict) -> tuple[str, dict]:
        """可选：后处理输出"""
        # 例如：提取生成的文件路径
        return raw_output, artifacts
```

### 步骤 2: 使用协调器执行

```python
from paperbench.solvers.aiscientist.subagents.coordinator import (
    SubagentCoordinator,
    SubagentTask,
)

# 创建协调器
coordinator = SubagentCoordinator(completer_config)

# 定义任务
tasks = [
    SubagentTask(
        subagent=CodeWriterSubagent(completer_config),
        task_description="Implement the algorithm from Section 3",
        dependencies=[],
        context_keys=[],
    ),
]

# 执行
result = await coordinator.run(computer, tasks, constraints)
print(result.synthesized_output)
```

---

## 日志与调试

### 对话日志

每个子代理的对话记录在独立文件中：

```
{run_dir}/subagent_logs/{subagent_name}_{run_id}.log
```

### 日志格式

使用 `LoggableMessages` 自动记录，包含：
- 系统提示词
- 用户消息
- 助手响应
- 工具调用和结果

---

## 设计特点

| 特性 | 说明 |
|------|------|
| **API 无关** | 支持 OpenAI/Azure/任意 LLM API |
| **共享环境** | 子代理与主代理共享 ComputerInterface |
| **并行执行** | 无依赖的任务自动并行 |
| **优雅完成** | 使用异常信号而非硬编码检查 |
| **完整日志** | 每个子代理独立记录对话 |
| **灵活扩展** | 继承基类即可创建新子代理 |

---

## 快速参考

### 关键导入

```python
# 基础类
from paperbench.solvers.aiscientist.subagents.base import (
    Subagent,
    SubagentConfig,
    SubagentOutput,
    SubagentStatus,
    SubagentCompleteTool,
    SubagentCompleteSignal,
)

# 协调器
from paperbench.solvers.aiscientist.subagents.coordinator import (
    SubagentCoordinator,
    SequentialCoordinator,
    SubagentTask,
    CoordinatorResult,
)
```

### 常见问题

**Q: 子代理如何返回结果？**  
A: 调用 `subagent_complete` 工具，会抛出 `SubagentCompleteSignal`，被 `run()` 方法捕获。

**Q: 如何在子代理间传递信息？**  
A: 使用 `SubagentTask.context_keys` 指定需要的上下文，协调器会自动传递已完成任务的输出。

**Q: 如何调试子代理？**  
A: 查看 `{run_dir}/subagent_logs/` 下的日志文件，或使用 `SequentialCoordinator` 串行执行。
