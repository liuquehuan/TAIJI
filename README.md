# TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes

The variety of data in data lakes presents significant challenges for data analytics, as data scientists must simultaneously analyze multi-modal data, including structured, semi-structured, and unstructured data. While Large Language Models (LLMs) have demonstrated promising capabilities, they still remain inadequate for multi-modal data analytics in terms of accuracy, efficiency, and freshness. First, current natural language (NL) or SQL-like query languages may struggle to precisely and comprehensively capture users' analytical intent. Second, relying on a single unified LLM to process diverse data modalities often leads to substantial inference overhead. Third, data stored in data lakes may be incomplete or outdated, making it essential to integrate external open-domain knowledge to generate timely and relevant analytics results.
In [this paper](https://arxiv.org/abs/2505.11270), we envision a new multi-modal data analytics system. Specifically, we propose a novel architecture built upon the Model Context Protocol (MCP), an emerging paradigm that enables LLMs to collaborate with knowledgeable agents.

## Python SDK

Python 3.10+。本仓库的发行名暂定为 `taiji-sdk`，导入名为 `taiji`。目前可从源码或 wheel 安装，尚未发布到 PyPI。

```bash
python -m pip install .
# 或先构建，再把 dist/ 中的 wheel 安装到其他环境：
python -m pip install build
python -m build
python -m pip install dist/taiji_sdk-0.1.0-py3-none-any.whl
```

在 Python 中配置主模型、注册数据源和算子 MCP 服务，再以自然语言查询：

```python
import asyncio
import os
import taiji

async def main():
    model_config = {
        "model": os.environ["OPENAI_MODEL"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "api_key": os.environ["OPENAI_API_KEY"],
    }
    # 通用 stdio MCP 配置；替换为你自己的服务启动命令。
    data_mcp_config = {"command": "python", "args": ["/absolute/path/data_server.py"], "env": {}}
    operator_mcp_config = {"command": "python", "args": ["/absolute/path/operator_server.py"], "env": {}}

    async with taiji.Client(model=model_config) as client:
        await client.add_mcp("data", data_mcp_config)
        # 数据源提供 register_table 工具；在同一个服务中分别注册两张表。
        await client.call_tool("data", "register_table", {
            "name": "company_left", "path": "/absolute/path/left.csv",
        })
        await client.call_tool("data", "register_table", {
            "name": "company_right", "path": "/absolute/path/right.csv",
        })
        await client.add_mcp("operator", operator_mcp_config)
        result = await client.query("找出左右表中属于同一家公司的记录")
        print(result.answer)
        print(result.status)
        print(result.artifacts)
        for call in result.tool_calls:
            print(call.server, call.tool, call.is_error)

asyncio.run(main())
```

Notebook 中直接 `await main()`。请在同一个异步任务中打开 Client、注册服务并关闭连接；每个 Client 内的自然语言查询依次执行。Client 保留成功完成的对话，`client.clear_history()` 可清空历史。

- `ModelConfig` 支持 OpenAI-compatible Chat Completions 接口，`base_url` 可以是 `/v1` 或完整 `/v1/chat/completions` 地址。`ModelConfig.from_env()` 显式读取环境变量；SDK 不自动加载 `.env`。
- 主模型默认使用原生 function calling。只支持原示例 JSON 工具调用方式的接口可配置 `tool_mode="json"`。
- `MCPConfig` / 字典支持 `command / args / env / cwd / timeout`；目前支持本地 stdio。工具按服务名路由，允许不同服务提供同名工具。
- `client.list_tools()` 查看发现的工具；`await client.call_tool(server, tool, arguments)` 可绕过主模型直接调用。
- `QueryResult` 包含 `answer / tool_calls / artifacts / status`，可用 `to_dict()` 导出。`artifacts` 每项为 `{server, tool, data}`，保留完整工具返回；主模型只接收摘要与有限预览。`status` 为 `success / partial / failed`，工具错误或达到 `max_turns` 上限返回 `partial`，模型通信错误抛出 `taiji.ModelError`。

核心 wheel 只包含通用 SDK，不包含 company 数据、fusedjoin 源码、模型权重、密钥或算法依赖。原 PostgreSQL 和图像 MCP 服务仍可通过相同配置接入。

## Company × fusedjoin 可执行示例

入口 `examples/company_query.py` 在一个 `main()` 中完成模型配置、MCP 注册和自然语言查询。这是两张不同公司表之间的实体匹配。先连接一个 `company` 数据源 MCP，再显式调用两次 `register_table`，将 `left.csv` 和 `right.csv` 分别注册为该服务中的 `company_left`、`company_right`。算子独立注册为 `fusedjoin`，因此一共是两个 MCP 服务。数据源为 `examples/company_mcp.py`，算子为 `examples/fusedjoin_mcp.py`，辅助代码位于 `examples/company_demo/`。它们不进入核心 wheel，运行示例需要保留本仓库源码及 company 数据目录。

```bash
python -m pip install '.[dev]'
python -m pip install -r examples/company_demo/requirements.txt
# 在仓库根目录运行；示例显式加载本地 .env。
python -m examples.company_query
```

`.env` 配置 `OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_API_KEY`。算子默认复用它们，也可分别设置 `VERIFY_MODEL / VERIFY_BASE_URL / VERIFY_API_KEY`。`EMBEDDING_MODEL` 可设为 SentenceTransformer 模型名或本地模型目录，默认 `sentence-transformers/all-MiniLM-L6-v2`；首次下载到仓库 `.model-cache/`，后续复用模型文件。

主模型与算子分别读取配置。算子模型密钥通过 MCP 的 `env` 传入，不添加到工具描述或查询提示词。真实运行会向配置的外部模型接口发送查询工具信息和需要判断的公司描述。

Notebook 中也可以直接运行同一个示例，并通过返回的 Python 对象获取完整结果：

```python
from examples.company_query import main

result = await main()
# result.answer / result.artifacts / result.tool_calls / result.status
```

两个服务使用相同的 Python 解释器和本地文件系统，通过绝对 CSV 路径交换数据。数据源启动时表目录为空，Python 通过 `register_table(name, path)` 注册每张表；名称必须唯一，文件路径为绝对路径。注册时自动识别 `id / name / description` 或一组共同前缀的字段，例如 `left_id / left_name / left_description`、`right_*` 或 `catalog_*`，不绑定表的连接角色。两张表可以位于不同目录。注册只在当前服务进程中有效，示例每次运行都会重新注册表，已保存的算法判断缓存仍可复用。

字段不符合上述命名或存在多组可识别字段时，需显式提供可选映射：`register_table(name, path, fields={"id": "uid", "name": "title", "description": "text"})`。字段不明确时会报错，不会按注册名称猜测。

`list_tables()` 列出注册的表；`get_table(name)` 返回 `{dataset, name, table}`，其中 `table` 包含 `path / fields / row_count / sha256`；`preview_table(name)` 查看指定表。主模型按名称分别取得两份引用，再调用算子的 `join_tables(left, right)`，本次连接的左右角色在这里指定。算子检查字段、唯一 ID 和文件哈希，并返回原始字符串 ID（保留前导零）。空名称、描述归一化为空字符串；空 ID 或重复 ID 报错。其他算法可通过同一表引用与匹配结果契约替换该服务，无需更改 TAIJI 核心。

算子保留 `fusedjoin/` 原始源码，复用 `run_fusedjoin.fused_candidates`、`v10.fused_score / mutual_nn_agree`、`M.tau_reject` 和 `cert_common.build_prompt("company", ...)`。默认 `k=10 / weight=0.5 / audit_rows=20 / epsilon=0.05 / seed=0`。先在随机审计左行的候选上生成 teacher 判断估计阈值，再对非自动接受、非阈值拒绝配对生成 neutral 判断；两个阶段都使用已确认的 company 全文提示词，输出保留全部通过配对，不额外强制一对一。默认 20 行是 demo 的运行预算配置，可在验证脚本中用 `--audit-rows` 调整；这里不宣称实验中的精度或统计认证保证。

原有 lenient/strict 门使用不同提示词且把最终失败映射为 False，不能直接拿来作为 company neutral 门。适配层采用独立的有限并发请求、重试和缓存实现；仅明确的 Yes/No 入缓存，失败配对保留未决状态。teacher 不完整时关闭阈值拒绝。缓存按阶段、模型、接口、company 提示词及记录内容隔离，位于 `.taiji-cache/`，`--no-cache` 禁用，删除该目录即可清空。MCP 服务日志写 stderr。

遇到 HTTP 400/401/403/404 这类模型配置或认证错误时，算子停止发起后续请求，保留已有匹配及未决配对，并在统计中报告 `fatal_model_error`。请在本地修正模型名、接口或密钥后重跑。

运行产物在 `outputs/company/<run-id>/`：

- `matches.csv`：`left_id / right_id / score / source`；融合分数不是概率。
- `result.json`：完整匹配、统计、未决配对、算法配置及运行状态。
- `query.json`（验证脚本生成）：自然语言回答、实际工具调用参数和结果、直接调用对照及可选评估。

默认 demo 的回答只展示匹配结果和文件路径，不展示缓存统计或耗时说明；缓存与完整统计仍正常保存在结果文件中。

对照、评估和自定义参数放在独立的验证脚本中：

```bash
python -m examples.company_validation --compare --evaluate
```

`--compare` 先直接调用算子，再用同配置自然语言查询，比较匹配、未决状态与配置；两次调用均可命中缓存。只执行算子可使用 `python -m examples.company_validation --direct`。原来的 `python -m examples.company_query --compare --evaluate` 等带参数命令仍兼容，会转入验证脚本。验证配置组装见 `examples/company_validation.py`；`TAIJI_JOIN_CONFIG` 可配置 `source_dir / k / weight / audit_rows / epsilon / seed / workers / retries / request_timeout`，验证模型和 embedding 单独通过服务环境变量配置。

独立留出评估（结果不提供给主模型或算子）：

```bash
python -m examples.company_demo.evaluate outputs/company/<run-id>/matches.csv --output outputs/evaluation.json
```

评估同时剔除 train.csv 的全部 200 个正负校准对，使用 60 个留出正例，评估域为 9,800 对。未标注配对按负例计是此 benchmark 的规则。训练标签和测试标签均不用于本适配器的匹配判断，teacher 由模型生成。对部分完成的结果，指标仅描述当前已输出的匹配，应结合未决计数理解。

开发验证：`python -m pytest -q`。离线测试通过真实 MCP 子进程和确定性的替代 join 服务检查跨服务传参、同名工具路由、结果预览、错误处理与清理；该替代服务的匹配结果不代表 fusedjoin 效果。

## fusedjoin 与纯 LLM 性能对照

```bash
python -m examples.company_benchmark
# 只查看配对范围、块数和并发配置，不调用模型：
python -m examples.company_benchmark --dry-run
```

默认比较两种匹配方法，使用同一 `.env` 中的验证模型（VERIFY_*，未配置时继承 OPENAI_*）、temperature=0、并发 4 和相同重试规则：

- `fusedjoin`：原有融合打分、自动接受、teacher 校准和 neutral 验证，关闭应用层判断缓存，重新记录实际请求 usage。
- `llm_batch`：纯 LLM 分块匹配。左右表各按 10 条分组，遍历全部 10×10 个块组合，用 100 个块请求覆盖 10,000 个配对。无 embedding、候选筛选或自动接受，允许一对多/多对多。采用相同 company-v7 匹配判据及完整描述，输出每块的匹配 ID 列表。

`--block-size` 可调整分块大小，`--workers` 调整两边并发，`--method fusedjoin` 或 `--method llm_batch` 可单独运行。默认两边均执行一次全新判断，不读取 `.taiji-cache/`。基线没有穷举 10,000 次独立 API 调用；分块减少重复发送的记录和提示词，同时仍覆盖完整配对空间。

产物位于 `outputs/company_benchmark/<run-id>/`：

- `comparison.csv / comparison.json / REPORT.md`：precision、recall、F1、输入/输出/总 token、匹配墙钟耗时、API 请求数、未决数和完整实验配置。
- 每种方法的 `matches.csv / result.json / evaluation.json`。
- `requests.json / requests.jsonl`：每个实际请求的阶段、配对或块 ID、usage、响应状态和延迟；不记录密钥或提示词正文。`llm_batch/blocks.jsonl` 保留逐块结果。

耗时统一统计匹配阶段，计入 fusedjoin 的 embedding 加载/编码和 teacher/neutral 请求，排除双方共有的 Python 模块导入、自然语言规划、MCP 启动与离线评估；模型文件已下载。token 取接口实际返回的 `usage`，包含重试。若有请求未返回有效 usage，总量标为缺失并保留已知小计，不能视为零。服务端可能自动缓存提示词前缀，返回的 cached_tokens 单独记录，总 token 不扣减缓存 token，也不等同于账单金额。

历史 fusedjoin 结果可保留用于准确率参考；历史请求没有 usage，性能对照通过新运行获取实测 token，不能把原判断缓存回放的耗时与无缓存 LLM 比较。两种方法的匹配输入布局和输出格式不同，报告会明确这一点；评估继续独立剔除全部 200 个正负校准对。

# data preparation

```sql
CREATE TABLE furniture(
  aid bigint,
  time bigint,
  neighborhood text,
  title text,
  url text,
  price bigint,
	PRIMARY KEY (aid)
);

CREATE TABLE img(
  aid bigint,
  img text,
	FOREIGN KEY (aid) REFERENCES furniture(aid)
);

CREATE TABLE img_3000(
  aid bigint,
  img text,
	FOREIGN KEY (aid) REFERENCES furniture(aid)
);

CREATE UNIQUE INDEX furniture_aid_idx ON furniture (aid);

COPY furniture (time, neighborhood, title, url, price, aid) FROM 'furnitures.csv' CSV HEADER;
COPY img (img, aid) FROM 'imgs.csv' CSV HEADER;
COPY img_3000 (img, aid) FROM '3000_imgs.csv' CSV HEADER;
```

# build server

```bash
mkdir mcp_server
cd mcp_server
# python>=3.10
uv python pin 3.12
uv init
uv add "mcp[cli]"
uv venv
source .venv/bin/activate

# to test server
npx -y @modelcontextprotocol/inspector uv run image_analyzer.py
```

# run

```python
python dbagent.py
```
