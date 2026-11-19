# fusedjoin — pipeline 源码副本

`D:\FDJ` 中实现 fused join 部署 pipeline 的源文件，**原样复制**，未做任何修改
（SHA-256 与原文件逐一核对一致）。源文件保持不动。

## 文件与原路径

| 文件 | 原路径 | 在 pipeline 中的作用 |
|---|---|---|
| `run_hybrid_join_v10_10.py` | `D:\FDJ\` | `fused_score`（`_pct`、dense+TF-IDF 百分位融合）、`mutual_nn_agree`（auto）、`build_candidates`、`load_data`、模型加载 |
| `run_fusedjoin.py` | `D:\FDJ\` | `fused_candidates`（C = 行 top-K ∪ 列 top-K）、`_frame` |
| `run_hybrid_join_v10_6_old.py` | `D:\FDJ\` | `tau_reject`（τR 行级 CP 免费拒绝）、`_gate_prompt` / `_llm_gate`（LLM 升级门）、`_final_metrics` |
| `cert_common.py` | `D:\FDJ\fused join exp\` | `build_prompt`（按数据集的教师/门 prompt）、`DS_SPECS`（prompt 截断口径） |
| `run_hybrid_join_v12.py` | `D:\FDJ\` | `run_fusedjoin` 与 `v10_6_old` 的导入依赖 |
| `fdj_struct_buckets.py` | `D:\FDJ\` | `v10_6_old` 的导入依赖 |
| `fdj_cx12.py` | `D:\FDJ\` | `v10_6_old` 的导入依赖（依赖 `fdj_struct_buckets`） |

后三个是导入闭包：不带上它们，前四个模块无法 `import`。均未改动。

## Pipeline 五阶段对应

```
fused_score (v10_10)                    s = w·pct(dense)+(1−w)·pct(sparse)
fused_candidates (run_fusedjoin)        C = row top-K ∪ col top-K
mutual_nn_agree (v10_10)                auto，0 次 LLM
tau_reject (v10_6_old)                  C∖auto 中 s ≤ τR 免费拒绝
_llm_gate (v10_6_old)                   s > τR 的带 → LLM yes/no
M = auto ∪ gate-yes                     发射集
```

prompt 由 `cert_common.build_prompt` + `DS_SPECS` 给出；human GT 只用于最终评估。

## 验证

在 `Desktop\fusedjoin` 下直接导入通过：

```python
import run_hybrid_join_v10_10 as v10
import run_fusedjoin as RJ
import run_hybrid_join_v10_6_old as M
import cert_common as CC
```

`v10.fused_score`、`RJ.fused_candidates`、`M.tau_reject`、`M._llm_gate`、
`CC.build_prompt` 均可调用。

运行门需要 `OPENAI_API_KEY` / `OPENAI_API_BASE` 环境变量（本文件夹内无任何密钥）。
