# 图像分析 MCP 服务

`image_analyzer.py` 从进程环境变量读取模型配置：

- `OPENAI_API_KEY`：模型接口密钥。
- `OPENAI_BASE_URL`：模型接口地址。
- `OPENAI_MODEL`：支持图片输入的模型名称。

通过 TAIJI 启动时，服务继承父进程加载的配置。单独启动前，需要先设置这些环境变量；该服务不会自行加载 `.env`。

本地 `.env` 不纳入 Git 跟踪。共享配置模板见仓库根目录的 `.env.example`，只填写占位内容。
