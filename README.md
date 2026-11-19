# TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes

TAIJI connects large language models with data sources and analytical operators through the Model Context Protocol (MCP), enabling natural-language analytics across structured, semi-structured, and unstructured data. See our [paper](https://arxiv.org/abs/2505.11270) for the architecture.

TAIJI provides two ways to run queries: a Python SDK and an interactive terminal interface.

## 1. Python SDK

The SDK lets you configure a model, connect MCP data sources and operators, and submit natural-language queries from Python.

### Install

Requires Python 3.10+. Install from the repository root:

```bash
python -m pip install .
```

### Run the Company Matching Demo

Install the demo dependencies and create a local configuration:

```bash
python -m pip install -r examples/company_demo/requirements.txt
cp .env.example .env
```

Set `OPENAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` in `.env`, then run:

```bash
python -m examples.company_query
```

The demo registers two company tables and a FusedJoin operator through MCP, then finds records referring to the same real-world company. See [company_query.py](examples/company_query.py) for the Python SDK usage. Results are saved under `outputs/company/`.

## 2. Terminal Interface

The original terminal interface supports interactive queries over PostgreSQL and an image-analysis MCP service.

### Setup

Requires Python 3.12+, Node.js/npm, and PostgreSQL. Run the following commands from the repository root in your Python environment:

```bash
python -m pip install httpx mcp python-dotenv openai
npm install
cp .env.example .env
```

Set `OPENAI_MODEL`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` in `.env`. For the terminal client, use the full endpoint, such as `https://your-provider.example/v1/chat/completions`.

Update the PostgreSQL connection and Python executable in `servers_config.json`, and the image directory in `mcp_server/image_analyzer.py`. Add `"env": {"OPENAI_BASE_URL": "https://your-provider.example/v1"}` to the image server configuration so it inherits the credentials and uses the API base URL.

### Prepare the Data

Load the furniture CSV files into PostgreSQL:

```sql
CREATE TABLE furniture (
    aid bigint PRIMARY KEY,
    time bigint,
    neighborhood text,
    title text,
    url text,
    price bigint
);

CREATE TABLE img (
    aid bigint REFERENCES furniture(aid),
    img text
);

CREATE TABLE img_3000 (
    aid bigint REFERENCES furniture(aid),
    img text
);

COPY furniture (time, neighborhood, title, url, price, aid)
    FROM 'furnitures.csv' CSV HEADER;
COPY img (img, aid) FROM 'imgs.csv' CSV HEADER;
COPY img_3000 (img, aid) FROM '3000_imgs.csv' CSV HEADER;
```

### Run

```bash
python dbagent.py
```

Enter a query such as: "Find the lowest-priced blue wooden chair."
