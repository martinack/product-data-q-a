# Synthetic product data

Generate fully synthetic product records and grounded Q&A pairs using a local Ollama model.

## Prerequisites

- Python 3.9+
- Ollama installed and running (the script calls the local Ollama API)

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

## Run

From the repo root:

```bash
python3 data_creation.py --count 3
```

### Useful flags

- `--model`: Ollama model name (default: `gpt-oss:20b`)
- `--output-dir`: where generated files are written (default: current directory)
- `--log-dir`: where conversation logs are written (default: `logs`)
- `--product-data-prompt`: path to the system prompt for product generation (default: `products_system.prompt`)
- `--questions-prompt`: path to the system prompt for Q&A generation (default: `questions_system.prompt`)

## Outputs

For each generated product:

- `<id>-data.json`: the raw JSON response used as the product record
- `<id>-qa.json`: the raw JSON response containing the questions/answers

Conversation logs (including prompts + outputs) are written under `logs/` by default.
