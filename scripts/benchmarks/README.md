# MATH-500: text I/O vs token-id I/O benchmark

`math500_token_vs_text.py` compares a vLLM OpenAI-compatible endpoint (or the
vLLM Router in front of it) on the **MATH-500** dataset under two scenarios, and
reports **accuracy** and **performance** side by side:

| Scenario | 中文 | Endpoint | Input | Output |
|----------|------|----------|-------|--------|
| `text`   | 输入 prompt / 输出 prompt | `POST /v1/chat/completions` | text `messages` | text `content` |
| `token`  | 输入 token id / 输出 token id | `POST /v1/completions` | `prompt` = list of token ids | `token_ids` (via `return_token_ids`) |

Both scenarios build the **same logical prompt** (the chat template is applied in
both cases), so the only difference is whether text or token ids cross the wire
on the generation request. This isolates the effect of token-id I/O:

- **text** path: server tokenizes the input and detokenizes the output on the hot
  generation request.
- **token** path: tokenization happens up front via `/tokenize`, the generation
  request carries raw token ids, and the response carries raw `token_ids` that
  are turned back to text via `/detokenize`. This avoids server-side
  (de)tokenization on the generation call and avoids "retokenization drift".

> Requires **vLLM >= 0.10.2** for the `return_token_ids` response field. If the
> server is older, the token scenario still sends token-id *input* but falls back
> to reading the completion `text` for the output (a warning is printed).

## Install dependencies

```bash
pip install aiohttp datasets
# optional, improves answer-equivalence checking:
pip install sympy antlr4-python3-runtime
# only needed with --local-tokenizer:
pip install transformers
```

## Usage

Point it at a running vLLM server or the vLLM Router:

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-samples 100 \
    --concurrency 16 \
    --max-tokens 2048 \
    --temperature 0.0 \
    --scenarios text token \
    --output math500_results.json
```

Run only one scenario:

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --scenarios token
```

Use a local dataset file (JSONL with `problem` and `answer` fields) instead of
the Hugging Face Hub:

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model my-model \
    --data-file ./math500.jsonl
```

## Key options

- `--base-url`        vLLM server / router base URL (default `http://127.0.0.1:8000`).
- `--model`           Model name as served by vLLM (**required**).
- `--scenarios`       `text`, `token`, or both (default both).
- `--num-samples`     Limit number of problems (default: all 500).
- `--concurrency`     Max in-flight requests (drives the throughput number).
- `--max-tokens` / `--temperature` / `--top-p` / `--seed`  Sampling params.
- `--local-tokenizer` Tokenize/detokenize with `transformers` locally instead of
  the server `/tokenize` & `/detokenize` endpoints.
- `--no-sympy`        Disable the sympy equivalence fallback (string match only).
- `--output`          Write full JSON results; add `--save-predictions` to include
  raw model outputs.

## What it reports

For each scenario:

- **Accuracy** — fraction of problems whose extracted `\boxed{}` answer matches
  the MATH-500 ground truth (normalized; optional sympy equivalence).
- **Wall time / Requests-per-second** — end-to-end throughput at the chosen
  concurrency.
- **Generation latency** — mean / median / p90 / p99 of the core generation call.
- **Output throughput** — completion tokens per second.
- **Mean output tokens** — average completion length.
- **Tokenize / Detokenize overhead** — extra client-side cost in the token
  scenario (reported separately so the generation latency stays comparable).

It also prints a `token vs text` comparison: accuracy delta (percentage points),
generation-latency speedup, and throughput ratio.

## Notes

- Use `--temperature 0.0` for the most stable accuracy comparison.
- The accuracy of both scenarios should be essentially identical for the same
  prompt; the token path's value is reduced server (de)tokenization work and the
  elimination of retokenization drift. Any large accuracy gap usually points to a
  chat-template / special-token mismatch — try `--local-tokenizer` to verify the
  prompt token ids match what the chat endpoint builds.
- Latency numbers depend heavily on backend load, concurrency, and `--max-tokens`.
  Run `text` and `token` back to back on an otherwise-idle server for a fair
  comparison.
