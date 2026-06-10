# MATH-500: text I/O vs token-id I/O benchmark

`math500_token_vs_text.py` compares a vLLM OpenAI-compatible endpoint (or the
vLLM Router in front of it) on the **MATH-500** dataset under two scenarios, and
reports **accuracy** and **performance** side by side:

**Both scenarios hit the same endpoint, `POST /v1/completions`**, so the *only*
difference is the input/output representation:

| Scenario | 中文 | Endpoint | Input | Output |
|----------|------|----------|-------|--------|
| `text`   | 输入 prompt / 输出 prompt | `POST /v1/completions` | `prompt` = **string** | text |
| `token`  | 输入 token id / 输出 token id | `POST /v1/completions` | `prompt` = **list of token ids** | `token_ids` (via `return_token_ids`) |

To keep the two scenarios comparing the exact same prompt, the prompt is built
once as token ids (chat template applied, or raw, or random). The `token`
scenario sends those ids directly; the `text` scenario sends the **string form**
of the same prompt (rendered via `/detokenize`, with `add_special_tokens=false`
since the ids already include any special tokens). In `--no-chat-template` mode
the `text` prompt is simply the raw problem text and the `token` prompt is its
tokenization.

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

## Prepare the dataset offline (recommended behind a proxy)

Download MATH-500 once to a local JSONL and run with `--data-file` so the
benchmark never touches the network at runtime. The Hub file already has the
expected `problem` / `answer` fields:

```bash
# direct
curl -L "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl" \
     -o math500_test.jsonl

# or via mirror (e.g. behind a restrictive proxy)
curl -L "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl" \
     -o math500_test.jsonl

# or export through the datasets library
python3 -c "from datasets import load_dataset; \
load_dataset('HuggingFaceH4/MATH-500', split='test').to_json('math500_test.jsonl')"
```

Then point the benchmark at it:

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8000 --model "Qwen/Qwen3-8B" \
    --data-file math500_test.jsonl --scenarios text token
```

> Proxy note: a transparent/system proxy can intercept `/v1/completions` and
> route it to a different OpenAI-compatible backend, causing intermittent
> `The model ... does not exist` (404) errors. Make sure the vLLM host bypasses
> the proxy, e.g. `export NO_PROXY=localhost,127.0.0.1,<vllm-host>`.

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

Measure TTFT / TPOT (streaming):

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-samples 100 \
    --concurrency 16 \
    --stream \
    --scenarios text token
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
- `--stream`           Use streaming (SSE) requests and additionally report
  **TTFT** (time to first token) and **TPOT** (time per output token). Without it,
  only end-to-end generation latency is measured.
- `--ignore-eos`       Force every request to emit exactly `--max-tokens` tokens
  (sets `ignore_eos` + `min_tokens`). Use for **pure performance** runs so both
  scenarios generate the same number of tokens (removes output-length as a
  confound). Makes accuracy meaningless.
- `--no-chat-template` Do **not** apply the chat template: the text scenario
  sends the raw problem text and the token scenario sends its raw token ids.
  Recommended for pure performance testing; keep the template **on** for accuracy
  on instruct models.
- `--random-input`     Ignore the dataset entirely and send fixed-length
  **random-token** prompts (no dataset/tokenizer needed). Pure throughput/latency
  testing — accuracy is meaningless. Pair with `--ignore-eos`.
- `--random-input-len` Prompt length in tokens for `--random-input` (default 1024).
- `--random-vocab-size` Upper bound for random token ids (default 32000).
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
- **TTFT** (with `--stream`) — time to first token: mean / median / p90 / p99 (ms).
- **TPOT** (with `--stream`) — time per output token (mean inter-token latency
  during decode, excluding the first token): mean / median / p90 (ms/tok).
- **Output throughput** — completion tokens per second.
- **Mean output tokens** — average completion length.
- **Tokenize / Detokenize overhead** — extra client-side cost in the token
  scenario (reported separately so the generation latency stays comparable).

It also prints a `token vs text` comparison: accuracy delta (percentage points),
generation-latency speedup, and throughput ratio.

## Two modes: accuracy vs pure performance

| Goal | Recommended flags |
|------|-------------------|
| **Accuracy** comparison | chat template ON (default), natural stopping, `--temperature 0.0`, run the full 500 |
| **Pure performance** comparison | `--no-chat-template --ignore-eos --stream --max-tokens <N>` |

For pure performance the prompt content is irrelevant — what matters is the
input/output token counts. `--ignore-eos` pins the output length to `--max-tokens`
so the `text` and `token` scenarios generate the same number of tokens, which is
required for a fair latency / throughput / TPOT comparison. The chat template is
unnecessary in this mode, so `--no-chat-template` sends raw text vs raw token ids
through `/v1/completions` for both scenarios — the cleanest apples-to-apples
isolation of server-side (de)tokenization overhead.

Pure-performance example:

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-samples 200 \
    --concurrency 32 \
    --no-chat-template --ignore-eos --stream \
    --max-tokens 1024 \
    --scenarios text token
```

Synthetic random-input throughput test (no dataset):

```bash
python scripts/benchmarks/math500_token_vs_text.py \
    --base-url http://127.0.0.1:8090 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --random-input --random-input-len 1024 \
    --ignore-eos --max-tokens 512 \
    --stream --concurrency 64 \
    --num-samples 500 \
    --scenarios text token
```

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
