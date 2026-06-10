#!/usr/bin/env python3
"""
MATH-500 accuracy & performance comparison: text I/O vs token-id I/O.

This script benchmarks a vLLM OpenAI-compatible endpoint (or the vLLM Router in
front of it) on the MATH-500 dataset under two scenarios and reports both
*accuracy* and *performance* side by side:

  1. "text"  (输入 prompt / 输出 prompt):
       POST /v1/chat/completions with text `messages`, read back text `content`.
       Tokenization happens on the server (prefill) and detokenization happens on
       the server (decode -> string).

  2. "token" (输入 token id / 输出 token id):
       - Apply the chat template and tokenize the prompt to token ids
         (via the server's /tokenize endpoint by default, so tokenization is
         guaranteed identical to scenario 1).
       - POST /v1/completions with `prompt` set to that list of token ids,
         requesting `return_token_ids=true`.
       - Read back the generated `token_ids` and turn them into text via
         /detokenize (or the tokenizer locally).
       This path avoids server-side (re)tokenization/detokenization on the hot
       generation request and avoids "retokenization drift".

Both scenarios use the *same logical prompt* (same chat template applied), so the
only difference is whether text or token ids cross the wire on the generation
request. That isolates the effect of token-id I/O on accuracy and latency.

Requires vLLM >= 0.10.2 for `return_token_ids`. (Without it, the token scenario
falls back to using the completion text, while still sending token-id input.)

Example
-------
    python scripts/benchmarks/math500_token_vs_text.py \
        --base-url http://127.0.0.1:8090 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --num-samples 100 \
        --concurrency 16 \
        --max-tokens 2048 \
        --temperature 0.0 \
        --scenarios text token \
        --output math500_results.json

Dataset: by default loads `HuggingFaceH4/MATH-500` (test split) via the
`datasets` library. Alternatively pass `--data-file` pointing at a JSONL file
with `problem` and `answer` fields.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("math500_bench")


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step. "
    "Put your final answer inside \\boxed{}."
)


def build_messages(problem: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]


# --------------------------------------------------------------------------- #
# Answer extraction & math equivalence (adapted from the Hendrycks MATH eval)
# --------------------------------------------------------------------------- #

def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the substring of the last \\boxed{...} (including the wrapper)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    depth = 0
    right_brace_idx = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        return s
    left_brace = "\\boxed{"
    if s.startswith(left_brace) and s.endswith("}"):
        return s[len(left_brace) : -1]
    left = "\\boxed "
    if s.startswith(left):
        return s[len(left) :]
    # \fbox variants
    if s.startswith("\\fbox{") and s.endswith("}"):
        return s[len("\\fbox{") : -1]
    return s


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer string from model output."""
    if text is None:
        return None
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    # Fallback: "answer is X" / "= X" at the very end
    m = re.search(r"(?:answer|Answer)\s*(?:is|:|=)\s*(.+)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    if len(substr) < 2:
                        return string
                    a, b = substr[0], substr[1]
                    if b != "{":
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b + substr[2:]
                except Exception:
                    return string
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_i = int(a)
        b_i = int(b)
        return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def normalize_answer(string: str) -> str:
    if string is None:
        return ""
    string = string.strip()
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if string.startswith("."):
        string = "0" + string
    if len(string.split("=")) == 2:
        # keep RHS of "x = ..."
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    # strip surrounding text{...}
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    string = string.replace("\\mbox", "").replace("\\text", "")
    string = string.rstrip(".")
    return string


def is_equiv(a: Optional[str], b: Optional[str], use_sympy: bool = True) -> bool:
    if a is None or b is None:
        return False
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    if use_sympy:
        try:
            from sympy import simplify
            from sympy.parsing.latex import parse_latex

            ea, eb = parse_latex(a), parse_latex(b)
            if simplify(ea - eb) == 0:
                return True
        except Exception:
            pass
    return False


def grade(prediction_text: str, ground_truth: str, use_sympy: bool) -> bool:
    pred = extract_answer(prediction_text)
    gt = extract_answer(ground_truth) or ground_truth
    return is_equiv(pred, gt, use_sympy=use_sympy)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_dataset_records(
    data_file: Optional[str],
    hf_name: str,
    split: str,
    num_samples: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if data_file:
        logger.info("Loading dataset from local file: %s", data_file)
        with open(data_file, "r", encoding="utf-8") as f:
            if data_file.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.load(f)
                records = data if isinstance(data, list) else data.get("data", [])
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                "The `datasets` package is required to load MATH-500 from the Hub.\n"
                "Install it with `pip install datasets`, or pass --data-file with a "
                "local JSONL containing `problem` and `answer` fields."
            )
        logger.info("Loading dataset %s [%s] from the Hugging Face Hub", hf_name, split)
        ds = load_dataset(hf_name, split=split)
        records = [dict(row) for row in ds]

    # normalize column names
    normalized: List[Dict[str, Any]] = []
    for r in records:
        problem = r.get("problem") or r.get("question") or r.get("prompt")
        answer = r.get("answer")
        if answer is None:
            answer = r.get("solution") or r.get("gt") or r.get("ground_truth")
        if problem is None or answer is None:
            continue
        normalized.append(
            {
                "problem": problem,
                "answer": str(answer),
                "solution": r.get("solution"),
                "level": r.get("level"),
                "subject": r.get("subject") or r.get("type"),
                "unique_id": r.get("unique_id"),
            }
        )

    if num_samples is not None:
        normalized = normalized[:num_samples]
    logger.info("Loaded %d problems", len(normalized))
    return normalized


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    index: int
    scenario: str
    ok: bool
    correct: bool = False
    gen_latency: float = 0.0          # seconds for the core generation call
    tokenize_latency: float = 0.0     # seconds spent tokenizing input (token scenario)
    detokenize_latency: float = 0.0   # seconds spent detokenizing output (token scenario)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_ids_returned: bool = False
    prediction: Optional[str] = None
    error: Optional[str] = None


async def _post_json(session, url: str, payload: dict, headers: dict, timeout: float) -> dict:
    import aiohttp

    async with session.post(
        url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        return json.loads(text)


# --------------------------------------------------------------------------- #
# Scenario runners
# --------------------------------------------------------------------------- #

class Benchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if args.api_key:
            self.headers["Authorization"] = f"Bearer {args.api_key}"
        self._local_tokenizer = None
        if args.local_tokenizer:
            from transformers import AutoTokenizer

            tok_name = args.tokenizer or args.model
            logger.info("Loading local tokenizer: %s", tok_name)
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                tok_name, trust_remote_code=True
            )

    # ---- text scenario ---------------------------------------------------- #
    async def run_text(self, session, index: int, record: Dict[str, Any]) -> RequestResult:
        messages = build_messages(record["problem"], self.args.system_prompt)
        payload = {
            "model": self.args.model,
            "messages": messages,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "seed": self.args.seed,
            "stream": False,
        }
        try:
            t0 = time.perf_counter()
            resp = await _post_json(
                session, f"{self.base_url}/v1/chat/completions", payload,
                self.headers, self.args.request_timeout,
            )
            gen_latency = time.perf_counter() - t0
            choice = resp["choices"][0]
            text = choice.get("message", {}).get("content") or ""
            usage = resp.get("usage", {}) or {}
            res = RequestResult(
                index=index, scenario="text", ok=True, gen_latency=gen_latency,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                prediction=text,
            )
            res.correct = grade(text, record["answer"], self.args.use_sympy)
            return res
        except Exception as e:  # noqa: BLE001
            return RequestResult(index=index, scenario="text", ok=False, error=str(e))

    # ---- token scenario --------------------------------------------------- #
    async def _tokenize(self, session, messages: List[Dict[str, str]]) -> Tuple[List[int], float]:
        if self._local_tokenizer is not None:
            t0 = time.perf_counter()
            ids = self._local_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            return list(ids), time.perf_counter() - t0
        payload = {
            "model": self.args.model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        t0 = time.perf_counter()
        resp = await _post_json(
            session, f"{self.base_url}/tokenize", payload, self.headers,
            self.args.request_timeout,
        )
        return list(resp["tokens"]), time.perf_counter() - t0

    async def _detokenize(self, session, token_ids: List[int]) -> Tuple[str, float]:
        if self._local_tokenizer is not None:
            t0 = time.perf_counter()
            text = self._local_tokenizer.decode(token_ids, skip_special_tokens=True)
            return text, time.perf_counter() - t0
        payload = {"model": self.args.model, "tokens": token_ids}
        t0 = time.perf_counter()
        resp = await _post_json(
            session, f"{self.base_url}/detokenize", payload, self.headers,
            self.args.request_timeout,
        )
        return resp.get("prompt", ""), time.perf_counter() - t0

    async def run_token(self, session, index: int, record: Dict[str, Any]) -> RequestResult:
        messages = build_messages(record["problem"], self.args.system_prompt)
        try:
            prompt_ids, tok_lat = await self._tokenize(session, messages)
            payload = {
                "model": self.args.model,
                "prompt": prompt_ids,
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "seed": self.args.seed,
                "stream": False,
                "return_token_ids": True,
            }
            t0 = time.perf_counter()
            resp = await _post_json(
                session, f"{self.base_url}/v1/completions", payload,
                self.headers, self.args.request_timeout,
            )
            gen_latency = time.perf_counter() - t0
            choice = resp["choices"][0]
            usage = resp.get("usage", {}) or {}

            out_token_ids = choice.get("token_ids")
            detok_lat = 0.0
            token_ids_returned = bool(out_token_ids)
            if token_ids_returned:
                text, detok_lat = await self._detokenize(session, out_token_ids)
            else:
                # return_token_ids unsupported -> fall back to the text field
                text = choice.get("text") or ""

            res = RequestResult(
                index=index, scenario="token", ok=True, gen_latency=gen_latency,
                tokenize_latency=tok_lat, detokenize_latency=detok_lat,
                prompt_tokens=int(usage.get("prompt_tokens", len(prompt_ids))),
                completion_tokens=int(
                    usage.get("completion_tokens", len(out_token_ids or []))
                ),
                token_ids_returned=token_ids_returned,
                prediction=text,
            )
            res.correct = grade(text, record["answer"], self.args.use_sympy)
            return res
        except Exception as e:  # noqa: BLE001
            return RequestResult(index=index, scenario="token", ok=False, error=str(e))

    # ---- driver ----------------------------------------------------------- #
    async def run_scenario(
        self, scenario: str, records: List[Dict[str, Any]]
    ) -> Tuple[List[RequestResult], float]:
        import aiohttp

        runner = self.run_text if scenario == "text" else self.run_token
        sem = asyncio.Semaphore(self.args.concurrency)
        results: List[RequestResult] = []

        connector = aiohttp.TCPConnector(limit=self.args.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:

            async def _wrapped(i: int, rec: Dict[str, Any]) -> RequestResult:
                async with sem:
                    return await runner(session, i, rec)

            tasks = [
                asyncio.create_task(_wrapped(i, rec)) for i, rec in enumerate(records)
            ]
            wall_start = time.perf_counter()
            done = 0
            for fut in asyncio.as_completed(tasks):
                res = await fut
                results.append(res)
                done += 1
                if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                    logger.info("[%s] %d/%d done", scenario, done, len(tasks))
            wall_time = time.perf_counter() - wall_start

        results.sort(key=lambda r: r.index)
        return results, wall_time


# --------------------------------------------------------------------------- #
# Aggregation & reporting
# --------------------------------------------------------------------------- #

def _classify_error(err: str) -> str:
    """Collapse error strings into a few buckets so failures are diagnosable."""
    if not err:
        return "unknown"
    low = err.lower()
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "connect" in low or "connection" in low or "refused" in low or "reset" in low:
        return "connection error"
    m = re.search(r"http (\d{3})", low)
    if m:
        # keep a short tail of the server message for HTTP errors
        tail = err.split(":", 1)[1].strip() if ":" in err else ""
        return f"HTTP {m.group(1)}: {tail[:160]}"
    if "token_ids" in low or "keyerror" in low or "'choices'" in low:
        return f"response parsing: {err[:160]}"
    return err[:160]


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def summarize(scenario: str, results: List[RequestResult], wall_time: float) -> Dict[str, Any]:
    ok = [r for r in results if r.ok]
    total = len(results)
    n_ok = len(ok)
    correct = sum(1 for r in ok if r.correct)
    gen_lat = [r.gen_latency for r in ok]
    comp_tokens = [r.completion_tokens for r in ok]
    total_comp_tokens = sum(comp_tokens)
    tok_lat = [r.tokenize_latency for r in ok if r.tokenize_latency > 0]
    detok_lat = [r.detokenize_latency for r in ok if r.detokenize_latency > 0]
    token_ids_returned = sum(1 for r in ok if r.token_ids_returned)

    from collections import Counter

    error_counts = Counter(
        _classify_error(r.error) for r in results if not r.ok and r.error
    )

    return {
        "errors": dict(error_counts.most_common()),
        "scenario": scenario,
        "total": total,
        "succeeded": n_ok,
        "failed": total - n_ok,
        "accuracy": (correct / n_ok) if n_ok else 0.0,
        "correct": correct,
        "wall_time_s": wall_time,
        "requests_per_s": (n_ok / wall_time) if wall_time > 0 else 0.0,
        "gen_latency_mean_s": statistics.mean(gen_lat) if gen_lat else 0.0,
        "gen_latency_median_s": statistics.median(gen_lat) if gen_lat else 0.0,
        "gen_latency_p90_s": _pct(gen_lat, 90),
        "gen_latency_p99_s": _pct(gen_lat, 99),
        "total_completion_tokens": total_comp_tokens,
        "mean_completion_tokens": (total_comp_tokens / n_ok) if n_ok else 0.0,
        "output_throughput_tok_per_s": (total_comp_tokens / wall_time) if wall_time > 0 else 0.0,
        "tokenize_latency_mean_ms": (statistics.mean(tok_lat) * 1000) if tok_lat else 0.0,
        "detokenize_latency_mean_ms": (statistics.mean(detok_lat) * 1000) if detok_lat else 0.0,
        "token_ids_returned": token_ids_returned,
    }


def print_report(summaries: Dict[str, Dict[str, Any]]) -> None:
    rows = [
        ("Accuracy", "accuracy", "{:.2%}"),
        ("Correct / OK", None, None),
        ("Succeeded / Total", None, None),
        ("Wall time (s)", "wall_time_s", "{:.2f}"),
        ("Requests/s", "requests_per_s", "{:.2f}"),
        ("Gen latency mean (s)", "gen_latency_mean_s", "{:.3f}"),
        ("Gen latency median (s)", "gen_latency_median_s", "{:.3f}"),
        ("Gen latency p90 (s)", "gen_latency_p90_s", "{:.3f}"),
        ("Gen latency p99 (s)", "gen_latency_p99_s", "{:.3f}"),
        ("Mean output tokens", "mean_completion_tokens", "{:.1f}"),
        ("Output throughput (tok/s)", "output_throughput_tok_per_s", "{:.1f}"),
        ("Tokenize overhead (ms)", "tokenize_latency_mean_ms", "{:.2f}"),
        ("Detokenize overhead (ms)", "detokenize_latency_mean_ms", "{:.2f}"),
    ]
    scenarios = list(summaries.keys())
    col_w = 28
    header = "Metric".ljust(col_w) + "".join(s.ljust(16) for s in scenarios)
    print("\n" + "=" * len(header))
    print("MATH-500: text I/O  vs  token-id I/O")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key, fmt in rows:
        line = label.ljust(col_w)
        for s in scenarios:
            summ = summaries[s]
            if key is None:
                if label.startswith("Correct"):
                    val = f"{summ['correct']}/{summ['succeeded']}"
                else:
                    val = f"{summ['succeeded']}/{summ['total']}"
            else:
                val = fmt.format(summ[key])
            line += val.ljust(16)
        print(line)
    print("=" * len(header))
    for s in scenarios:
        summ = summaries[s]
        if summ.get("failed"):
            print(f"  [warn] scenario '{s}' had {summ['failed']} failed request(s):")
            for msg, cnt in summ.get("errors", {}).items():
                print(f"           {cnt:>4}x  {msg}")
            if summ["succeeded"] < max(1, summ["total"] // 2):
                print(
                    f"  [warn] scenario '{s}': accuracy is computed over only "
                    f"{summ['succeeded']} successful sample(s) and is NOT reliable. "
                    "Fix the failures above first."
                )
        if s == "token" and summ.get("token_ids_returned", 0) < summ.get("succeeded", 0):
            print(
                f"  [warn] scenario 'token': only {summ['token_ids_returned']}/"
                f"{summ['succeeded']} responses included token_ids "
                "(server may predate vLLM 0.10.2 `return_token_ids`; "
                "fell back to text output)."
            )
    print()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare MATH-500 accuracy & performance: text I/O vs token-id I/O on a vLLM endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default="http://127.0.0.1:8000",
                   help="Base URL of the vLLM server or vLLM Router.")
    p.add_argument("--model", required=True, help="Model name as served by vLLM.")
    p.add_argument("--api-key", default=None, help="Bearer token, if the endpoint requires auth.")

    p.add_argument("--scenarios", nargs="+", choices=["text", "token"],
                   default=["text", "token"], help="Which scenarios to run.")

    # dataset
    p.add_argument("--data-file", default=None,
                   help="Local JSONL/JSON with `problem` and `answer` fields. "
                        "If omitted, downloads from the Hugging Face Hub.")
    p.add_argument("--hf-name", default="HuggingFaceH4/MATH-500",
                   help="Hugging Face dataset name (used when --data-file is unset).")
    p.add_argument("--split", default="test", help="Dataset split.")
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limit number of problems (default: all).")

    # sampling
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)

    # execution
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max in-flight requests.")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="Per-request timeout in seconds.")
    p.add_argument("--local-tokenizer", action="store_true",
                   help="Tokenize/detokenize locally with transformers instead of "
                        "the server /tokenize & /detokenize endpoints.")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer name for --local-tokenizer (defaults to --model).")
    p.add_argument("--no-sympy", dest="use_sympy", action="store_false",
                   help="Disable sympy-based equivalence fallback (string match only).")

    # output
    p.add_argument("--output", default=None,
                   help="Optional path to write full JSON results.")
    p.add_argument("--save-predictions", action="store_true",
                   help="Include per-sample predictions in the JSON output.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        import aiohttp  # noqa: F401
    except ImportError:
        logger.error("aiohttp is required. Install with `pip install aiohttp`.")
        return 1

    records = load_dataset_records(
        args.data_file, args.hf_name, args.split, args.num_samples
    )
    if not records:
        logger.error("No problems loaded; aborting.")
        return 1

    bench = Benchmark(args)

    summaries: Dict[str, Dict[str, Any]] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for scenario in args.scenarios:
        logger.info("Running scenario '%s' over %d problems (concurrency=%d) ...",
                    scenario, len(records), args.concurrency)
        results, wall_time = asyncio.run(bench.run_scenario(scenario, records))
        summaries[scenario] = summarize(scenario, results, wall_time)
        if args.output:
            all_results[scenario] = [
                {
                    k: v for k, v in vars(r).items()
                    if args.save_predictions or k != "prediction"
                }
                for r in results
            ]

    print_report(summaries)

    if "text" in summaries and "token" in summaries:
        t, k = summaries["text"], summaries["token"]
        acc_delta = (k["accuracy"] - t["accuracy"]) * 100
        if t["gen_latency_mean_s"] > 0:
            speedup = t["gen_latency_mean_s"] / k["gen_latency_mean_s"] if k["gen_latency_mean_s"] else 0.0
        else:
            speedup = 0.0
        print("Comparison (token vs text):")
        print(f"  accuracy delta : {acc_delta:+.2f} percentage points")
        print(f"  gen-latency speedup (text/token mean): {speedup:.3f}x")
        if k["output_throughput_tok_per_s"] and t["output_throughput_tok_per_s"]:
            print(
                f"  throughput ratio (token/text): "
                f"{k['output_throughput_tok_per_s'] / t['output_throughput_tok_per_s']:.3f}x"
            )
        print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "base_url": args.base_url,
                        "model": args.model,
                        "scenarios": args.scenarios,
                        "num_samples": len(records),
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "concurrency": args.concurrency,
                    },
                    "summaries": summaries,
                    "results": all_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info("Wrote results to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
