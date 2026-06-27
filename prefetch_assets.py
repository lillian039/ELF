#!/usr/bin/env python
"""Robustly pre-fetch HF assets for the ELF conditional eval through a flaky
proxy. Each attempt is bounded by a SIGALRM watchdog; downloads resume from
.incomplete via HTTP range, so repeated bounded attempts accumulate bytes.

Run with the proxy env set:
    http_proxy=http://127.0.0.1:7899 https_proxy=http://127.0.0.1:7899 \
    no_proxy=127.0.0.1,localhost python prefetch_assets.py
"""
import os
import signal
import sys
import time
import traceback

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "20"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

MODELS = [
    "embedded-language-flows/ELF-B-de-en-torch",
    "embedded-language-flows/ELF-B-xsum-torch",
]
DATASETS = [
    "embedded-language-flows/wmt14_de-en_validation_t5",
    "embedded-language-flows/xsum_validation_t5",
]


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def robust(fn, what, attempts=100000, per_attempt=180):
    for i in range(1, attempts + 1):
        signal.alarm(per_attempt)
        try:
            out = fn()
            signal.alarm(0)
            print(f"[OK] {what} (attempt {i})", flush=True)
            return out
        except _Timeout:
            signal.alarm(0)
            print(f"[try {i}] {what}: watchdog abort after {per_attempt}s; resuming", flush=True)
        except Exception as e:  # noqa: BLE001
            signal.alarm(0)
            msg = str(e).replace("\n", " ")[:140]
            # Proxy-down (connection refused) errors fast; back off a bit longer
            # so we don't burn attempts during an extended down-window.
            refused = "refused" in msg.lower() or "ConnectionError" in type(e).__name__
            print(f"[try {i}] {what}: {type(e).__name__}: {msg}", flush=True)
            time.sleep(10 if refused else 4)
    raise RuntimeError(f"giving up on {what}")


def fetch_model(repo):
    from huggingface_hub import snapshot_download
    robust(lambda: snapshot_download(repo_id=repo, repo_type="model",
                                     max_workers=1, etag_timeout=20),
           f"model {repo}")


def fetch_dataset(path):
    from utils.data_utils import load_dataset_split
    ds = robust(lambda: load_dataset_split(path), f"dataset {path}")
    print(f"     {path}: {len(ds)} rows, cols={list(ds.column_names)}", flush=True)


def main():
    print("https_proxy:", os.environ.get("https_proxy"), flush=True)
    for m in MODELS:
        fetch_model(m)
    for d in DATASETS:
        fetch_dataset(d)
    print("ALL_PREFETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
