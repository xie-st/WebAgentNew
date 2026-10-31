# Installation Guide

## Standard install

```bash
pip install effgen                    # base install (all core features)
pip install effgen[dev]                # dev tools (pytest, ruff, mypy, pytest-forked)
pip install effgen[rag]                # sentence-transformers + faiss-cpu
pip install effgen[vector-db]          # faiss + chromadb + qdrant
pip install effgen[vllm]               # vLLM backend (NVIDIA GPUs)
pip install effgen[mlx]                # MLX backend (Apple Silicon)
```

> **On an NVIDIA GPU box?** The default PyTorch wheel on PyPI may be built for a
> newer CUDA runtime than your driver supports, which silently disables the GPU
> (`torch.cuda.is_available()` is `False` even though `nvidia-smi` lists your
> GPUs). Pick a torch wheel that matches your driver — see
> [GPU / CUDA compatibility](#gpu--cuda-compatibility) below. The bundled
> `./install.sh` does this automatically.

### Use-case extras

Install only the slice you need instead of the everything-extra `[all]`. The
base install already leaves the heavy optional stacks (vLLM, vector DBs, OCR,
OpenCV, spaCy/Stanza) out, so these pull just their use case on top:

```bash
pip install effgen[api]                # hosted-inference SDKs (groq/together/fireworks/replicate/cerebras/hf)
pip install effgen[local]              # local-model quantization + GGUF backends (bitsandbytes, llama-cpp)
pip install effgen[server]             # OIDC/JWT auth + Prometheus metrics for the API server
pip install effgen[tools-web]          # web-search / browsing tools
pip install effgen[tools-docs]         # document parsing (PDF / DOCX / XLSX)
```

> `[all]` is the **CI / everything** extra — it installs every optional
> dependency so the full test suite runs. It is large and slow to resolve;
> prefer a use-case extra above unless you specifically need everything. See
> [Installing the `[all]` extra](#installing-the-all-extra) for the constraints
> lock it requires.

### Installing the `[all]` extra

`[all]` pulls vLLM plus every provider SDK and the full Google client stack.
Under the `protobuf>=5.29.5` security floor this dependency graph is too deep
for pip to resolve on its own (`resolution-too-deep`). Install it with the
committed constraints lock, which pins a single consistent, CVE-safe solution:

```bash
# from a clone of the repo:
pip install -e ".[all]" -c requirements-all-lock.txt

# or against the published package:
pip install "effgen[all]" -c https://raw.githubusercontent.com/ctrl-gaurav/effGen/main/requirements-all-lock.txt
```

Regenerate the lock after changing dependencies (requires
[uv](https://github.com/astral-sh/uv)):

```bash
uv pip compile pyproject.toml --extra all --output-file requirements-all-lock.txt
```

> **Why isn't `flash-attn` in `[all]`?**
> `flash-attn`'s own `setup.py` imports `torch` during wheel-metadata generation,
> but pip's isolated build environment does not have `torch` available at that
> moment. This is a well-known upstream bug in `flash-attn` — any package that
> lists `flash-attn` as a dependency will cause `pip install` to fail. To keep
> `pip install effgen[all]` working for everyone, `flash-attn` is kept out of
> `[all]` and installed separately (see below).

## GPU / CUDA compatibility

PyTorch wheels are built against a specific CUDA runtime (`torch 2.x+cu124`,
`+cu130`, …). An NVIDIA driver is **forward-compatible only**: a driver that
reports `CUDA Version: 12.4` (`nvidia-smi`) can run torch built for CUDA ≤ 12.x,
but **not** a torch built for CUDA 13. Installing a CUDA-13 wheel on a CUDA-12.4
driver leaves you with `torch.cuda.is_available() == False` and everything runs
(slowly) on the CPU.

Check your driver's maximum CUDA version first:

```bash
nvidia-smi    # top-right: "CUDA Version: 12.4"
```

Then install a matching torch wheel **before** installing effgen (or let
`./install.sh` detect the driver and do it for you):

| Your environment | torch index URL | Install command |
|---|---|---|
| **CPU only** (no NVIDIA GPU) | `cpu` | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| **CUDA 12.1–12.7 driver** (e.g. 12.4) | `cu124` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124` |
| **CUDA 12.8+ driver** | `cu128` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu128` |
| **CUDA 13.x driver** | `cu130` | `pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu130` |
| **vLLM** (NVIDIA GPU) | matches vLLM's pinned torch | see [Installing vLLM](#installing-vllm-optional) |

```bash
# Example: a host with a CUDA 12.4 driver
pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124
pip install effgen
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

### Keeping your GPU torch when installing extras (`-c constraints-cu1xx.txt`)

Installing the wrong wheel is only half the trap. The *other* half: once you have
a working GPU torch, a later `pip install -e ".[all]"` (or any extra that pulls a
torch-pinning dependency such as vLLM) can let pip's resolver **silently upgrade
torch back to a newer-CUDA wheel** — your GPU goes dark again with no install-time
warning. To prevent that, the repo ships a small constraints file per CUDA line
that pins torch/torchvision/torchaudio to a driver-compatible build. Apply it with
`-c` and an extras install can never move torch:

```bash
# CUDA 12.4 box: install the GPU torch, then install extras under the constraint
pip install "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[local]" -c constraints-cu124.txt
python -c "import torch; print(torch.cuda.is_available())"   # still True
```

| Your driver (`nvidia-smi` CUDA) | Constraints file |
|---|---|
| **CUDA 12.1–12.7** (e.g. 12.4) | `constraints-cu124.txt` |
| **CUDA 12.8+** | `constraints-cu128.txt` |
| **CUDA 13.x** | `constraints-cu130.txt` |
| **CPU only** | `constraints-cpu.txt` |

The constraints files carry their own `--extra-index-url`, so they also work on a
fresh env (no pre-installed torch). For the everything-extra `[all]` — which pins
vLLM to one exact torch — use the matching vLLM/torch pair from
[Installing vLLM](#installing-vllm-optional) rather than a generic constraint, or
the committed `requirements-all-lock.txt`. `scripts/check_install_constraints.sh`
verifies the flow end to end (an extras install over a pinned torch leaves torch
and `torch.cuda.is_available()` untouched).

effGen detects a torch-CUDA / driver mismatch at runtime: when it sees physical
NVIDIA GPUs but `torch.cuda` cannot use them, it prints **one** warning naming the
torch CUDA build vs the driver's CUDA version and pointing back here, instead of
silently running on the CPU. Set `EFFGEN_NO_GPU_WARN=1` to silence it if you are
deliberately running CPU-only on a GPU box.

## Installing flash-attn (optional, NVIDIA GPUs only)

**Step 1 — install effgen first (gets torch and everything else):**

```bash
pip install -e ".[all]" -c requirements-all-lock.txt
```

**Step 2 — install flash-attn with build isolation disabled:**

```bash
pip install flash-attn --no-build-isolation
```

`--no-build-isolation` lets flash-attn's setup.py reuse the torch already
installed in your environment, bypassing the bug.

### Requirements for flash-attn

- NVIDIA GPU with compute capability ≥ 7.5 (Turing or newer)
- CUDA toolkit (`nvcc`) matching your torch CUDA version
- GCC 9+ and a few GB of RAM for compilation (can take 10–30 minutes)

If the build fails, prefer the official pre-built wheel from
<https://github.com/Dao-AILab/flash-attention/releases> matching your
exact `(python, torch, cuda)` triple.

## Installing vLLM (optional)

vLLM gives much higher throughput than the Transformers backend, but it is the
trickiest optional stack to install because **each vLLM release pins one exact
torch version**, and that torch build determines the CUDA runtime. The latest
vLLM pins a CUDA-13 torch, so a plain `pip install effgen[vllm]` on a CUDA-12
driver will pull `torch ...+cu130`, which then reports
`torch.cuda.is_available() == False` and fails to import vLLM's compiled
extension (`libcudart.so.13: cannot open shared object file`).

Pick the vLLM release whose pinned torch matches your driver:

| Driver (`nvidia-smi` CUDA) | torch build | Known-good vLLM | Install |
|---|---|---|---|
| **CUDA 12.1–12.7** (e.g. 12.4) | `torch==2.6.0+cu124` | `vllm==0.8.5.post1` | see below |
| **CUDA 12.8+** | `torch==2.7.1+cu128` | `vllm==0.10.1.1` | swap the version + `cu128` index |
| **CUDA 13.x** (driver ≥ 580) | `torch>=2.11+cu130` | latest (`pip install effgen[vllm]`) | latest works directly |

```bash
# CUDA 12.4 box: install a CUDA-12.4 torch first, then a matching vLLM
pip install effgen
pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124
pip install "vllm==0.8.5.post1"
python -c "import torch; from vllm import LLM; print('vLLM ready:', torch.cuda.is_available())"
```

`pip check` must report no conflicts after this — if vLLM and torch disagree on
versions, you installed a mismatched pair (re-check the table above).

Two effGen safety nets make a mismatch easy to diagnose rather than mysterious:

- `VLLMEngine.load()` reports an ABI/CUDA import failure (e.g. a missing
  `libcudart`) as exactly that, instead of "vLLM is not installed".
- `load_model(..., engine="auto-fast")` uses vLLM only when it imports cleanly
  and a GPU is usable, and **transparently falls back to the Transformers
  backend** otherwise — so opting into speed never hard-fails your program.

## Supported Python versions

`effgen` officially supports Python **3.11, 3.12, 3.13, 3.14**.

**Python 3.14 needs a lock file for `effgen[all]`.** The base install, the CLI,
the `dev` extra and every provider extra install normally there:

```bash
pip install effgen              # works on 3.14
pip install "effgen[openai]"    # any extra by name works on 3.14
```

`effgen[all]` is the exception, and the reason is pip's resolver rather than any
one package. `all` pulls `vllm` on a wide range (`>=0.2.7,<1.0`). Recent `vllm`
releases carry 3.14 wheels for themselves and their compiled dependencies, but
when pip backtracks over an unrelated conflict it walks that range backwards
until it reaches a release pinned to `numba==0.61`, which predates 3.14 and
whose source build refuses to run there. Constraining `vllm` upward instead ends
in `resolution-too-deep`.

Pin the whole set instead:

```bash
pip install -r requirements-all-py314-lock.txt
pip install --no-deps effgen
```

That is the same answer `requirements-all-lock.txt` gives on 3.11–3.13; the two
are not interchangeable, because the 3.11 lock pins `torch==2.2.1` and
`vllm==0.4.1`, neither of which has a 3.14 wheel.

3.10 was dropped for 1.0.0. It reached the point where holding it back cost
more than it bought: several `pyupgrade` rewrites and a few runtime branches
existed only to keep 3.10 working, and `tomllib`, `asyncio.timeout`,
`datetime.UTC` and the `TimeoutError` unification are all stdlib from 3.11.

## Verifying your install

```bash
python -c "import effgen; print(effgen.__version__)"
python -c "from effgen import Agent; print(Agent)"
```
