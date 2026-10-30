# Secure Code Execution — CodeExecutor & Sandbox

## Threat Model

When an agent receives an attacker-controlled prompt, it may produce arbitrary
Python/shell code and pass it to `CodeExecutor`. Without isolation, this code
runs directly on the host with the privileges of the effGen process.

**Attack surface:** Prompt injection → agent emits `os.system("rm -rf /")` or
exfiltrates secrets via network calls.

**Mitigations provided by the sandbox layer:**

| Control | DockerSandbox | SubprocessSandbox |
|---|---|---|
| Network isolation | ✅ `--network=none` | ✅ `unshare --map-root-user --net` (Linux, no privileges) |
| Write confinement | ✅ `--read-only` | ✅ every mount remounted read-only except the working directory, locked by a nested `unshare --mount` |
| Read confinement | ✅ `--read-only` (no host mount) | ❌ the host filesystem stays readable |
| Memory cap | ✅ `--memory=256m` | ⚠️ `ulimit -v` (advisory) |
| CPU cap | ✅ `--cpus=1` | ⚠️ `ulimit -t` (advisory) |
| PID limit | ✅ `--pids-limit=100` | ⚠️ `ulimit -u 256` |
| Capability drop | ✅ `--cap-drop=ALL` | ⚠️ unprivileged user namespace (no host root) |
| Privilege escalation | ✅ `--no-new-privileges` | ⚠️ user-namespace UID mapping |

> The SubprocessSandbox network and write isolation use **unprivileged user
> namespaces** — they do **not** require `CAP_SYS_ADMIN` or root. If user
> namespaces are disabled on the host, the sandbox degrades to `ulimit`-only
> mode and logs a warning naming exactly which protections are inactive. Every
> `SandboxResult` reports what the run actually enforced in
> `filesystem_confined` and `writable_root`, so a caller never has to assume.

---

## Sandbox Backends

### DockerSandbox (Default when Docker available)

Runs code in a container built from `effgen/sandbox:python-3.11` (or the
stock `python:3.11-slim` if the custom image is not present locally).

```text
docker run --rm
  --read-only
  --tmpfs /tmp:size=64m,noexec
  -v <tmpdir>:/workspace:ro
  -w /workspace
  --network none
  --cap-drop ALL
  --no-new-privileges
  --security-opt no-new-privileges:true
  --pids-limit 100
  --memory 256m
  --memory-swap 256m
  --cpus 1
  effgen/sandbox:python-3.11
  python3 script.py
```

Build the custom sandbox image:

```bash
docker build -f deploy/sandbox/Dockerfile.sandbox \
             -t effgen/sandbox:python-3.11 .
```

### SubprocessSandbox (Fallback)

Used when Docker daemon is unreachable. Provides **partial isolation** via an
unprivileged user namespace: the network is isolated and writes are confined to
one directory, but reads are not. Code is fed to the interpreter over **stdin**,
so no script file is written to the host filesystem.

The **scratch space** — the one writable directory — is `SandboxConfig.workdir`,
else the workspace directory (`EFFGEN_WORKSPACE`), else the calling process's own
directory. That is the same root the file and shell tools use, so executed code
can write the files the agent just created and nothing else.

On Linux with unprivileged user namespaces available:

```bash
unshare --map-root-user --mount --net bash -c '
  ulimit -v 262144 -t 10 -u 256
  mount --bind "$ROOT" "$ROOT"              # the scratch space, kept writable
  while IFS= read -r line; do               # every other mount goes read-only
    ...                                     # (skipping /proc and existing ro mounts)
    mount -o remount,bind,ro "$mountpoint"
  done < /proc/self/mountinfo
  mount -t tmpfs none /tmp                  # private temp directory
  mount -t tmpfs none /dev/shm              # shared memory keeps working
  cd "$ROOT"
  exec unshare --map-root-user --mount python3 -
'
```

- `--map-root-user` — root inside the namespace only; no host privileges.
- `--net`           — fresh network namespace with no interfaces → outbound
                      network blocked (DNS + connect fail). No `CAP_SYS_ADMIN`.
- `--mount`         — a private mount namespace in which every mount is
                      remounted read-only except the scratch space and the
                      private `/tmp` and `/dev/shm`.
- the **nested** `unshare` — hands the interpreter a fresh user and mount
                      namespace, which makes the kernel lock those mounts.
                      Without it, executed code (root in the outer namespace)
                      remounts them read-write in one line.

Availability is probed by running the real command over two throwaway
directories and requiring both halves of the contract: a write inside the
scratch space succeeds and a write outside it fails. Anything short of that
leaves confinement off — the result then reports `filesystem_confined=False`
and the previous behavior applies.

**Caveats:**
- **Reads are not confined, but the credential stores are masked.** Executed
  code can read every ordinary file the calling user can read. What it cannot
  read are the per-user credential stores: `~/.ssh`, `~/.aws`, `~/.gnupg`,
  `~/.kube`, `~/.docker`, `~/.azure`, `~/.config/gcloud`, the credential files
  beside them (`~/.netrc`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`),
  `/etc/shadow` and the mounted-secret directories. Inside the sandbox each is
  covered — a directory by an empty tmpfs, a file by `/dev/null` — so a read
  succeeds and returns nothing rather than failing in a way that confirms the
  path exists. This is a **deny-list**, reported as
  `SandboxResult.credential_reads_masked`, not read confinement. Use
  DockerSandbox when reads must be confined rather than masked.
- `/proc` stays writable, because the nested `unshare` writes
  `/proc/self/uid_map` — but it is the sandbox's *own* `/proc`. The run gets a
  private PID namespace (`--pid --fork --mount-proc`), so executed code sees
  only its own process (`ls /proc` shows one pid, against ~1,850 on the host)
  and cannot read another process's `cmdline` or `environ`. Reported as
  `SandboxResult.process_table_isolated`; a host that cannot create the
  namespace degrades to the shared process table and says so.
- A mount nested *inside* the scratch space becomes read-only; the scratch
  space's own mount is the one bound read-write.
- Requires unprivileged user namespaces (`kernel.unprivileged_userns_clone=1`
  / `user.max_user_namespaces>0`). When unavailable, the sandbox degrades to
  `ulimit`-only mode and logs a warning listing the inactive protections.
- Memory limit is advisory (`ulimit -v`), not enforced by cgroups.
- A loud `WARNING` is emitted at startup when this fallback is selected as the
  default.

To let a task write somewhere else, widen `EFFGEN_WORKSPACE` to a directory that
contains it — or, for fully trusted code, set `EFFGEN_SANDBOX_BACKEND=off`.

### OffSandbox (Explicit opt-out — UNSAFE)

Set `EFFGEN_SANDBOX_BACKEND=off` to disable sandboxing entirely. Code then runs
directly on the host with the privileges of the effGen process. A loud warning
is emitted on every execution, and this backend is **never** chosen by
auto-resolution. Use only for fully trusted code in controlled environments.

### FirecrackerSandbox (Stub — not implemented)

Interface stub for future Firecracker MicroVM integration. Raises
`NotImplementedError` if called. Full implementation requires:
- Firecracker binary + KVM device
- A minimal guest rootfs (Alpine initrd)
- `jailer` for additional isolation
- vsock/REST API for code injection

---

## Configuration

All options are environment-driven:

| Environment Variable | Default | Description |
|---|---|---|
| `EFFGEN_SANDBOX_BACKEND` | `auto` | `auto`, `docker`, `subprocess`, `firecracker`, `off` (unsafe) |
| `EFFGEN_SANDBOX_TIMEOUT` | `10` | Execution timeout in seconds |
| `EFFGEN_SANDBOX_MEMORY` | `256m` | Memory limit (Docker-style: `256m`, `1g`) |
| `EFFGEN_SANDBOX_IMAGE` | `effgen/sandbox:python-3.11` | Custom Docker image to use |

**Auto-selection:** When `EFFGEN_SANDBOX_BACKEND=auto` (default), effGen picks
DockerSandbox if the Docker daemon is reachable; otherwise SubprocessSandbox
with a warning.

Per-call overrides (passed to `CodeExecutor`):

```python
result = await executor._execute(
    code="print('hi')",
    language="python",
    timeout=30,
    memory_limit="512m",
    network_enabled=False,
)
```

---

## Quick Start

### Using CodeExecutor in an Agent

```python
from effgen import Agent, AgentConfig
from effgen.tools.builtin.code_executor import CodeExecutor

agent = Agent(
    config=AgentConfig(provider="openai", model="gpt-4o-mini", tools=[CodeExecutor()]),
)

response = agent.run(
    "Write and run a Python script that computes the first 10 prime numbers."
)
print(response.output)
```

### Direct Sandbox Usage

```python
import asyncio
from effgen.security.sandbox import get_sandbox, SandboxConfig

async def main():
    config = SandboxConfig(
        backend="auto",   # docker if available, else subprocess
        timeout=15,
        memory_limit="256m",
        network_enabled=False,
    )
    sandbox = await get_sandbox(config)
    result = await sandbox.run(
        code="print(sum(range(100)))",
        language="python",
        config=config,
    )
    print(f"stdout: {result.stdout}")
    print(f"backend: {result.backend_used}")
    print(f"writes confined: {result.filesystem_confined}")
    print(f"writable root: {result.writable_root}")

asyncio.run(main())
```

---

## Building the Sandbox Image

```bash
# Build
docker build \
  -f deploy/sandbox/Dockerfile.sandbox \
  -t effgen/sandbox:python-3.11 \
  .

# Smoke-test network isolation
docker run --rm --network=none effgen/sandbox:python-3.11 \
  python3 -c "
import urllib.request, sys
try:
    urllib.request.urlopen('https://example.com', timeout=3)
    print('FAIL: network was accessible')
    sys.exit(1)
except Exception as e:
    print(f'PASS: network blocked ({type(e).__name__})')
    sys.exit(0)
"
```

---

## Security Recommendations

1. **Always use DockerSandbox in production.** Install Docker and add the
   effGen service account to the `docker` group.
2. **Do not set `network_enabled=True`** unless the task explicitly requires
   network access and you have reviewed the code.
3. **Pin the sandbox image** to a specific digest in production environments:
   ```bash
   EFFGEN_SANDBOX_IMAGE=effgen/sandbox:python-3.11@sha256:<digest>
   ```
4. **Monitor container resource usage** via Docker stats or cAdvisor to detect
   abuse.
5. **Treat SubprocessSandbox as a development-only fallback.** It confines
   writes to the working directory and masks the named credential stores, but
   reads in general are not confined, so code that can be prompted into reading
   a secret from an *unlisted* location still can. Never rely on it for
   workloads handling untrusted code; check `filesystem_confined` and
   `credential_reads_masked` on the result if you need to know what a given run
   enforced.

---

## Limitations & Future Work

- **Firecracker integration** (`FirecrackerSandbox`) — planned for a future
  release; provides VM-level isolation with sub-second boot times.
- **Language support** — currently Python, JavaScript (Node), Bash/sh.
  Adding Rust, Go, Ruby planned.
- **seccomp profiles** — custom `--seccomp-profile` for DockerSandbox to
  further restrict syscalls (e.g., block `ptrace`, `mount`).
- **gVisor** (`runsc`) runtime — alternative to Firecracker for stronger
  syscall interposition without full VM overhead.
