import { Cpu } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';

export default function Hardware() {
  return (
    <DocPage
      subtitle="What effGen can see about the machine it is on, and how it decides what will fit."
      icon={<Cpu size={48} />}
    >
      <p>
        Everything on this page is about <Link to="/local-models">running a model locally</Link> —
        cloud providers need none of it. <code>effgen.hardware</code> answers what kind of machine
        this is and which backend suits it; <code>effgen.gpu</code> answers what the GPUs are doing
        right now, whether torch can actually use them, and which device a job should get.
      </p>

      <h2>What is on this machine?</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen models status`} />

      <Terminal
        command="effgen models status"
        output={`
Model & GPU Status
          GPU Status (physical, all processes)           
┏━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ GPU ┃ Name       ┃ Total   ┃ Used    ┃ Free    ┃ Util ┃
┡━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ 0   │ NVIDIA A40 │ 44.4 GB │ 27.6 GB │ 16.8 GB │ 100% │
│ 1   │ NVIDIA A40 │ 44.4 GB │ 0.0 GB  │ 44.4 GB │ 100% │
│ 2   │ NVIDIA A40 │ 44.4 GB │ 0.0 GB  │ 44.4 GB │ 100% │
│ 3   │ NVIDIA A40 │ 44.4 GB │ 44.4 GB │ 0.0 GB  │ 100% │
│ 4   │ NVIDIA A40 │ 44.4 GB │ 44.4 GB │ 0.0 GB  │ 100% │
│ 5   │ NVIDIA A40 │ 44.4 GB │ 0.0 GB  │ 44.4 GB │ 100% │
│ 6   │ NVIDIA A40 │ 44.4 GB │ 43.9 GB │ 0.5 GB  │ 100% │
│ 7   │ NVIDIA A40 │ 44.4 GB │ 44.0 GB │ 0.4 GB  │ 100% │
└─────┴────────────┴─────────┴─────────┴─────────┴──────┘

No models currently loaded in this process.

Capability profiles registered: 12`}
        caption="Captured on an eight-GPU A40 host. The GPU table is physical and covers every process on the machine, not just this one — which is what you need when a load fails and something else is holding the memory."
      />

      <p>
        <code>--json</code> gives the same thing to a script, which is how a scheduler or a
        pre-flight check should read it:
      </p>

      <Terminal
        command="effgen models status --json"
        output={`{
  "cuda_available": true,
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 27.587,
      "free_gb": 16.834,
      "utilization_pct": 100.0
    },
    {
      "index": 1,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 0.0,
      "free_gb": 44.421,
      "utilization_pct": 100.0
    },
    {
      "index": 2,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 0.0,
      "free_gb": 44.421,
      "utilization_pct": 100.0
    },
    {
      "index": 3,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 44.373,
      "free_gb": 0.048,
      "utilization_pct": 100.0
    },
    {
      "index": 4,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 44.373,
      "free_gb": 0.048,
      "utilization_pct": 100.0
    },
    {
      "index": 5,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 0.0,
      "free_gb": 44.421,
      "utilization_pct": 100.0
    },
    {
      "index": 6,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 43.883,
      "free_gb": 0.538,
      "utilization_pct": 100.0
    },
    {
      "index": 7,
      "name": "NVIDIA A40",
      "total_gb": 44.421,
      "used_gb": 44.034,
      "free_gb": 0.388,
      "utilization_pct": 100.0
    }
  ],
  "loaded_models": [],
  "capability_profiles": 12
}`}
        maxLines={20}
      />

      <h2>Is CUDA usable?</h2>

      <p>
        "Are there GPUs" and "can torch use them" are different questions, and the gap between them
        is the most common local-inference failure. <code>get_cuda_status()</code> answers both at
        once and says why when they disagree.
      </p>

      <CodeBlock filename="cuda.py" code={`from effgen.gpu import cuda_usable, get_cuda_status, physical_gpu_count

status = get_cuda_status()
print("usable        :", status.usable)
print("physical GPUs :", status.physical_gpus)
print("torch CUDA    :", status.torch_cuda)
print("driver CUDA   :", status.driver_cuda)
print("mismatch      :", status.mismatch)
print("message       :", status.message)
print()
print(cuda_usable(), physical_gpu_count())`} />

      <Terminal command="python cuda.py" output={`usable        : True
physical GPUs : 8
torch CUDA    : 13.0
driver CUDA   : 13.3
mismatch      : False
message       : None

True 8`} />

      <ParamTable
        nameLabel="CudaStatus field"
        params={[
          {
            name: 'usable',
            type: 'bool',
            description: 'Whether torch can use a GPU right now. This is the one to branch on.',
          },
          {
            name: 'physical_gpus',
            type: 'int',
            description: 'NVIDIA GPUs the machine has, whether or not torch can reach them.',
          },
          { name: 'torch_cuda', type: 'str | None', description: 'The CUDA version PyTorch was built against.' },
          { name: 'driver_cuda', type: 'str | None', description: 'The highest CUDA runtime the installed driver supports.' },
          {
            name: 'mismatch',
            type: 'bool',
            description: 'True when GPUs exist but torch cannot use them — usually a driver older than the build.',
          },
          { name: 'message', type: 'str | None', description: 'What is wrong, in a sentence, when something is.' },
          { name: 'torch_installed', type: 'bool', default: 'True', description: 'False when torch is not importable at all.' },
        ]}
        caption={
          <>
            <code>cuda_usable()</code>, <code>physical_gpu_count()</code>,{' '}
            <code>torch_cuda_version()</code> and <code>driver_cuda_version()</code> are the
            individual answers. <code>warn_cuda_mismatch_once()</code> emits one process-wide warning
            when GPUs exist and torch cannot use them, so a silent fall back to CPU is not silent.
          </>
        }
      />

      <Callout type="warning" title="A driver older than the torch build">
        <p>
          <code>mismatch=True</code> with a <code>driver_cuda</code> below <code>torch_cuda</code>{' '}
          means the GPUs are there and unusable. Nothing effGen does fixes it — install a torch build
          matching the driver, or update the driver. Until then local inference runs on CPU, which is
          why the warning exists.
        </p>
      </Callout>

      <h2>Which backend suits this machine?</h2>

      <CodeBlock filename="platform.py" code={`from effgen.hardware import (
    HardwarePlatform,
    get_best_local_backend,
    is_apple_silicon,
    is_cuda_available,
    is_mlx_available,
    platform,
)

print("platform      :", platform.detect_platform())
print("best backend  :", get_best_local_backend())
print("apple silicon :", is_apple_silicon())
print("cuda          :", is_cuda_available())
print("mlx           :", is_mlx_available())
print("known         :", [p.value for p in HardwarePlatform])`} />

      <Terminal command="python platform.py" output={`platform      : HardwarePlatform.CUDA
best backend  : vllm
apple silicon : False
cuda          : True
mlx           : False
known         : ['apple_silicon', 'cuda', 'cpu']`} />

      <ApiTable
        headers={['Platform', 'Best local backend', 'Notes']}
        rows={[
          [
            <code>cuda</code>,
            <code>vllm</code>,
            <>
              Falls back to <code>transformers</code> when vLLM is not installed. This is the fast
              path.
            </>,
          ],
          [
            <code>apple_silicon</code>,
            <code>mlx</code>,
            <>
              When MLX is installed. <code>get_unified_memory_gb()</code> is the memory figure that
              matters there — there is no separate VRAM.
            </>,
          ],
          [
            <code>cpu</code>,
            <code>transformers</code>,
            <>
              Works, and is slow for anything but a small model. <code>gguf</code> is usually the
              better answer on a CPU-only box.
            </>,
          ],
        ]}
        caption={
          <>
            <code>detect_platform()</code> returns the <code>HardwarePlatform</code>;{' '}
            <code>get_best_local_backend()</code> turns it into a backend name. Both are what{' '}
            <Link to="/local-models">local models and engines</Link> uses when you do not name an
            engine.
          </>
        }
      />

      <h2>Watching the GPUs</h2>

      <p>
        <code>GPUMonitor</code> samples on its own thread and raises alerts when a threshold is
        crossed. Metrics land in the collection cycle, so start it and give it one interval before
        reading.
      </p>

      <CodeBlock filename="monitor.py" code={`import time

from effgen.gpu import GPUMonitor, MonitorConfig

monitor = GPUMonitor(MonitorConfig(update_interval=0.5, enable_logging=False))
monitor.start()
time.sleep(1.5)                       # let one collection cycle land
metrics = monitor.get_metrics()
summary = monitor.get_summary()
monitor.stop()

for device_id, gpu in sorted(metrics.items()):
    used, total = gpu.vram_used / 2**30, gpu.vram_total / 2**30
    print(f"gpu {device_id}: {used:5.1f}/{total:.1f} GiB  util {gpu.gpu_utilization:.0%}"
          f"  {gpu.temperature:.0f}C  {gpu.power_usage:.0f}W  {len(gpu.processes)} procs")

print()
print("devices:", summary["num_devices"], " alerts raised:", summary["total_alerts"])`} />

      <Terminal
        command="python monitor.py"
        output={`no NVIDIA GPU visible to this process`}
        caption="A shared host with every card busy. The process count per device is how you find out whose job is holding the memory."
      />

      <ParamTable
        nameLabel="MonitorConfig"
        params={[
          { name: 'update_interval', type: 'float', default: '1.0', description: 'Seconds between samples.' },
          { name: 'enable_alerts', type: 'bool', default: 'True', description: 'Raise alerts when a threshold is crossed.' },
          { name: 'enable_logging', type: 'bool', default: 'True', description: 'Log each collection. Off is quieter in a script.' },
          { name: 'log_interval', type: 'float', default: '60.0', description: 'Seconds between log lines when logging is on.' },
          { name: 'vram_warning_threshold', type: 'float', default: '0.8', description: 'Fraction of VRAM used that raises a warning.' },
          { name: 'vram_critical_threshold', type: 'float', default: '0.95', description: 'And a critical alert.' },
          { name: 'gpu_utilization_warning', type: 'float', default: '0.9', description: 'Sustained utilisation that raises a warning.' },
          { name: 'temperature_warning', type: 'float', default: '80.0', description: 'Degrees Celsius.' },
          { name: 'temperature_critical', type: 'float', default: '90.0', description: 'Degrees Celsius.' },
          { name: 'power_warning_threshold', type: 'float', default: '0.9', description: 'Fraction of the power limit.' },
        ]}
      />

      <ApiTable
        headers={['GPUMonitor', 'What it does']}
        rows={[
          [
            <>
              <code>start()</code> / <code>stop()</code>
            </>,
            'Run and halt the sampling thread.',
          ],
          [
            <code>get_metrics(device_id=None)</code>,
            <>
              The latest <code>GPUMetrics</code> per device, or one device.
            </>,
          ],
          [<code>get_metrics_history()</code>, 'The samples kept so far.'],
          [<code>get_summary()</code>, 'Device count, whether monitoring is active, alert count, and a per-device digest.'],
          [
            <code>add_alert_callback(fn)</code>,
            <>
              Called with an <code>Alert</code> — <code>device_id</code>, <code>metric_type</code>,{' '}
              <code>level</code>, <code>message</code>, <code>value</code>, <code>threshold</code>.
              This is the hook to a pager.
            </>,
          ],
          [
            <>
              <code>get_alerts()</code> / <code>clear_alerts()</code>
            </>,
            'Read and drain what has been raised.',
          ],
        ]}
        caption={
          <>
            A <code>GPUMetrics</code> carries <code>vram_total</code>, <code>vram_used</code>,{' '}
            <code>vram_free</code> (bytes), <code>gpu_utilization</code> and{' '}
            <code>memory_utilization</code> (fractions, not percentages),{' '}
            <code>temperature</code>, <code>power_usage</code>, <code>power_limit</code>,{' '}
            <code>fan_speed</code> and <code>processes</code>.
          </>
        }
      />

      <h2>Allocating a device</h2>

      <p>
        On a shared machine, "which GPU should this job take" is a real question.{' '}
        <code>GPUAllocator</code> answers it against live free memory rather than a static
        assignment.
      </p>

      <CodeBlock filename="allocator.py" code={`from effgen.gpu import AllocationRequest, AllocationStrategy, GPUAllocator

allocator = GPUAllocator()
print("cuda available:", allocator.is_cuda_available())

request = AllocationRequest(
    requester_id="my-job",
    memory_required=8 * 2**30,          # 8 GiB
    num_gpus=1,
    strategy=AllocationStrategy.BALANCED,
)
print("can allocate  :", allocator.can_allocate(request))
allocation = allocator.allocate(request)
print("allocation    :", allocation)
if allocation:
    allocator.deallocate("my-job")`} />

      <Terminal
        command="python allocator.py"
        output={`cuda available: True
can allocate  : True
allocation    : Allocation(requester_id='my-job', device_ids=[1], memory_allocated={1: 8589934592}, parallelism_type=<ParallelismType.NONE: 'none'>, timestamp=1787542944.8308406)`}
        caption="Device 1 was the balanced choice at that moment — it had the most free memory. Ask again a minute later and the answer can differ, which is the point."
      />

      <ParamTable
        nameLabel="AllocationRequest"
        params={[
          { name: 'requester_id', type: 'str', required: true, description: 'Your handle for the allocation; deallocate by the same id.' },
          { name: 'memory_required', type: 'int', required: true, description: 'Bytes. Weights plus activations plus the KV cache, not just weights.' },
          { name: 'num_gpus', type: 'int', default: '1', description: 'How many devices the job needs.' },
          {
            name: 'strategy',
            type: 'AllocationStrategy',
            default: 'BALANCED',
            description: (
              <>
                <code>GREEDY</code> takes the first that fits, <code>BALANCED</code> spreads load,{' '}
                <code>OPTIMIZE</code> packs, <code>PRIORITY</code> honours the priority field.
              </>
            ),
          },
          {
            name: 'parallelism_type',
            type: 'ParallelismType',
            default: 'NONE',
            description: (
              <>
                <code>TENSOR</code>, <code>PIPELINE</code>, <code>DATA</code> or{' '}
                <code>NONE</code> — recorded on the allocation so a multi-GPU job says what it is
                doing.
              </>
            ),
          },
          { name: 'preferred_devices', type: 'list[int] | None', default: 'None', description: 'Try these first.' },
          { name: 'priority', type: 'int', default: '0', description: 'Higher wins under PRIORITY.' },
          { name: 'allow_shared', type: 'bool', default: 'True', description: 'Whether a device already carrying work is a candidate.' },
        ]}
        caption={
          <>
            <code>can_allocate(request)</code> asks without taking anything;{' '}
            <code>allocate(request)</code> returns an <code>Allocation</code> or <code>None</code>;{' '}
            <code>deallocate(requester_id)</code> releases it. <code>get_device_info()</code>,{' '}
            <code>get_available_memory()</code>, <code>get_total_memory()</code> and{' '}
            <code>list_allocations()</code> are the read side.
          </>
        }
      />

      <Callout type="note" title="It books, it does not enforce">
        <p>
          The allocator tracks what effGen handed out in this process. It cannot stop another
          process — or another user on a shared node — from taking the same memory. Pin the device
          with <code>CUDA_VISIBLE_DEVICES</code> once you have an answer, and treat{' '}
          <code>can_allocate</code> as advice that can go stale between the question and the load.
        </p>
      </Callout>

      <h2>Sizing a model</h2>

      <ApiTable
        headers={['Precision', 'Bytes per parameter', 'A 7B model needs roughly']}
        rows={[
          ['float32', '4', '28 GB'],
          ['float16 / bfloat16', '2', '14 GB'],
          ['8-bit', '1', '7 GB'],
          ['4-bit', '0.5', '3.5 GB'],
        ]}
        caption={
          <>
            Weights only. Add the KV cache, which grows with context length and batch size, and the
            activations — in practice budget 20–30% above the weight figure before deciding
            something fits. <Link to="/local-models">Local models and engines</Link> covers
            quantisation and the engine flags.
          </>
        }
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <>
              <code>cuda_usable()</code> is False on a machine with GPUs
            </>,
            <>
              <code>physical_gpus &gt; 0</code> and <code>mismatch=True</code> — torch cannot use
              them.
            </>,
            <>
              Read <code>status.message</code>. Usually a driver older than the CUDA version torch
              was built against.
            </>,
          ],
          [
            <>
              <code>get_metrics()</code> returns an empty dict
            </>,
            'The monitor has not sampled yet, or was never started.',
            <>
              Call <code>start()</code> and wait one <code>update_interval</code> before reading.
            </>,
          ],
          [
            'CUDA out of memory on a card the status table says is free',
            'Another process took it between the check and the load. The table is a snapshot.',
            <>
              Pin with <code>CUDA_VISIBLE_DEVICES</code> and retry, or lower the memory the engine
              reserves.
            </>,
          ],
          [
            'Every device shows 100% utilisation but low memory',
            'Something else on the host is compute-bound. Utilisation is machine-wide, not yours.',
            <>
              Read the process count per device from <code>GPUMetrics.processes</code> before
              concluding it is your job.
            </>,
          ],
          [
            <>
              <code>get_best_local_backend()</code> says <code>transformers</code> on a CUDA box
            </>,
            'vLLM is not installed.',
            <>
              <code>pip install "effgen[vllm]"</code>, or name the engine yourself.
            </>,
          ],
          [
            'A model that fits on paper does not load',
            'The weight figure ignores the KV cache and activations.',
            <>
              Shorten the context, lower the batch size, or quantise. The table above is weights
              only.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>effgen models status --json</code> is new, and <code>models info</code> now answers
          for a local engine id as well as a hosted one.
        </p>
      </Callout>

      <SeeAlso paths={['/local-models', '/installation', '/observability']} />
    </DocPage>
  );
}
