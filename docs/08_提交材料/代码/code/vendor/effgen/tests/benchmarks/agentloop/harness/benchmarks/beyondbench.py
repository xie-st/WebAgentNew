"""The three algorithmic reasoning sets, run with a code execution tool.

Problems are generated, not downloaded, so the generators and the seed are what
pin the data. They are ports of the ones used for the paper run: same seeds,
same task list, same sizes, so the problems are the same problems.

  bb_easy   list and arithmetic tasks (LLMThinkBench)   12 tasks x 4 sizes x 5
  bb_med    "what comes next" sequences                  5 tasks x 5 lengths x 5
  bb_hard   NP-hard puzzles                             14 configs x 5
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any

from ..types import Sample
from .base import Benchmark

SAMPLES_PER_VARIATION = 5

RAW_SYSTEM = "You are a helpful assistant. Solve the problem step by step."

TOOL_SYSTEM = (
    "You are a helpful assistant that solves problems by writing and running code.\n"
    "Use the python_exec tool: write a short program that computes the answer and "
    "prints it, then read the printed value.\n"
    "Do not work the answer out by hand when you can compute it.\n"
    "End your reply with the final answer in \\boxed{answer}."
)


# ==================================================================== bb_easy

EASY_TASKS = [
    "sorting", "sum", "multiplication", "subtraction", "division",
    "odd_count", "even_count", "find_maximum", "find_minimum",
    "mean", "median", "mode",
]
EASY_LIST_SIZES = [8, 16, 32, 64]
EASY_MIN_VAL, EASY_MAX_VAL = -1000, 1000
LIST_TASKS = {
    "sorting", "sum", "multiplication", "odd_count", "even_count",
    "find_maximum", "find_minimum", "mean", "median", "mode",
}

EASY_PROMPTS = {
    "sorting": "Solve this question: Sort this list in ascending order: {data}",
    "sum": "Solve this question: Calculate the sum of these numbers: {data}",
    "multiplication": "Solve this question: Calculate the product of these numbers: {data}",
    "subtraction": "Solve this question: Calculate {a} - {b}",
    "division": "Solve this question: Calculate {a} / {b} (round to 2 decimal places if needed)",
    "comparison": "Solve this question: Compare {a} and {b}. Is {a} greater than, less than, or equal to {b}?",
    "absolute_difference": "Solve this question: Calculate the absolute difference |{a} - {b}|",
    "odd_count": "Solve this question: Count how many odd numbers are in this list: {data}",
    "even_count": "Solve this question: Count how many even numbers are in this list: {data}",
    "find_maximum": "Solve this question: Find the maximum value in this list: {data}",
    "find_minimum": "Solve this question: Find the minimum value in this list: {data}",
    "mean": "Solve this question: Calculate the mean (average) of these numbers: {data}",
    "median": "Solve this question: Find the median of these numbers: {data}",
    "mode": "Solve this question: Find the mode (most frequent value) in this list: {data}",
}
BOXED_SUFFIX = "\n\nProvide your final answer in \\boxed{answer} at the end of your response."


def _easy_points(task: str, count: int, list_size: int, seed: int) -> list[dict]:
    random.seed(seed)
    points = []
    for _ in range(count):
        if task in LIST_TASKS:
            points.append({"data": random.sample(range(EASY_MIN_VAL, EASY_MAX_VAL + 1), list_size)})
        else:
            a = random.randint(EASY_MIN_VAL, EASY_MAX_VAL)
            b = random.randint(EASY_MIN_VAL, EASY_MAX_VAL)
            if task == "division" and b == 0:
                b = 1
            points.append({"a": a, "b": b})
    return points


def _easy_truth(task: str, point: dict) -> Any:
    if task == "sorting":
        return sorted(point["data"])
    if task == "sum":
        return sum(point["data"])
    if task == "multiplication":
        result = 1
        for n in point["data"]:
            result *= n
        return result
    if task == "subtraction":
        return point["a"] - point["b"]
    if task == "division":
        return round(point["a"] / point["b"], 2)
    if task == "comparison":
        a, b = point["a"], point["b"]
        return "greater than" if a > b else ("less than" if a < b else "equal to")
    if task == "absolute_difference":
        return abs(point["a"] - point["b"])
    if task == "odd_count":
        return sum(1 for x in point["data"] if x % 2 != 0)
    if task == "even_count":
        return sum(1 for x in point["data"] if x % 2 == 0)
    if task == "find_maximum":
        return max(point["data"])
    if task == "find_minimum":
        return min(point["data"])
    if task == "mean":
        return round(sum(point["data"]) / len(point["data"]), 2)
    if task == "median":
        s = sorted(point["data"])
        n = len(s)
        return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
    if task == "mode":
        return Counter(point["data"]).most_common(1)[0][0]
    return None


def _boxed(response: str) -> str | None:
    if not response:
        return None
    for pattern in [r"\\boxed\{([^{}]+)\}", r"\\boxed\{\\text\{([^{}]+)\}\}", r"\[boxed\{([^{}]+)\}\]"]:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()
    return None


# Phrases a model uses to introduce its answer when it does not box it.
_ANSWER_LEAD = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|the\s+result\s+is|answer)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)


def _answer_text(response: str) -> tuple[str | None, bool]:
    """The span of a response that holds the answer, and whether it was boxed.

    ``\\boxed{}`` first, because that is what every prompt in this file asks
    for. Without a fallback though, this scored whether a framework preserved
    the LaTeX rather than whether it got the problem right: effGen's ReAct loop
    and Smolagents both rewrite the closing line and lose the marker, so a
    correct sorted list written in prose counted as wrong. bb_med always had a
    fallback; bb_easy did not, and that alone moved its column ordering.

    The flag matters because the two spans are read differently. Boxed content
    is the answer and nothing else, so its first number is the answer. A prose
    span is a sentence that may restate the working ("45 + 12 = 57"), so its
    *last* number is.
    """
    boxed = _boxed(response)
    if boxed:
        return boxed, True
    if not response:
        return None, False
    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if not lines:
        return None, False
    # An explicit lead-in anywhere in the response wins over position.
    for line in reversed(lines):
        match = _ANSWER_LEAD.search(line)
        if match and match.group(1).strip():
            return match.group(1).strip(), False
    return lines[-1], False


def _pick_number(nums: list[str], boxed: bool) -> str | None:
    """The number that is the answer: the first if boxed, the last if prose."""
    if not nums:
        return None
    return nums[0] if boxed else nums[-1]


def _parse_easy(answer: str | None, task: str, boxed: bool = True) -> Any:
    if not answer:
        return None
    try:
        if task == "sorting":
            nums = re.findall(r"-?\d+", answer)
            return [int(n) for n in nums] if nums else None
        if task == "comparison":
            low = answer.lower()
            if "greater" in low:
                return "greater than"
            if "less" in low:
                return "less than"
            if "equal" in low:
                return "equal to"
            return answer
        if task in {"division", "mean", "median"}:
            picked = _pick_number(re.findall(r"-?\d+\.?\d*", answer), boxed)
            return float(picked) if picked is not None else None
        picked = _pick_number(re.findall(r"-?\d+", answer), boxed)
        return int(picked) if picked is not None else None
    except (ValueError, IndexError):
        return None


class BeyondBenchEasy(Benchmark):
    key = "bb_easy"
    label = "BB-Easy"
    category = "Coding"
    tools = ("python_exec",)

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return sample.question

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        samples = []
        for task in EASY_TASKS:
            for size in EASY_LIST_SIZES:
                for i, point in enumerate(
                    _easy_points(task, SAMPLES_PER_VARIATION, size, seed)
                ):
                    prompt = EASY_PROMPTS[task].format(**point) + BOXED_SUFFIX
                    samples.append(
                        Sample(
                            sample_id=f"bb_easy-{task}-{size}-{i}",
                            question=prompt,
                            answer=_easy_truth(task, point),
                            meta={"task": task, "list_size": size},
                        )
                    )
        return self._slice(samples, limit, offset)

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        task = sample.meta["task"]
        span, boxed = _answer_text(output)
        predicted = _parse_easy(span, task, boxed=boxed)
        truth = sample.answer
        if predicted is None:
            return False, None
        if task == "sorting":
            return predicted == list(truth), predicted
        if task == "comparison":
            return str(predicted).lower() == str(truth).lower(), predicted
        if task in {"division", "mean", "median"}:
            try:
                return abs(float(predicted) - float(truth)) < 0.1, predicted
            except (TypeError, ValueError):
                return False, predicted
        return predicted == truth, predicted


# ===================================================================== bb_med

MED_TASKS = ["fibonacci", "geometric", "algebraic", "prime", "complex_pattern"]
MED_LENGTHS = [5, 6, 8, 10, 12]
MED_PROMPT = (
    "Solve this question: Find the next term in this sequence: {sequence}"
    "\n\nProvide your final answer in \\boxed{{answer}} at the end of your response."
)
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
]


def _med_sequence(task: str, length: int, seed_val: int) -> list[int]:
    random.seed(seed_val)
    if task == "geometric":
        a, r = random.randint(1, 5), random.choice([2, 3, -2])
        return [a * (r**i) for i in range(length + 2)]
    if task == "algebraic":
        a, d = random.randint(1, 10), random.randint(1, 5)
        return [a + d * i for i in range(length + 2)]
    if task == "prime":
        max_start = max(0, len(PRIMES) - length - 2)
        start = random.randint(0, min(5, max_start)) if max_start > 0 else 0
        return PRIMES[start : start + length + 2]
    if task == "complex_pattern":
        # squares plus a linear term: harder to spot than the other four.
        a, b = random.randint(1, 4), random.randint(1, 6)
        return [a * (i**2) + b * i for i in range(1, length + 3)]
    a, b = random.randint(1, 5), random.randint(1, 5)
    seq = [a, b]
    for _ in range(length):
        seq.append(seq[-1] + seq[-2])
    return seq


class BeyondBenchMedium(Benchmark):
    key = "bb_med"
    label = "BB-Med"
    category = "Coding"
    tools = ("python_exec",)

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return sample.question

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        samples = []
        for task in MED_TASKS:
            for length in MED_LENGTHS:
                for i in range(SAMPLES_PER_VARIATION):
                    seq = _med_sequence(task, length, seed + i)
                    samples.append(
                        Sample(
                            sample_id=f"bb_med-{task}-{length}-{i}",
                            question=MED_PROMPT.format(sequence=seq[:-1]),
                            answer=seq[-1],
                            meta={"task": task, "length": length, "sequence": seq[:-1]},
                        )
                    )
        return self._slice(samples, limit, offset)

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        span, boxed = _answer_text(output)
        if span is None:
            return False, None
        picked = _pick_number(re.findall(r"-?\d+", str(span)), boxed)
        if picked is None:
            return False, None
        predicted = int(picked)
        return predicted == sample.answer, predicted


# ==================================================================== bb_hard

HANOI_DISKS = [3, 4, 5, 6]
NQUEENS_SIZES = [4, 6, 8]
GRAPH_NODES = [12, 16, 20]
MATRIX_SIZES = [3, 5, 7, 10]

HARD_PROMPTS = {
    "tower_hanoi": """Solve this Tower of Hanoi puzzle with {n} disks.

RULES:
1. Only one disk can be moved at a time
2. A larger disk cannot be placed on top of a smaller disk
3. Only the topmost disk on any peg can be moved

INITIAL STATE:
Peg A: {initial_state} (disk 1 is smallest, disk {n} is largest)
Peg B: []
Peg C: []

GOAL: Move all disks from Peg A to Peg C.

Provide the complete sequence of moves using format: "Move disk X from Y to Z" """,
    "n_queens": """N-Queens Problem:

Place {n} queens on a {n}x{n} chessboard such that no two queens threaten each other.

Provide your answer as a list of column positions (0-indexed) for each row.
Format: [col0, col1, col2, ...]""",
    "graph_coloring": """Graph Coloring Problem:

Vertices: {vertices}
Edges: {edges}

Assign colors (numbers 0, 1, 2, ...) to vertices so no adjacent vertices share the same color.
Use at most {budget} different colors.
Give the complete assignment for every vertex on the last line, in this format
and nothing else: {{vertex: color, ...}}""",
    "matrix_chain": """Matrix Chain Multiplication:

Multiply {n} matrices with dimensions: {dimensions}
What is the minimum number of scalar multiplications needed?

Give the final answer as a single number in \\boxed{{answer}} at the end of your
response.""",
}


class _Pegs:
    def __init__(self, pegs):
        self.pegs = {k: list(v) for k, v in pegs.items()}

    def valid(self, src, dst):
        if not self.pegs.get(src):
            return False
        if not self.pegs.get(dst):
            return True
        return self.pegs[src][-1] < self.pegs[dst][-1]

    def move(self, src, dst):
        if not self.valid(src, dst):
            return False
        self.pegs[dst].append(self.pegs[src].pop())
        return True

    def solved(self, target, n):
        return self.pegs[target] == list(range(n, 0, -1))


_HANOI_RE = re.compile(
    r"move\s+(?:disk\s+)?(\d+)\s+from\s+(?:peg\s+)?([abc])\s+to\s+(?:peg\s+)?([abc])",
    re.IGNORECASE,
)


def _hanoi_runs(output: str) -> list[list[tuple[int, str, str]]]:
    """Every contiguous block of move lines in the output, in order.

    Models routinely give the sequence twice — once while working through it
    and once under "Final Answer". Reading the whole output as one list
    replayed the solution on an already-solved board, so the second copy always
    failed as an invalid move. That marked a correct answer wrong for any
    framework whose output restates its plan, which is a formatting difference,
    not a reasoning one. Each block is checked on its own instead.
    """
    runs: list[list[tuple[int, str, str]]] = []
    current: list[tuple[int, str, str]] = []
    for line in (output or "").splitlines():
        match = _HANOI_RE.search(line)
        if match:
            disk, src, dst = match.groups()
            current.append((int(disk), src.upper(), dst.upper()))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _hanoi_run_solves(moves: list[tuple[int, str, str]], n: int) -> bool:
    state = _Pegs({"A": list(range(n, 0, -1)), "B": [], "C": []})
    for _disk, src, dst in moves:
        if not state.move(src, dst):
            return False
    return state.solved("C", n)


def _check_hanoi(output: str, n: int) -> tuple[bool, str]:
    runs = _hanoi_runs(output)
    if not runs:
        return False, "no moves found"
    # The last block is the model's answer; the earlier ones are its working.
    # Accept if any single block is a complete, legal solution.
    for moves in reversed(runs):
        if _hanoi_run_solves(moves, n):
            return True, f"solved in {len(moves)} moves"
    longest = max(len(m) for m in runs)
    return False, f"no legal solution ({len(runs)} block(s), longest {longest} moves)"


def _nqueens_ok(positions: list[int], n: int) -> bool:
    if len(positions) != n or not all(0 <= p < n for p in positions):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if positions[i] == positions[j] or abs(positions[i] - positions[j]) == j - i:
                return False
    return True


def _check_nqueens(output: str, n: int) -> tuple[bool, str]:
    # Read the *last* list of the right length. The old code took the first
    # bracketed list anywhere in the text, which on a reasoning trace is an
    # example or a partial board, not the answer.
    candidates = re.findall(r"\[([0-9,\s]+)\]", output or "")
    if not candidates:
        return False, "could not parse"
    sized = []
    for raw in candidates:
        positions = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        if len(positions) == n:
            sized.append(positions)
    if not sized:
        return False, "wrong number of positions"
    if _nqueens_ok(sized[-1], n):
        return True, "valid solution"
    return False, "queens attack each other"


def _greedy_colour_count(vertices: list[int], edges: list) -> int:
    """Colours a greedy pass needs. Always achievable, so it is a fair budget."""
    neighbours: dict[int, set[int]] = {v: set() for v in vertices}
    for u, v in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    assigned: dict[int, int] = {}
    for v in vertices:
        taken = {assigned[n] for n in neighbours[v] if n in assigned}
        colour = 0
        while colour in taken:
            colour += 1
        assigned[v] = colour
    return len(set(assigned.values())) if assigned else 0


def _parse_colouring(text: str) -> dict[int, int] | None:
    """Read `vertex: colour` pairs out of one brace group.

    LaTeX is the reason this is not a plain split. A model that writes a set
    inside a box writes `\\boxed{\\{0: 0, 1: 1\\}}`, and the brace scanner in
    `_check_coloring` hands this function `0: 0, 1: 1\\` — the escape from the
    closing `\\}` rides along on the last value. `int("1\\")` raises, and the
    old code answered a single bad pair by discarding the whole colouring, so a
    complete and correct assignment scored as "could not parse". Reading the
    pairs with a regex ignores the LaTeX punctuation instead of choking on it.

    Still returns None when a group holds no pairs at all, so a brace group that
    is not a colouring is rejected. The caller is what checks that the pairs
    actually cover every vertex.
    """
    coloring = {
        int(vertex): int(colour)
        for vertex, colour in re.findall(r"(-?\d+)\s*:\s*(-?\d+)", text or "")
    }
    return coloring or None


def _check_coloring(
    output: str, vertices: list[int], edges: list, budget: int
) -> tuple[bool, str]:
    # Read the *last* brace group that parses as a full colouring. The old code
    # took the first one anywhere in the text, which is usually a format example
    # or a LaTeX group, not the answer.
    groups = re.findall(r"\{([^{}]+)\}", output or "")
    coloring = None
    for raw in reversed(groups):
        parsed = _parse_colouring(raw)
        if parsed and set(parsed) == set(vertices):
            coloring = parsed
            break
    if coloring is None:
        return False, "could not parse a colouring of every vertex"
    for u, v in edges:
        if coloring.get(u) == coloring.get(v):
            return False, f"conflict on edge {u}-{v}"
    used = len(set(coloring.values()))
    # Without a budget the task is trivial: one colour per vertex is always a
    # proper colouring, so any model that says "give each vertex its own
    # colour" scored 100%. The budget is what a greedy pass achieves, so it is
    # always satisfiable, and it is stated in the prompt.
    if used > budget:
        return False, f"used {used} colours, budget was {budget}"
    return True, f"valid with {used} colours"


def _matrix_chain_optimal(dims: list[int]) -> int:
    n = len(dims) - 1
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                dp[i][j] = min(
                    dp[i][j], dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                )
    return int(dp[0][n - 1])


class BeyondBenchHard(Benchmark):
    key = "bb_hard"
    label = "BB-Hard"
    category = "Coding"
    tools = ("python_exec",)
    # A 6-disk Hanoi solution is 63 moves, and a 20-vertex colouring is 20
    # pairs. At the default 1024 tokens every framework was cut off mid-answer
    # on the larger instances, so nobody scored on them at all.
    min_max_tokens = 3072

    def system_prompt(self, with_tools: bool) -> str:
        return TOOL_SYSTEM if with_tools else RAW_SYSTEM

    def user_prompt(self, sample: Sample, with_tools: bool) -> str:
        return sample.question

    def load(self, limit=None, offset=0, seed=42) -> list[Sample]:
        samples: list[Sample] = []

        for n in HANOI_DISKS:
            for i in range(SAMPLES_PER_VARIATION):
                state = list(range(n, 0, -1))
                samples.append(
                    Sample(
                        sample_id=f"bb_hard-hanoi-{n}-{i}",
                        question=HARD_PROMPTS["tower_hanoi"].format(n=n, initial_state=state),
                        answer=None,
                        meta={"task": "tower_hanoi", "n_disks": n, "variation": f"{n}_disks"},
                    )
                )

        for n in NQUEENS_SIZES:
            for i in range(SAMPLES_PER_VARIATION):
                samples.append(
                    Sample(
                        sample_id=f"bb_hard-nqueens-{n}-{i}",
                        question=HARD_PROMPTS["n_queens"].format(n=n),
                        answer=None,
                        meta={"task": "n_queens", "n": n, "variation": f"{n}_queens"},
                    )
                )

        for nodes in GRAPH_NODES:
            for i in range(SAMPLES_PER_VARIATION):
                random.seed(seed + i + nodes * 100)
                vertices = list(range(nodes))
                edges = [
                    (a, b)
                    for a in range(nodes)
                    for b in range(a + 1, nodes)
                    if random.random() < 0.5
                ]
                budget = _greedy_colour_count(vertices, edges)
                samples.append(
                    Sample(
                        sample_id=f"bb_hard-coloring-{nodes}-{i}",
                        question=HARD_PROMPTS["graph_coloring"].format(
                            vertices=vertices, edges=edges, budget=budget
                        ),
                        answer=None,
                        meta={
                            "task": "graph_coloring",
                            "vertices": vertices,
                            "edges": edges,
                            "budget": budget,
                            "variation": f"{nodes}_nodes",
                        },
                    )
                )

        for count in MATRIX_SIZES:
            for i in range(SAMPLES_PER_VARIATION):
                random.seed(seed + i + count * 100)
                dims = [random.randint(5, 30) for _ in range(count + 1)]
                samples.append(
                    Sample(
                        sample_id=f"bb_hard-matrix-{count}-{i}",
                        question=HARD_PROMPTS["matrix_chain"].format(
                            n=count, dimensions=dims
                        ),
                        answer=_matrix_chain_optimal(dims),
                        meta={
                            "task": "matrix_chain",
                            "dimensions": dims,
                            "variation": f"{count}_matrices",
                        },
                    )
                )

        return self._slice(samples, limit, offset)

    def score(self, sample: Sample, output: str) -> tuple[bool, Any]:
        task = sample.meta["task"]
        if task == "tower_hanoi":
            ok, msg = _check_hanoi(output, sample.meta["n_disks"])
        elif task == "n_queens":
            ok, msg = _check_nqueens(output, sample.meta["n"])
        elif task == "graph_coloring":
            ok, msg = _check_coloring(
                output,
                sample.meta["vertices"],
                sample.meta["edges"],
                sample.meta["budget"],
            )
        elif task == "matrix_chain":
            expected = sample.answer
            # Read the answer the model settled on, not every integer it wrote.
            # `expected in found` over the whole output scored a hit whenever a
            # dynamic-programming table happened to print the optimum among its
            # intermediate values, so it rewarded verbose reasoning and
            # penalised short answers.
            span, boxed = _answer_text(output)
            picked = _pick_number(re.findall(r"-?\d+", span or ""), boxed)
            ok = picked is not None and int(picked) == expected
            msg = f"expected {expected}, read {picked}"
        else:
            ok, msg = False, "unknown task"
        return ok, msg
