# BSD 2-Clause License
# Short utility: clamp a value to [lo, hi].

def clamp(value: float, lo: float, hi: float) -> float:
    """Return value clamped to [lo, hi]."""
    if lo > hi:
        raise ValueError(f'lo ({lo}) must be <= hi ({hi})')
    return max(lo, min(value, hi))
