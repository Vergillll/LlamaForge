"""Elastic VRAM: when other apps need the GPU, move a loaded model's layers to
the CPU instead of unloading it, keep its conversation (slotkeep.py), and move
them back once the memory is free again.

The planning is pure (ladder, footprints, choose, plan_keys). The controller is
shaped like gamemode.GameMode: a 5 s tick, debounced, with every side effect
behind an injected io object so the tests drive it with a fake."""
import collections
import threading
import time


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def ladder(settings, layout):
    """The steps a model can take, from the user's own setting (index 0, no
    keys of ours) down to nothing on the GPU. Each step is {"keys": what we
    write to models.ini, "gpu_layers", "cpu_moe", "layers"}. Never above what
    the user chose. MoE models move experts first (25 % of the blocks at a
    time), which costs far less speed than whole layers. [] without a block
    count."""
    kv = (layout or {}).get("kv") or {}
    n = _int(kv.get("block_count")) or 0
    if n <= 0:
        return []
    s = {str(k).strip().lower(): v for k, v in (settings or {}).items()}
    user_ngl = _int(s.get("n-gpu-layers", s.get("ngl")))
    top = n if user_ngl is None or user_ngl < 0 else min(user_ngl, n)
    moe = (_int(kv.get("expert_count")) or 0) > 0 and "cpu-moe" not in s
    cpu_moe = (_int(s.get("n-cpu-moe")) or 0) if moe else 0
    steps = [{"keys": {}, "gpu_layers": top, "cpu_moe": cpu_moe, "layers": n}]
    if moe:
        for pct in (25, 50, 75, 100):
            c = n * pct // 100
            if c > steps[-1]["cpu_moe"]:
                steps.append({"keys": {"n-cpu-moe": str(c)}, "gpu_layers": top,
                              "cpu_moe": c, "layers": n})
    base = dict(steps[-1]["keys"])
    for pct in (75, 50, 25, 0):
        g = n * pct // 100
        if g < steps[-1]["gpu_layers"]:
            steps.append({"keys": {**base, "n-gpu-layers": str(g)}, "gpu_layers": g,
                          "cpu_moe": steps[-1]["cpu_moe"], "layers": n})
    return steps


def label(step):
    t = f"{step['gpu_layers']}/{step['layers']}"
    return t + (f", experts of {step['cpu_moe']} on CPU" if step["cpu_moe"] else "")
