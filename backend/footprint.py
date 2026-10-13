"""How much GPU and system memory one model load takes: predicted from the GGUF,
or measured on an earlier load of the same configuration.

The prediction mirrors where llama.cpp puts things: the input embedding always
stays in system RAM, n-gpu-layers offloads the *last* N blocks (the output layer
only when N exceeds the block count), cpu-moe / n-cpu-moe / override-tensor pull
matching tensors back to the CPU, and each block's KV cache lives with its
block. Hybrid models (Qwen3-Next/3.5, Nemotron-H) only have KV on their
attention layers, sliding-window layers only cache the window, and layers that
share KV (Gemma 3n/4 E-models) or predict ahead (MTP) cache nothing.

A prediction is only the first guess. A footprint measured around a real load
(per-GPU used before vs after) replaces it, and the memory an unload frees is
the better reading still: the CUDA pool grows on the first long prompt, so what
comes back at unload is what the model really held.

Stdlib only.
"""
import hashlib
import json
import os
import re
import threading

import atomicio
import slots
from vramwise import constants as _vw

MIB = 1024 * 1024
NOISE_MIB = 64          # per-GPU deltas below this are other apps breathing
MMPROJ_COMPUTE_MIB = 320   # the projector's compute buffer; measured 177 (gemma) - 252 (qwen)

_KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32,
             "q4_1": 20 / 32, "iq4_nl": 18 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32,
             # low-bit cache types some builds add (block bytes / 32 values)
             "q6_0": 26 / 32, "q6_1": 28 / 32, "q3_0": 14 / 32, "q3_1": 16 / 32,
             "q2_0": 10 / 32, "q2_1": 12 / 32}
# KVarN record caches (kvarnN, N bits a value): a record holds one 128-token
# group of one head slice, packed at N bits plus uint16 scale/zero-point per row
# and one uint16 per column. Head dims map to (record dim, slices).
_KVARN = re.compile(r"kvarn([234568])$")
_KVARN_GROUP = 128
_KVARN_SLICES = {64: (64, 1), 128: (128, 1), 256: (128, 2), 512: (128, 4)}
_TAIL_ROLLBACK = 1      # rows kept for the one-token removal probe when nothing asks for more
# Archs whose sliding-window layout lives in llama.cpp's code, not in the GGUF:
# every Nth layer is global, the rest slide.
_SWA_PERIOD = {"gemma2": 2, "gemma3": 6, "cohere2": 4, "gpt-oss": 2}
_EXPS = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps"
_ALIASES = {"c": "ctx-size", "ngl": "n-gpu-layers", "gpu-layers": "n-gpu-layers",
            "ot": "override-tensor", "ctk": "cache-type-k", "ctv": "cache-type-v",
            "ncmoe": "n-cpu-moe", "cmoe": "cpu-moe", "np": "parallel",
            "ub": "ubatch-size", "dev": "device"}
_PLACEMENT = {"device", "dev", "split-mode", "sm", "main-gpu", "mg", "tensor-split", "ts"}


def _norm(settings):
    out = {}
    for k, v in (settings or {}).items():
        if v is None:
            continue
        k = str(k).strip().lower()
        out[_ALIASES.get(k, k)] = str(v).strip()
    return out


def _flag(s, key):
    v = s.get(key)
    return v is not None and v.lower() not in ("0", "false", "off", "no", "disabled")


def _enabled(s, key):
    """A default-on llama.cpp switch: off by `key = false` or by `no-key`."""
    return not (_flag(s, "no-" + key) or s.get(key, "").lower()
                in ("0", "false", "off", "no", "disabled"))


def _int(s, key, default):
    try:
        return int(float(s[key]))
    except (KeyError, ValueError):
        return default


def _overrides(s):
    """[(compiled regex, on_cpu)] in llama.cpp's first-match order, or None if a
    pattern doesn't compile (then the RAM/VRAM split is a guess)."""
    rules = []
    for part in (s.get("override-tensor") or "").split(","):
        if "=" not in part:
            continue
        pat, buft = part.rsplit("=", 1)
        try:
            rules.append((re.compile(pat.strip()), buft.strip().upper().startswith("CPU")))
        except re.error:
            return None
    if _flag(s, "cpu-moe"):
        rules.append((re.compile(_EXPS), True))
    for i in range(_int(s, "n-cpu-moe", 0)):
        rules.append((re.compile(rf"blk\.{i}{_EXPS}"), True))
    return rules


def _kvarn_record(record_dim, bits, value):
    """Bytes of one KVarN record: K records are record_dim rows x 128 tokens,
    V records 128 rows x record_dim; uint16 scale and zero point per row, one
    uint16 per column."""
    rows, cols = (_KVARN_GROUP, record_dim) if value else (record_dim, _KVARN_GROUP)
    return (rows * cols * bits + 7) // 8 + (2 * rows + cols) * 2


def _cache_type(name):
    """("kvarn", bits) / ("plain", bytes a value) / (None, 2.0) for a name this
    module doesn't know - counted as f16, and the estimate says so."""
    t = (name or "f16").strip().lower()
    m = _KVARN.match(t)
    if m:
        return "kvarn", int(m.group(1))
    if t in _KV_BYTES:
        return "plain", _KV_BYTES[t]
    return None, 2.0


def _round_up(n, step):
    return -(-n // step) * step


def _kv_layers(kv, n_layer):
    """Per block: (k_elems, v_elems, swa, heads, k_dim, v_dim) per token, or None
    for no KV cache."""
    hkv = kv.get("head_count_kv")
    heads = kv.get("head_count") or 1
    emb = kv.get("embedding_length") or 0
    kl = kv.get("key_length") or (emb // heads if heads else 0)
    vl = kv.get("value_length") or kl
    pattern = kv.get("sliding_window_pattern")
    window = kv.get("sliding_window")
    interval = kv.get("full_attention_interval")
    n_kv = n_layer - (kv.get("nextn_predict_layers") or 0) - (kv.get("shared_kv_layers") or 0)
    out = []
    for i in range(n_layer):
        h = hkv[i] if isinstance(hkv, list) and i < len(hkv) else (
            hkv if isinstance(hkv, int) else heads)
        if i >= n_kv or not h or (interval and (i + 1) % interval):
            out.append(None)
            continue
        if not window:
            swa = False
        elif isinstance(pattern, list) and i < len(pattern):
            swa = bool(pattern[i])
        else:
            period = _SWA_PERIOD.get(kv.get("_arch", ""))
            swa = bool(period) and i % period < period - 1
        k = kv.get("key_length_swa", kl) if swa else kl
        v = kv.get("value_length_swa", vl) if swa else vl
        out.append((h * k, h * v, swa, h, k, v))
    return out


def _mib(b):
    return int(round(b / MIB))


def _projector(s, mmproj_bytes, cpu_only):
    """(first-GPU MiB, RAM MiB) for the mmproj: mtmd puts it, with its compute
    buffer, on the first GPU backend whatever `device` says."""
    if not mmproj_bytes:
        return 0, 0
    if cpu_only or not _enabled(s, "mmproj-offload"):
        return 0, _mib(mmproj_bytes)
    return _mib(mmproj_bytes) + MMPROJ_COMPUTE_MIB, 0


def predict(settings, layout, extra_gpu_bytes=0, size_bytes=None, mmproj_bytes=0):
    """{need_mib (GPU, incl. one process overhead), first_gpu_mib, ram_mib,
    weights_gpu_mib, weights_ram_mib, kv_gpu_mib, kv_ram_mib, staging_mib, ctx,
    ctx_from_model (ctx when ctx-size is unset and the file's own was used, else 0),
    confident}.
    settings: the model's effective settings ([*] merged with its section).
    layout: gguf.layout(); None falls back to the file size and a generous KV.
    extra_gpu_bytes: what goes where the model goes (a draft model).
    first_gpu_mib: the projector's share, charged to CUDA0 by the planner."""
    s = _norm(settings)
    cpu_only = (s.get("device") or "").lower() == "none"
    first_mib, proj_ram = _projector(s, mmproj_bytes, cpu_only)
    if not layout:
        ctx = _int(s, "ctx-size", 0) or 8192
        kv = _vw.KV_GB_PER_1K_CONTEXT * ctx / 1000 * 1e9
        need = (size_bytes or 0) + kv + extra_gpu_bytes
        return {"need_mib": _mib(need) + slots.OVERHEAD_MIB, "first_gpu_mib": first_mib,
                "ram_mib": proj_ram, "weights_gpu_mib": _mib(size_bytes or 0),
                "weights_ram_mib": 0, "kv_gpu_mib": _mib(kv), "kv_ram_mib": 0,
                "staging_mib": 0, "ctx": ctx, "ctx_from_model": 0, "confident": False}

    kv = dict(layout.get("kv") or {}, _arch=layout.get("arch") or "")
    n_layer = int(kv.get("block_count") or 0)
    ngl = s.get("n-gpu-layers", "").lower()
    if cpu_only:
        n_gpu = 0
    elif ngl in ("", "all", "auto") or _int(s, "n-gpu-layers", -1) < 0:
        n_gpu = n_layer + 1
    else:
        n_gpu = _int(s, "n-gpu-layers", n_layer + 1)
    first_gpu = max(n_layer - n_gpu, 0)
    output_on_gpu = n_gpu > n_layer
    rules = _overrides(s)
    confident = rules is not None
    rules = rules or []

    w_gpu = w_ram = 0
    names = set()
    embd = 0
    cpu_blocks = {}                         # block -> bytes of it left on the CPU
    # the next-token (MTP) blocks at the end are only loaded for MTP speculation;
    # otherwise the loader drops them ("model has unused tensor ... ignoring")
    nextn = int(kv.get("nextn_predict_layers") or 0)
    skip_from = n_layer - nextn if nextn and "mtp" not in s.get("spec-type", "").lower() else None
    for name, _typ, nbytes in layout.get("tensors") or []:
        names.add(name)
        m = re.match(r"blk\.(\d+)\.", name)
        if m and skip_from is not None and int(m.group(1)) >= skip_from:
            continue
        if name.startswith(("token_embd", "per_layer_token_embd")):
            on_gpu = False
            if name.startswith("token_embd."):
                embd += nbytes
        elif m:
            on_gpu = int(m.group(1)) >= first_gpu and n_gpu > 0
        else:
            on_gpu = output_on_gpu
        for rx, to_cpu in rules:
            if rx.search(name):
                on_gpu = not to_cpu and not cpu_only
                break
        if on_gpu:
            w_gpu += nbytes
        else:
            w_ram += nbytes
            if m:
                cpu_blocks[m.group(1)] = cpu_blocks.get(m.group(1), 0) + nbytes
    if output_on_gpu and "output.weight" not in names:
        w_gpu += embd                       # tied output: llama.cpp copies the embedding

    ctx = _int(s, "ctx-size", 0) or int(kv.get("context_length") or 4096)
    par = _int(s, "parallel", -1)
    n_seq = par if par > 0 else 4              # unset / -1: llama-server picks 4 slots
    # one KV pool shared by the slots unless asked otherwise; the default turns
    # it on only when the slot count is left to the server
    unified = _flag(s, "kv-unified") or (par <= 0 and not _flag(s, "no-kv-unified"))
    ubatch = _int(s, "ubatch-size", 512)
    window = int(kv.get("sliding_window") or 0)
    swa_cells = min(ctx, _round_up(window * n_seq + ubatch, 256))
    if _flag(s, "swa-full"):
        swa_cells = ctx
    tail = max(_int(s, "kv-tail-tokens", 0), 0)
    uncertain = []

    def ctype(key, swa):
        name = (s.get(key + "-swa") if swa else None) or s.get(key) or "f16"
        kind, val = _cache_type(name)
        note = f"cache type {name} is unknown here: counted as f16"
        if kind is None and note not in uncertain:
            uncertain.append(note)
        return kind, val

    kv_off = _enabled(s, "kv-offload")
    kv_gpu = kv_ram = 0.0
    parts = {"body": 0.0, "exact": 0.0, "stage": 0.0}      # the GPU share of each
    geo = dict(ctx=ctx, swa_cells=swa_cells, window=window, n_seq=n_seq,
               unified=unified, ubatch=ubatch, tail=tail)
    for i, lay in enumerate(_kv_layers(kv, n_layer)):
        if lay is None:
            continue
        k, v, swa, heads, kd, vd = lay
        kk, kx = ctype("cache-type-k", swa)
        vk, vx = ctype("cache-type-v", swa)
        if "kvarn" in (kk, vk):
            body, stage, exact = _kvarn_layer(heads, kd, vd, (kk, kx), (vk, vx), swa, geo, uncertain)
        else:
            body, stage = (swa_cells if swa else ctx) * (k * kx + v * vx), 0
            exact = _exact_rows(tail, swa, geo, rounded=False) * (k + v) * 2
        if kv_off and n_gpu > 0 and i >= first_gpu:
            kv_gpu += body + stage + exact
            parts["body"] += body
            parts["stage"] += stage
            parts["exact"] += exact
        else:
            kv_ram += body + stage + exact

    # op offload runs prompt batches on the GPU even for CPU-resident weights,
    # copying one block's worth at a time into the compute buffer
    staging = (max(cpu_blocks.values()) if cpu_blocks and not cpu_only and n_gpu > 0
               and _enabled(s, "op-offload") else 0)
    gpu = 0 if cpu_only else (_mib(w_gpu) + _mib(kv_gpu) + _mib(extra_gpu_bytes)
                              + _mib(staging) + slots.OVERHEAD_MIB)
    ram = _mib(w_ram) + _mib(kv_ram) + (_mib(extra_gpu_bytes) if cpu_only else 0) + proj_ram
    return {"need_mib": gpu, "first_gpu_mib": first_mib, "ram_mib": ram,
            "weights_gpu_mib": _mib(w_gpu), "weights_ram_mib": _mib(w_ram),
            "kv_gpu_mib": _mib(kv_gpu), "kv_ram_mib": _mib(kv_ram),
            # the GPU KV, split: cache body, exact (unquantized) tail rows with
            # their rollback reserve, and a record cache's F16 staging ring
            "kv_body_mib": _mib(parts["body"]), "kv_exact_mib": _mib(parts["exact"]),
            "kv_stage_mib": _mib(parts["stage"]),
            "staging_mib": _mib(staging), "ctx": ctx,
            "ctx_slot": ctx if unified else ctx // n_seq, "n_seq": n_seq,
            "ctx_from_model": 0 if _int(s, "ctx-size", 0) else ctx,
            "confident": confident and not uncertain, "uncertain": uncertain}


def _exact_rows(tail, swa, geo, rounded):
    """Rows of exact (unquantized) KV a layer keeps for --kv-tail-tokens N:
    (N + rollback) per sequence. N is capped by the layer's window (sliding
    layers) or by a slot's context; a record cache rounds it up to whole
    128-token groups first."""
    if not tail:
        return 0
    n = _round_up(tail, _KVARN_GROUP) if rounded else tail
    cap = geo["window"] if swa and geo["window"] else (
        geo["ctx"] if geo["unified"] else geo["ctx"] // geo["n_seq"])
    return (min(n, cap) + _TAIL_ROLLBACK) * geo["n_seq"]


def _kvarn_layer(heads, kd, vd, ktype, vtype, swa, geo, uncertain):
    """(body, staging, exact) bytes of one layer's KVarN cache, sized the way
    the cache allocates it.

    Body: whole 128-token groups of records per stream - every cell of a full
    attention layer, the window plus the tokens in flight on a sliding one.
    Staging: an F16 ring of 128-token groups, two (the live group and a
    rollback slot) per sequence sharing the pool, plus one permanent sink
    group on full attention layers. Exact tail: --kv-tail-tokens rows per
    sequence in F16, never fewer than one group; a tail that covers a sliding
    layer's whole window replaces its records and staging outright.
    Checked byte for byte against llama-bench --kv-memory (FINDINGS, W6)."""
    n_seq, unified, ctx = geo["n_seq"], geo["unified"], geo["ctx"]
    n_stream = 1 if unified else n_seq
    row = heads * (kd + vd)                         # F16 values a token, K and V
    # a record cache keeps at least one group exact even with no tail asked for
    # (llama-bench --kv-memory: kv_tail_tokens_effective 128 with the flag unset)
    tail = max(geo["tail"], _KVARN_GROUP)
    exact = _exact_rows(tail, swa, geo, rounded=True) * row * 2
    window = geo["window"]
    if swa and window and _round_up(tail, _KVARN_GROUP) >= window:
        return 0, 0, exact                          # the exact ring is the whole cache
    if swa:
        visible = -(-min(geo["swa_cells"], window or geo["swa_cells"]) // _KVARN_GROUP) + 1
        in_flight = max(1, -(-geo["ubatch"] // _KVARN_GROUP))
        groups, stage_groups = max(1, visible + in_flight - 1), 2
    else:
        groups = -(-_round_up(-(-ctx // n_stream), 256) // _KVARN_GROUP)
        stage_groups = 2 * (n_seq if unified else 1) + 1
    cells = groups * _KVARN_GROUP * n_stream

    def side(dim, ctyp, value):
        kind, val = ctyp
        if kind != "kvarn":
            note = "a KVarN cache with a non-KVarN K or V type: counted as given"
            if note not in uncertain:
                uncertain.append(note)
            return cells * heads * dim * val
        sl = _KVARN_SLICES.get(dim)
        if not sl:
            note = f"KVarN with head size {dim} is not sized here: approximated"
            if note not in uncertain:
                uncertain.append(note)
            return cells * heads * dim * (val + 0.375) / 8
        rdim, slices = sl
        return groups * n_stream * heads * slices * _kvarn_record(rdim, val, value)

    body = side(kd, ktype, False) + side(vd, vtype, True)
    stage = 2 * row * _KVARN_GROUP * stage_groups * n_stream
    return body, stage, exact


def merge(saved, unsaved):
    """Effective settings for a prediction: `saved` with the editor's unsaved
    values on top, every key in one spelling; a blank unsaved value unsets the
    key, whichever spelling the section used for it."""
    out = _norm(saved)
    for k, v in (unsaved or {}).items():
        k = str(k).strip().lower()
        k = _ALIASES.get(k, k)
        v = "" if v is None else str(v).strip()
        if v:
            out[k] = v
        else:
            out.pop(k, None)
    return out


def gpu_total(p):
    """Everything a prediction puts on the GPU, the projector included (one GPU)."""
    return p["need_mib"] + p["first_gpu_mib"]


def _search(fits, lo, hi):
    """Largest x in [lo, hi] with fits(x), for fits true up to some x and false
    after it; None when even lo doesn't fit."""
    if not fits(lo):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
    return lo


def fit_limits(settings, layout, budget_mib, ctx_max, n_layer, moe, **kw):
    """Where the editor's sliders stop fitting in budget_mib, the others held:
    {"ctx": largest context (1024-token steps) or None, "layers": for a MoE model
    the fewest MoE layers on the CPU, for a dense one the most GPU layers, or None}."""
    s = dict(settings or {})
    fits = lambda over: gpu_total(predict(dict(s, **over), layout, **kw)) <= budget_mib
    step = 1024
    c = _search(lambda i: fits({"ctx-size": str(i * step)}), 2, max(ctx_max // step, 2))
    out = {"ctx": c * step if c else None, "layers": None}
    if n_layer:
        if moe:          # GPU memory falls as layers move out: search on the count kept on GPU
            g = _search(lambda i: fits({"n-cpu-moe": str(n_layer - i)}), 0, n_layer)
            out["layers"] = None if g is None else n_layer - g
        else:
            out["layers"] = _search(lambda i: fits({"n-gpu-layers": str(i)}), 0, n_layer + 1)
    return out


_MEM_EXACT = {"c", "ngl", "ot", "np", "ub", "b", "fa", "ctk", "ctv", "md", "ncmoe",
              "cmoe", "m", "hf", "hfr", "mmproj", "parallel"}
_MEM_PARTS = ("ctx", "cache", "gpu-layers", "moe", "override-tensor", "mmproj", "batch",
              "flash", "swa", "kv", "draft", "model", "fit", "lora", "rope", "yarn",
              "spec", "mtp", "nextn")


def key(model, build, settings):
    """Identity of a load for the footprint store: the model, the build that runs
    it, and every setting that changes memory. Sampling and placement are left
    out (placement is the second level of the store)."""
    rel = sorted((k, v) for k, v in (
        (str(k).strip().lower(), str(v).strip()) for k, v in (settings or {}).items()
        if v is not None)
        if k not in _PLACEMENT and (k in _MEM_EXACT or any(p in k for p in _MEM_PARTS)))
    h = hashlib.sha1(json.dumps(rel).encode()).hexdigest()[:12]
    return f"{model}|{build or ''}|{h}"


class Store:
    """footprints.json: {key: {"0,1": {"0": MiB, "1": MiB}}} - per load identity,
    per device set, per GPU."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._read()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def measured(self, k):
        """{(devices,): {gpu: MiB}} for slots.plan()."""
        out = {}
        with self._lock:
            for dk, fp in (self._data.get(k) or {}).items():
                try:
                    devs = tuple(int(x) for x in dk.split(",") if x != "")
                    out[devs] = {int(g): int(m) for g, m in fp.items()}
                except (ValueError, AttributeError):
                    continue
        return out

    def record(self, k, devices, fp, when="load"):
        fp = {int(g): int(m) for g, m in (fp or {}).items() if m >= NOISE_MIB}
        if not fp:
            return
        dk = ",".join(str(d) for d in devices)
        with self._lock:
            cur = self._data.setdefault(k, {})
            old = {int(g): int(m) for g, m in (cur.get(dk) or {}).items()}
            if when == "unload" and old:
                fp = {g: max(old.get(g, 0), fp.get(g, 0)) for g in set(old) | set(fp)}
            cur[dk] = {str(g): m for g, m in sorted(fp.items())}
            atomicio.write_json(self.path, self._data)
