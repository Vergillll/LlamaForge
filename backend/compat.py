"""Which llama.cpp family can load a GGUF, read from its tensor type ids.

Mainline llama.cpp and ik_llama.cpp share ggml's type enum up to BF16 (30) and
MXFP4 (39), then diverge: each has quants the other lacks, and the loader
refuses a type it doesn't know with "invalid ggml type N" before it reads a
byte of weights. The tensor infos in the GGUF header say which types a file
uses, so this is known before anything loads.

It's advice, never a refusal: a third fork can reuse an id (PrismML's
llama.cpp fork also calls 142 PQ2_0), and LlamaForge can't tell which fork a
binary is. The enums below are from each project's ggml/include/ggml.h,
checked 2026-10-05; an id this table doesn't know reads as "unknown".

Ids also move with time: ik retired 142 (IQ2_TN) and later reused it for
PQ2_0, and an ik built in between crashes on a PQ2_0 file instead of refusing
it. So build_types() reads the enum of the ik checkout that's installed here.

Pure stdlib; for_model() reads only the GGUF header, via gguf.layout().
"""
import os
import re

import gguf

# ggml type ids with the same meaning in both families
SHARED = frozenset(range(0, 4)) | frozenset(range(6, 31)) | {39}
# the same id, a different type: mainline Q1_0 vs ik Q1_0_G128. Don't guess.
AMBIGUOUS = frozenset({41})

_NAMES_SHARED = (
    "0 F32 1 F16 2 Q4_0 3 Q4_1 6 Q5_0 7 Q5_1 8 Q8_0 9 Q8_1 10 Q2_K 11 Q3_K 12 Q4_K "
    "13 Q5_K 14 Q6_K 15 Q8_K 16 IQ2_XXS 17 IQ2_XS 18 IQ3_XXS 19 IQ1_S 20 IQ4_NL "
    "21 IQ3_S 22 IQ2_S 23 IQ4_XS 24 I8 25 I16 26 I32 27 I64 28 F64 29 IQ1_M 30 BF16 "
    "39 MXFP4")
_NAMES_MAINLINE = "34 TQ1_0 35 TQ2_0 40 NVFP4 41 Q1_0 42 Q2_0"
_NAMES_IK = (
    "31 Q4_0_4_4 32 Q4_0_4_8 33 Q4_0_8_8 36 I2_S 41 Q1_0_G128 97 Q8_0_X4 98 Q8_1_X4 "
    "99 Q8_2_X4 133 Q6_0 134 IQ1_BN 135 IQ2_BN 136 Q8_K64 137 IQ2_K 138 IQ3_K "
    "139 IQ4_K 140 IQ5_K 141 IQ6_K 142 PQ2_0 143 PTQ1_0 144 IQ4_KS 145 IQ2_KS "
    "146 IQ4_KSS 147 Q8_K16 148 Q8_K32 149 Q8_KR8 150 Q8_K128 151 Q8_KV 152 IQ5_KS "
    "153 IQ2_KT 154 IQ3_KT 155 IQ4_KT 156 IQ3_KS 157 IQ2_KL 158 IQ1_KT "
    "159 Q1_0_G128_R8 160 PQ2_0_R8 161 PTQ1_0_R8 202 Q4_0_R8 206 Q5_0_R4 208 Q8_0_R8 "
    "210 Q2_K_R4 211 Q3_K_R4 212 Q4_K_R4 213 Q5_K_R4 214 Q6_K_R4 216 IQ2_XXS_R4 "
    "217 IQ2_XS_R4 218 IQ3_XXS_R4 219 IQ1_S_R4 220 IQ4_NL_R4 221 IQ3_S_R4 "
    "222 IQ2_S_R4 223 IQ4_XS_R8 229 IQ1_M_R4 230 BF16_R16 233 Q6_0_R4 335 IQ2_BN_R4 "
    "337 IQ2_K_R4 338 IQ3_K_R4 339 IQ4_K_R4 340 IQ5_K_R4 344 IQ4_KS_R4 "
    "345 IQ4_KS_R16 352 IQ5_KS_R4 353 MXFP4_R8 397 Q8_K_R16 398 Q8_KV_R8 399 Q8_K_R8")


def _table(s):
    it = iter(s.split())
    return {int(i): n for i, n in zip(it, it)}


_SHARED_N, _MAIN_N, _IK_N = _table(_NAMES_SHARED), _table(_NAMES_MAINLINE), _table(_NAMES_IK)
MAINLINE = SHARED | frozenset(_MAIN_N)
IK = SHARED | frozenset(_IK_N)

UNKNOWN = {"class": "unknown", "types": [], "ids": []}
_CACHE = {}       # normcased path -> ((path, mtime_ns, size), verdict)
_SRC_CACHE = {}   # ggml.h path -> (mtime_ns, ids)
# a live enum line; a retired one is commented out ("// depricated: GGML_TYPE_IQ2_TN = 142,")
_ENUM_RE = re.compile(r"^\s*GGML_TYPE_\w+\s*=\s*(\d+)", re.M)
_ARCH_CACHE = {}  # llama-arch.cpp path -> (mtime_ns, names)
_ARCH_RE = re.compile(r"\{\s*LLM_ARCH_\w+\s*,\s*\"([^\"]+)\"\s*\}")
_NOT_MODELS = frozenset({"clip", "(unknown)"})   # quantize's dummy, the fallback


def type_name(t):
    if t in _SHARED_N:
        return _SHARED_N[t]
    if t in AMBIGUOUS:
        return f"{_MAIN_N[t]} / {_IK_N[t]}"
    return _MAIN_N.get(t) or _IK_N.get(t) or f"type {t}"


def classify(types):
    """"any" | "mainline-only" | "ik-only" | "unknown" for a set of type ids."""
    types = set(types)
    if not types or types & AMBIGUOUS:
        return "unknown"
    main, ik = types <= MAINLINE, types <= IK
    if main and ik:
        return "any"
    return "mainline-only" if main else "ik-only" if ik else "unknown"


def deciding(types):
    """Names of the types that make a file family-specific (not shared), by id."""
    return [type_name(t) for t in sorted(set(types) - SHARED)]


def for_model(path):
    """{"class", "types", "ids"} for a model file (all its shards); unknown when it
    can't be read. Cached until the file's mtime or size changes."""
    if not path:
        return dict(UNKNOWN)
    try:
        st = os.stat(path)
    except OSError:
        return dict(UNKNOWN)
    key = os.path.normcase(os.path.abspath(path))
    stamp = (key, st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return dict(hit[1])
    lay = gguf.layout(path)
    if lay is None:
        out = dict(UNKNOWN)
    else:
        types = {t for _, t, _ in lay["tensors"]}
        out = {"class": classify(types), "types": deciding(types), "ids": sorted(types)}
    _CACHE[key] = (stamp, out)
    return dict(out)


def build_types(src):
    """The type ids a checkout's ggml.h defines (retired slots excluded), or
    None when it can't tell. Cached until the header changes."""
    if not src:
        return None
    path = os.path.join(src, "ggml", "include", "ggml.h")
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    hit = _SRC_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            ids = frozenset(int(m) for m in _ENUM_RE.findall(f.read())) or None
    except OSError:
        return None
    _SRC_CACHE[path] = (mtime, ids)
    return ids


def build_archs(src):
    """The model architectures a checkout's src/llama-arch.cpp names, or None
    when it can't tell. Cached until the file changes."""
    if not src:
        return None
    path = os.path.join(src, "src", "llama-arch.cpp")
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    hit = _ARCH_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            names = frozenset(_ARCH_RE.findall(f.read())) - _NOT_MODELS or None
    except OSError:
        return None
    _ARCH_CACHE[path] = (mtime, names)
    return names


def arch_advice(arch, known):
    """One line when the checkout predates the file's architecture, else ""."""
    if not arch or not known or arch in known or arch in _NOT_MODELS:
        return ""
    return (f"Architecture {arch} is newer than this llama.cpp build: it stops with "
            f"\"unknown model architecture\". Update llama.cpp in Build / Update.")


def missing(ids, known):
    """Names of the file's types a build doesn't define; [] when unknown."""
    return [type_name(t) for t in ids if t not in known] if known else []


def advice(cls, types, engine, build="", ik_built=False, ik_missing=()):
    """One line for the model editor when the build this model runs on can't
    load the file, else "". `engine` is config's active_engine; `build` the
    model's own pin (builds.py; "ik_llama" is ik), which wins over the engine;
    `ik_built` says ik_llama.cpp is there to pick; `ik_missing` names the
    file's types the ik checkout here predates (missing())."""
    names = ", ".join(types) or "a quant type"
    on_ik = build == "ik_llama" if build else engine == "ikllama"
    stale = ", ".join(ik_missing)
    if cls == "ik-only" and not on_ik:
        head = (f"Uses {names}, which mainline llama.cpp can't load (it stops with "
                f"\"invalid ggml type\").")
        if ik_built and stale:
            return head + (f" ik_llama.cpp does, but the build here predates {stale}: update "
                           f"it in Build / Update, then pick it as this model's build.")
        if ik_built:
            return head + " ik_llama.cpp can: pick it as this model's build."
        return head + (" ik_llama.cpp defines it, and so may the llama.cpp fork the model "
                       "was published for. Build ik_llama.cpp in Build / Update, then pick "
                       "it as this model's build.")
    if cls == "mainline-only" and on_ik:
        if build:
            return (f"Uses {names}, which ik_llama.cpp doesn't have. Pick a llama.cpp "
                    f"build for this model.")
        return (f"Uses {names}, which ik_llama.cpp doesn't have. Switch the engine to "
                f"llama.cpp to load it.")
    if on_ik and stale:
        return (f"Uses {stale}, newer than the ik_llama.cpp build here: it can crash loading "
                f"the file instead of saying why. Update ik_llama.cpp in Build / Update.")
    return ""


def load_failure(t):
    """The fix for llama.cpp's "invalid ggml type <t>" (see diag.py)."""
    name = type_name(t)
    if t in IK and t not in MAINLINE:
        return (f"The file uses {name} (type {t}), a quant mainline llama.cpp doesn't have. "
                f"ik_llama.cpp can load it, and so may the llama.cpp fork the model was "
                f"published for.")
    if t in MAINLINE and t not in IK:
        return (f"The file uses {name} (type {t}), which is newer than this build or "
                f"mainline-only. Update llama.cpp (Build / Update tab); ik_llama.cpp "
                f"doesn't have it.")
    if t in MAINLINE or t in IK:
        return (f"The file uses {name} (type {t}), which is newer than this build. "
                f"Update it (Build / Update tab).")
    return (f"The file uses type {t}, which neither mainline llama.cpp nor ik_llama.cpp "
            f"defines: it was made for another fork. The model card should say which "
            f"build it needs.")
