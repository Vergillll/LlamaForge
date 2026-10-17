"""Per-model build pins: config.json model_builds = {model id: ref}.

A ref is a prebuilt install's directory name under engines/llama.cpp (the
names launch profiles pin) or "ik_llama" for ik_llama_server_bin. A pin to
the build the router already runs is no pin: the model stays a router slot.
Any other pin runs the model in its own process (slotproc). Pure.
"""
import os

IK = "ik_llama"


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def _same(a, b):
    return bool(a and b) and _norm(a) == _norm(b)


def router_bin(cfg):
    """The binary the router runs (routes._active_server_bin's rule)."""
    if cfg.get("active_engine") == "ikllama":
        return cfg.get("ik_llama_server_bin", "")
    return cfg.get("server_bin", "")


def _ik_bin(cfg):
    p = cfg.get("ik_llama_server_bin") or ""
    return p if p and os.path.exists(p) else ""


def options(cfg, installs):
    """[{ref, label, server_bin, router}] the model editor offers."""
    rb = router_bin(cfg)
    out = [{"ref": os.path.basename(i["dir"]),
            "label": i.get("label") or i.get("tag") or os.path.basename(i["dir"]),
            "server_bin": i["server_bin"], "router": _same(i["server_bin"], rb)}
           for i in installs if i.get("server_bin")]
    ik = _ik_bin(cfg)
    if ik:
        out.append({"ref": IK, "label": "ik_llama.cpp", "server_bin": ik,
                    "router": _same(ik, rb)})
    return out


def resolve(ref, cfg, installs):
    """(server_bin, "") or (None, why) for a ref."""
    if ref == IK:
        ik = _ik_bin(cfg)
        return (ik, "") if ik else (None, "ik_llama.cpp isn't built - build it in Build / Update")
    inst = next((i for i in installs if os.path.basename(i["dir"]) == ref), None)
    if not inst or not inst.get("server_bin"):
        return None, (f"build {ref} is no longer installed - pick another build "
                      f"or reinstall it from Build / Update")
    return inst["server_bin"], ""


def pinned_bin(mid, cfg, installs):
    """(server_bin, "") when `mid` runs in its own process; (None, "") when the
    router serves it; (None, why) for a pin that can't be honoured. Never falls
    back to the router: the router's build may be why the model was pinned."""
    ref = (cfg.get("model_builds") or {}).get(mid)
    if not ref:
        return None, ""
    sbin, err = resolve(ref, cfg, installs)
    if err:
        return None, err
    return (None, "") if _same(sbin, router_bin(cfg)) else (sbin, "")


def source_dir(mid, cfg, installs):
    """The checkout the binary that loads `mid` was built from, or "" when it
    isn't one (a prebuilt install, a binary elsewhere, a broken pin)."""
    sbin, err = pinned_bin(mid, cfg, installs)
    if err:
        return ""
    sbin = sbin or router_bin(cfg)
    if not sbin:
        return ""
    if _same(sbin, cfg.get("ik_llama_server_bin")):
        return cfg.get("ik_llama_src") or ""
    for key in ("llama_src", "ik_llama_src"):
        src = cfg.get(key)
        if src and _norm(sbin).startswith(_norm(src) + os.sep):
            return src
    return ""


def validate(ref, cfg, installs):
    """'' when `ref` may be stored ("" clears the pin), else why not."""
    if ref == "":
        return ""
    if not isinstance(ref, str):
        return "build must be a string"
    if ref != IK and ref not in {os.path.basename(i["dir"]) for i in installs}:
        return f"unknown build {ref!r}"
    return resolve(ref, cfg, installs)[1]


def protected_dirs(cfg, installs):
    """Install dirs a model pins, so pruning old builds never breaks one."""
    refs = set((cfg.get("model_builds") or {}).values())
    return [i["dir"] for i in installs if os.path.basename(i["dir"]) in refs]
