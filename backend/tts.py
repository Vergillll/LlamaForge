"""Text-to-speech through llama.cpp's own `llama-tts` tool.

llama-server has no speech endpoint, but upstream llama.cpp ships `llama-tts`
(tools/tts), a one-shot binary that runs a TTS backbone GGUF plus an audio
mmproj and writes a WAV. It knows two model families: Qwen3-TTS and Kyutai's
Pocket TTS. This module wraps it as an OpenAI-style POST /v1/audio/speech on
the panel, next to the chat shims.

    <tts_dir>/models/          backbone *.gguf + mmproj-*.gguf; every backbone
    <tts_dir>/models/<name>/   is one model, and each subfolder may hold more
    <tts_dir>/voices/          reference clips (wav/mp3/flac); the file name is
                               the voice name. Both families clone the clip.

Only GGUFs in llama.cpp's model + mmproj layout work. Other popular "TTS GGUF"
repos target other runtimes (qwentts.cpp, pockettts.cpp, CrispASR) and do not
load in llama-tts.

Each request runs llama-tts as a short-lived process (load, speak, exit): no
VRAM is held between requests, at the cost of a couple of seconds of model
load. One job at a time, so two requests never stack two copies on the GPU.

Output is WAV (or raw PCM): encoding mp3/opus would need a codec, and this
backend is stdlib-only. Every common player and browser plays WAV. A request
for another format gets a 400 instead of WAV bytes it would mislabel.

Pure Python stdlib.
"""
import io, os, re, shutil, subprocess, sys, tempfile, threading, urllib.request, wave

import gguf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
REPO_FILES = ["Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
              "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"]
REPO_BYTES = 1847874400 + 446422912

# One-click downloads on the Voice tab. The first entry is the default Get and
# lands in models/ itself, where earlier versions put it.
CATALOG = [
    {"id": "qwen3-tts", "label": "Qwen3-TTS 1.7B", "repo": REPO, "files": REPO_FILES,
     "bytes": REPO_BYTES, "subdir": "", "license": "Apache-2.0",
     "note": "10 languages, clones any voice clip, best on a GPU."},
    {"id": "pocket-tts-en", "label": "Pocket TTS (English)",
     "repo": "EryriLabs/pocket-tts-GGUF",
     "files": ["pocket-tts-en.gguf", "mmproj-pocket-tts-en.gguf"],
     "bytes": 159390816 + 59858080, "subdir": "pocket-tts-en", "license": "CC-BY-4.0",
     "note": "English, about 100M parameters, faster than real time even on a CPU. "
             "Always speaks in a voice clip's voice.",
     # Pocket TTS is near-silent without a reference clip, so it comes with one:
     # a volunteer's donated voice from Kyutai's CC0 voice-donation set.
     "voice": {"name": "kyutai-cc0-2181",
               "url": "https://huggingface.co/kyutai/tts-voices/resolve/main/"
                      "voice-donations/2181_enhanced.wav"}},
]

# What llama-tts does per backbone architecture (general.architecture).
ENGINES = {
    "qwen3tts":  {"engine": "Qwen3-TTS",  "needs_voice": False, "langs": True},
    "pockettts": {"engine": "Pocket TTS", "needs_voice": True,  "langs": False},
}
_UNKNOWN_ENGINE = {"engine": "", "needs_voice": False, "langs": True}

BIN_NAMES = ("llama-tts.exe", "llama-tts")
AUDIO_EXT = (".wav", ".mp3", ".flac")
FORMATS = ("wav", "pcm")   # what /v1/audio/speech can return (no codecs in stdlib)
LANGS = ("en", "zh", "de", "it", "pt", "es", "ja", "ko", "fr", "ru")
MAX_CHARS = 2000
# Qwen3-TTS speaks at 12 frames per second of audio; llama-tts stops at end of
# speech, -n is only the ceiling. ~15 characters per spoken second, doubled for
# slow voices and pauses, never under llama-tts' own default of 512.
FRAMES_PER_CHAR = 1.6
MIN_FRAMES, MAX_FRAMES = 512, 4096
TIMEOUT = 600
_VOICE_RE = re.compile(r"^[\w][\w .-]{0,63}$")
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class TtsError(Exception):
    """`status` is the HTTP status the panel answers with."""
    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------- where
def tts_dir(cfg):
    return (cfg or {}).get("tts_dir") or os.path.join(ROOT, "tts")


def models_dir(cfg):
    return os.path.join(tts_dir(cfg), "models")


def voices_dir(cfg):
    return os.path.join(tts_dir(cfg), "voices")


def find_bin(server_bin):
    """llama-tts beside the active llama-server: a source build and an
    official release both put every tool in the same bin directory."""
    if not server_bin:
        return None
    d = os.path.dirname(server_bin)
    for name in BIN_NAMES:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def find_model(folder):
    """(backbone, mmproj) in folder, either None when absent."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".gguf"))
    except OSError:
        return None, None
    mm = [n for n in names if n.lower().startswith("mmproj")]
    back = [n for n in names if not n.lower().startswith("mmproj")]
    pick = lambda xs: os.path.join(folder, xs[0]) if xs else None
    return pick(back), pick(mm)


def _common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


_ARCH = {}   # (path, mtime) -> architecture; status() runs on every tab load


def _arch(path):
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _ARCH:
        _ARCH[key] = (gguf.metadata(path) or {}).get("architecture")
    return _ARCH[key]


def _folder_models(folder):
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".gguf")
                       and os.path.isfile(os.path.join(folder, n)))
    except OSError:
        return []
    mm = [n for n in names if n.lower().startswith("mmproj")]
    if not mm:
        return []
    out = []
    for b in (n for n in names if not n.lower().startswith("mmproj")):
        # The closest name wins: a Q4_K_M backbone shares the Q8_0 projector.
        best = max(mm, key=lambda m: _common(m.lower()[len("mmproj-"):], b.lower()))
        path = os.path.join(folder, b)
        arch = _arch(path)
        out.append({"id": os.path.splitext(b)[0], "model": path,
                    "mmproj": os.path.join(folder, best), "arch": arch or "",
                    **ENGINES.get(arch, _UNKNOWN_ENGINE)})
    return out


def list_models(folder):
    """Every usable backbone + mmproj pair: models/ first, then each subfolder."""
    out = _folder_models(folder)
    try:
        subs = sorted(n for n in os.listdir(folder) if os.path.isdir(os.path.join(folder, n)))
    except OSError:
        return out
    for sub in subs:
        out += _folder_models(os.path.join(folder, sub))
    return out


def pick_model(models, want=None):
    """The model named `want`, else the first. OpenAI clients send "tts-1"."""
    for m in models:
        if want and m["id"] == want:
            return m
    return models[0] if models else None


def catalog_entry(cid):
    for e in CATALOG:
        if e["id"] == cid:
            return e
    return None


def entry_dir(models_folder, entry):
    return os.path.join(models_folder, entry["subdir"]) if entry["subdir"] else models_folder


def installed(models_folder, entry):
    d = entry_dir(models_folder, entry)
    return all(os.path.isfile(os.path.join(d, f)) for f in entry["files"])


def _urlopen_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "LlamaForge"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(8 * 1024 * 1024)


def fetch_voice(folder, voice, opener=_urlopen_bytes):
    """Save a catalog starter voice unless it is already there. -> fetched?"""
    if resolve_voice(folder, voice["name"]):
        return False
    save_voice(folder, voice["name"], opener(voice["url"]))
    return True


def list_voices(folder):
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(os.path.splitext(n)[0] for n in names
                  if os.path.splitext(n)[1].lower() in AUDIO_EXT
                  and os.path.isfile(os.path.join(folder, n)))


def resolve_voice(folder, name, default=None):
    """Path of the named voice clip, else of `default`, else None.

    An unknown name is not an error: OpenAI clients send "alloy" and friends
    by default, and should still get speech. Names are matched against the
    folder listing, never joined onto it, so "../x" cannot escape."""
    for want in (name, default):
        if not isinstance(want, str) or not _VOICE_RE.match(want):
            continue
        for n in os.listdir(folder) if os.path.isdir(folder) else []:
            stem, ext = os.path.splitext(n)
            if stem == want and ext.lower() in AUDIO_EXT:
                return os.path.join(folder, n)
    return None


MIN_CLIP_S, MAX_CLIP_S = 1.0, 60.0


def save_voice(folder, name, data):
    """Store a reference clip as <folder>/<name>.wav. The panel records and
    re-encodes in the browser, so only a real WAV is accepted here; any other
    format already saved under that name is replaced."""
    if not isinstance(name, str) or not _VOICE_RE.match(name) or name != name.strip():
        raise ValueError("voice name: letters, digits, space, dot, dash; up to 64")
    try:
        with wave.open(io.BytesIO(data)) as w:
            secs = w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, EOFError):
        raise ValueError("not a WAV file")
    if not MIN_CLIP_S <= secs <= MAX_CLIP_S:
        raise ValueError(f"clip is {secs:.1f}s; use {MIN_CLIP_S:.0f} to {MAX_CLIP_S:.0f} seconds")
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, name + ".wav")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    for ext in AUDIO_EXT:
        if ext != ".wav":
            try: os.remove(os.path.join(folder, name + ext))
            except OSError: pass
    return dest


def delete_voice(folder, name):
    path = resolve_voice(folder, name)
    if not path:
        return False
    os.remove(path)
    return True


# ----------------------------------------------------------------- request
def parse_request(body):
    """Validate an OpenAI /v1/audio/speech body -> {text, voice, lang, format,
    model}. `model` picks a model by id; anything else ("tts-1") gets the
    first one. `speed` is accepted and ignored."""
    text = (body or {}).get("input")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input must be non-empty text")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise ValueError(f"input is {len(text)} characters; the limit is {MAX_CHARS}")
    lang = str(body.get("language") or "en").strip().lower()
    if lang not in LANGS:
        raise ValueError(f"language must be one of {', '.join(LANGS)}")
    fmt = body.get("response_format") or "wav"
    fmt = fmt.strip().lower() if isinstance(fmt, str) else fmt
    if fmt not in FORMATS:   # say so; WAV bytes under an .mp3 name fail to decode
        raise ValueError(f"response_format {fmt!r} is not supported; use wav or pcm")
    voice = body.get("voice") if isinstance(body.get("voice"), str) else ""
    model = body.get("model") if isinstance(body.get("model"), str) else ""
    return {"text": text, "voice": voice, "lang": lang, "format": fmt, "model": model}


def build_cmd(binary, model, mmproj, text, out, lang="en", speaker=None):
    frames = int(len(text) * FRAMES_PER_CHAR) + 120
    frames = max(MIN_FRAMES, min(MAX_FRAMES, frames))
    cmd = [binary, "-m", model, "--mmproj", mmproj, "-p", text, "-o", out,
           "-n", str(frames)]
    if lang:                        # None for models whose language is in the weights
        cmd += ["--tts-lang", lang]
    if speaker:
        cmd += ["--tts-speaker-file", speaker]
    return cmd


# --------------------------------------------------------------------- wav
def wav_to_pcm(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.readframes(w.getnframes())


def wav_seconds(data):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


# ------------------------------------------------------------------ status
def status(cfg, want=None):
    binary = find_bin(cfg.get("server_bin", ""))
    mdir = models_dir(cfg)
    models = list_models(mdir)
    active = pick_model(models, want) or {}
    missing = []
    if not binary:
        missing.append("binary")
    if not active:
        missing.append("model")
    catalog = [dict({k: e[k] for k in ("id", "label", "repo", "bytes", "license", "note")},
                    installed=installed(mdir, e)) for e in CATALOG]
    return {"ready": not missing, "missing": missing, "binary": binary or "",
            "model": active.get("model", ""), "mmproj": active.get("mmproj", ""),
            "model_id": active.get("id", ""), "models": models, "catalog": catalog,
            "voices": list_voices(voices_dir(cfg)),
            "default_voice": cfg.get("tts_default_voice", ""),
            "voices_dir": voices_dir(cfg), "models_dir": models_dir(cfg),
            "repo": REPO, "repo_files": REPO_FILES, "repo_bytes": REPO_BYTES,
            "langs": list(LANGS), "max_chars": MAX_CHARS}


def _tail(text, lines=12):
    keep = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(keep[-lines:])


class Speaker:
    """Runs llama-tts, one job at a time. `run` is injectable for tests."""

    def __init__(self, run=subprocess.run):
        self._run = run
        self.lock = threading.Lock()
        self.busy = False
        self.last_tmp = None

    def speak(self, cfg, req):
        st = status(cfg)
        if not st["ready"]:
            raise TtsError("text-to-speech is not set up: missing " + " and ".join(st["missing"]), 503)
        m = pick_model(st["models"], req.get("model"))
        vdir = voices_dir(cfg)
        speaker = resolve_voice(vdir, req["voice"], cfg.get("tts_default_voice"))
        if not speaker and m["needs_voice"]:
            # Pocket TTS has no voice of its own: borrow any saved clip, so an
            # OpenAI client sending "alloy" still gets speech.
            saved = list_voices(vdir)
            speaker = resolve_voice(vdir, saved[0]) if saved else None
            if not speaker:
                raise TtsError(f"{m['engine']} needs a voice clip: record or upload one "
                               "on the Voice tab first.", 400)
        with self.lock:
            self.busy = True
            tmp = tempfile.mkdtemp(prefix="lf-tts-")
            self.last_tmp = tmp
            try:
                out = os.path.join(tmp, "speech.wav")
                cmd = build_cmd(st["binary"], m["model"], m["mmproj"], req["text"],
                                out, req["lang"] if m["langs"] else None, speaker)
                try:
                    r = self._run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=TIMEOUT, creationflags=_NO_WINDOW)
                except subprocess.TimeoutExpired:
                    raise TtsError(f"llama-tts took longer than {TIMEOUT}s")
                except OSError as e:
                    raise TtsError(f"could not start llama-tts: {e}")
                if r.returncode != 0 or not os.path.isfile(out):
                    log = _tail((r.stderr or "") + "\n" + (r.stdout or ""))
                    raise TtsError(f"llama-tts exited with {r.returncode}\n{log}".rstrip())
                with open(out, "rb") as f:
                    data = f.read()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                self.busy = False
        return wav_to_pcm(data) if req["format"] == "pcm" else data
