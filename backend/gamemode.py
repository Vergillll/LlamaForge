"""Game mode: while a game runs, unload every model except the ones the user
keeps, and bring them back when it closes.

A "game" is a process running from a game library folder (Steam, Epic, GOG,
Xbox, Riot, EA, Ubisoft), an executable the user named, or, on Windows, an app
holding exclusive full-screen Direct3D. Detection only runs while the setting
is on. It is debounced both ways so a launcher splash or a loading screen
doesn't flap models in and out.
"""
import os
import threading
import time

import osplat

# folder that holds games -> the game's own folder is the next path part
_LIBRARIES = ("steamapps/common/", "epic games/", "gog galaxy/games/", "gog games/",
              "xboxgames/", "riot games/", "ea games/", "ubisoft game launcher/games/")
# launchers, shared runtimes and background services living in those folders
_NOT_GAMES = {"steamworks shared", "launcher", "epic online services", "directxredist",
              "riot client", "steam controller configs"}

FULLSCREEN = "a full-screen game"


def game_from_path(path):
    """The game a process path belongs to, or None."""
    if not path:
        return None
    p = path.replace("\\", "/")
    low = p.lower()
    for lib in _LIBRARIES:
        i = low.find("/" + lib)
        if i < 0:
            continue
        rest = p[i + 1 + len(lib):]
        name = rest.split("/", 1)[0]
        if "/" in rest and name and name.lower() not in _NOT_GAMES:
            return name
    return None


def _exe_key(name):
    n = os.path.basename(name.strip().replace("\\", "/")).lower()
    return n[:-4] if n.endswith(".exe") else n


def find_game(paths, extra=(), fullscreen=False):
    """First game among running process paths, else the full-screen fallback."""
    wanted = {_exe_key(x) for x in extra if x and x.strip()}
    for path in paths:
        if not path:
            continue
        if wanted and _exe_key(path) in wanted:
            return os.path.basename(path.replace("\\", "/"))
        name = game_from_path(path)
        if name:
            return name
    return FULLSCREEN if fullscreen else None


def detect(cfg):
    """Look at the machine now. Never raises."""
    try:
        paths = osplat.process_image_paths()
    except Exception:
        paths = []
    try:
        full = osplat.fullscreen_d3d()
    except Exception:
        full = False
    return find_game(paths, cfg.get("game_mode_games") or (), full)


class GameMode:
    def __init__(self, cfg, detect_fn, loaded, unload, load, on_hits=2, off_misses=3):
        self._cfg, self._detect = cfg, detect_fn
        self._loaded, self._unload, self._load = loaded, unload, load
        self.on_hits, self.off_misses = on_hits, off_misses
        self._lock = threading.Lock()
        self._hits = self._misses = 0
        self._active = False
        self._enabled = False
        self._game = None
        self._parked = []          # what we unloaded, in order, to bring back
        self._since = None
        self._thread = None

    # -- one look ---------------------------------------------------------
    def tick(self):
        with self._lock:
            c = self._cfg()
            self._enabled = bool(c.get("game_mode"))
            if not self._enabled:
                self._hits = self._misses = 0
                if self._active:
                    self._end(c)
                return
            game = self._detect(c)
            if game:
                self._hits, self._misses = self._hits + 1, 0
            else:
                self._hits, self._misses = 0, self._misses + 1
            if game and (self._active or self._hits >= self.on_hits):
                if not self._active:
                    self._active, self._since = True, time.time()
                self._game = game
                self._enforce(c)
            elif self._active and self._misses >= self.off_misses:
                self._end(c)

    def safe_tick(self):
        try:
            self.tick()
        except Exception:
            pass

    def _enforce(self, c):
        keep = set(c.get("game_mode_keep") or ())
        for mid in self._loaded():
            if mid in keep:
                continue
            if self._unload(mid) and mid not in self._parked:
                self._parked.append(mid)

    def _end(self, c):
        parked, self._parked = self._parked, []
        self._active, self._game, self._since = False, None, None
        if c.get("game_mode_restore", True):
            for mid in parked:
                try:
                    self._load(mid)
                except Exception:
                    pass

    def status(self):
        with self._lock:
            return {"enabled": self._enabled, "active": self._active, "game": self._game,
                    "parked": list(self._parked), "since": self._since}

    # -- background loop --------------------------------------------------
    def start(self, interval=5.0):
        if self._thread:
            return

        def loop():
            while True:
                self.safe_tick()
                time.sleep(interval)
        self._thread = threading.Thread(target=loop, daemon=True, name="game-mode")
        self._thread.start()
