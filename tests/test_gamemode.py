"""Game mode: unload models while a game runs, bring them back after."""
import conftest_paths  # noqa: F401
import unittest

import gamemode


class GameFromPathTest(unittest.TestCase):
    def test_library_folders_name_the_game(self):
        cases = {
            r"D:\SteamLibrary\steamapps\common\Cyberpunk 2077\bin\x64\Cyberpunk2077.exe": "Cyberpunk 2077",
            r"C:\Program Files\Epic Games\Fortnite\FortniteGame\Binaries\Win64\x.exe": "Fortnite",
            r"C:\XboxGames\Forza Horizon 5\Content\ForzaHorizon5.exe": "Forza Horizon 5",
            r"C:\Riot Games\VALORANT\live\VALORANT.exe": "VALORANT",
            "/home/u/.local/share/Steam/steamapps/common/Hades/Hades": "Hades",
        }
        for path, name in cases.items():
            self.assertEqual(gamemode.game_from_path(path), name, path)

    def test_launchers_and_shared_tools_are_not_games(self):
        for path in (r"C:\Program Files (x86)\Steam\steam.exe",
                     r"C:\Program Files (x86)\Steam\steamapps\common\Steamworks Shared\_CommonRedist\x.exe",
                     r"C:\Program Files (x86)\Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe",
                     r"C:\Program Files (x86)\Epic Games\Epic Online Services\EpicOnlineServices.exe",
                     r"C:\Riot Games\Riot Client\RiotClientServices.exe",
                     r"C:\Windows\explorer.exe", "", None):
            self.assertIsNone(gamemode.game_from_path(path), path)


class FindGameTest(unittest.TestCase):
    def test_a_library_process(self):
        paths = [r"C:\Windows\explorer.exe", r"E:\Games\steamapps\common\Hades II\Hades2.exe"]
        self.assertEqual(gamemode.find_game(paths), "Hades II")

    def test_user_named_executables_match_with_or_without_exe(self):
        paths = [r"C:\Games\Elden Ring\Game\eldenring.exe"]
        self.assertEqual(gamemode.find_game(paths, extra=["EldenRing.exe"]), "eldenring.exe")
        self.assertEqual(gamemode.find_game(paths, extra=["eldenring"]), "eldenring.exe")

    def test_full_screen_counts_when_nothing_else_does(self):
        self.assertEqual(gamemode.find_game([], fullscreen=True), "a full-screen game")
        self.assertIsNone(gamemode.find_game([r"C:\Windows\explorer.exe"]))


class Harness:
    def __init__(self, keep=(), restore=True):
        self.cfg = {"game_mode": True, "game_mode_keep": list(keep), "game_mode_restore": restore}
        self.game = None
        self.loaded = ["big", "small"]
        self.calls = []

    def unload(self, mid):
        self.calls.append(("unload", mid))
        self.loaded.remove(mid)
        return True

    def load(self, mid):
        self.calls.append(("load", mid))
        self.loaded.append(mid)
        return True

    def mode(self):
        return gamemode.GameMode(lambda: self.cfg, lambda c: self.game, lambda: list(self.loaded),
                                 self.unload, self.load, on_hits=2, off_misses=3)


class ControllerTest(unittest.TestCase):
    def test_needs_two_sightings_then_unloads_all_but_the_exceptions(self):
        h = Harness(keep=["small"]); gm = h.mode()
        h.game = "Hades"
        gm.tick()
        self.assertEqual(h.calls, [])                       # one sighting is not enough
        gm.tick()
        self.assertEqual(h.calls, [("unload", "big")])
        self.assertEqual(h.loaded, ["small"])
        st = gm.status()
        self.assertTrue(st["active"])
        self.assertEqual((st["game"], st["parked"]), ("Hades", ["big"]))

    def test_a_model_loaded_mid_game_is_unloaded_too(self):
        h = Harness(); gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        h.loaded.append("sneaky")
        gm.tick()
        self.assertNotIn("sneaky", h.loaded)
        self.assertEqual(gm.status()["parked"], ["big", "small", "sneaky"])

    def test_models_come_back_after_three_misses(self):
        h = Harness(); gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        h.game = None
        gm.tick(); gm.tick()
        self.assertEqual(h.loaded, [])                      # not yet: could be a loading screen
        gm.tick()
        self.assertEqual(h.loaded, ["big", "small"])
        self.assertFalse(gm.status()["active"])
        self.assertEqual(gm.status()["parked"], [])

    def test_a_flicker_does_not_end_it(self):
        h = Harness(); gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        h.game = None; gm.tick(); gm.tick()
        h.game = "Hades"; gm.tick()
        h.game = None; gm.tick(); gm.tick()
        self.assertTrue(gm.status()["active"])

    def test_restore_off_leaves_them_unloaded(self):
        h = Harness(restore=False); gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        h.game = None; gm.tick(); gm.tick(); gm.tick()
        self.assertEqual(h.loaded, [])
        self.assertNotIn("load", [c[0] for c in h.calls])

    def test_turning_it_off_mid_game_brings_them_back(self):
        h = Harness(); gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        h.cfg["game_mode"] = False
        gm.tick()
        self.assertEqual(h.loaded, ["big", "small"])
        self.assertFalse(gm.status()["active"])

    def test_off_means_no_detection_at_all(self):
        h = Harness(); h.cfg["game_mode"] = False
        gm = gamemode.GameMode(lambda: h.cfg, lambda c: self.fail("detected while off"),
                               lambda: [], h.unload, h.load)
        gm.tick()
        self.assertEqual(gm.status()["enabled"], False)

    def test_a_failed_unload_is_not_restored_later(self):
        h = Harness(); gm = h.mode()
        h.unload = lambda mid: False
        gm = h.mode()
        h.game = "Hades"; gm.tick(); gm.tick()
        self.assertEqual(gm.status()["parked"], [])

    def test_errors_in_one_tick_do_not_kill_the_loop(self):
        h = Harness(); gm = h.mode()

        def boom():
            raise OSError("router down")
        gm._loaded = boom
        h.game = "Hades"; gm.tick(); gm.safe_tick()        # no exception escapes
        self.assertTrue(gm.status()["active"])


class RouteWiringTest(unittest.TestCase):
    def setUp(self):
        import os, tempfile, config
        self.config = config
        self._saved = config.CONFIG
        config.CONFIG = os.path.join(tempfile.mkdtemp(), "config.json")

    def tearDown(self):
        self.config.CONFIG = self._saved

    def test_settings_are_writable_and_public(self):
        import routes
        code, out = routes.post_config(routes.Req({
            "game_mode": True, "game_mode_keep": ["embed"], "game_mode_games": ["eldenring.exe"],
            "game_mode_restore": False}))
        self.assertEqual(code, 200)
        self.assertNotIn("rejected", out)
        self.assertEqual(out["config"]["game_mode_keep"], ["embed"])
        self.assertIs(out["config"]["game_mode_restore"], False)

    def test_defaults(self):
        c = self.config.load()
        self.assertEqual((c["game_mode"], c["game_mode_keep"], c["game_mode_restore"]), (False, [], True))

    def test_loaded_and_unload_go_through_the_router(self):
        import routes
        from unittest import mock
        models = {"data": [{"id": "a", "status": {"value": "loaded"}},
                           {"id": "b", "status": {"value": "unloaded"}},
                           {"id": "default", "status": {"value": "loaded"}}]}
        calls = []

        def fake_router(path, method="GET", body=None, timeout=30):
            calls.append((path, method, body))
            return (200, models) if path == "/models" else (200, {"success": True})
        with mock.patch.object(routes, "router", fake_router), \
             mock.patch.object(routes.PROCS, "status", return_value={}), \
             mock.patch.object(routes, "_slots_on", return_value=False), \
             mock.patch.object(routes.PROCS, "has", return_value=False):
            self.assertEqual(routes._game_loaded(), ["a"])
            self.assertTrue(routes._game_unload("a"))
        self.assertIn(("/models/unload", "POST", {"model": "a"}), calls)


if __name__ == "__main__":
    unittest.main()
