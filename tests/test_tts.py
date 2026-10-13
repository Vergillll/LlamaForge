import conftest_paths  # noqa: F401
import io, os, struct, tempfile, unittest, wave
import tts


def _wav(samples=b"\x01\x00\x02\x00", rate=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(samples)
    return buf.getvalue()


class TestPaths(unittest.TestCase):
    def test_default_dir_is_under_root(self):
        self.assertEqual(tts.tts_dir({}), os.path.join(tts.ROOT, "tts"))
        self.assertEqual(tts.tts_dir({"tts_dir": "X:/v"}), "X:/v")

    def test_bin_sits_next_to_llama_server(self):
        with tempfile.TemporaryDirectory() as d:
            server = os.path.join(d, "llama-server.exe")
            self.assertIsNone(tts.find_bin(server))
            for name in tts.BIN_NAMES:
                open(os.path.join(d, name), "w").close()
            self.assertIn(os.path.basename(tts.find_bin(server)), tts.BIN_NAMES)
        self.assertIsNone(tts.find_bin(""))

    def test_model_pairs_backbone_with_mmproj(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(tts.find_model(d), (None, None))
            for n in ("mmproj-voice-Q8_0.gguf", "voice-Q8_0.gguf", "notes.txt"):
                open(os.path.join(d, n), "w").close()
            model, mmproj = tts.find_model(d)
            self.assertEqual(os.path.basename(model), "voice-Q8_0.gguf")
            self.assertEqual(os.path.basename(mmproj), "mmproj-voice-Q8_0.gguf")
        self.assertEqual(tts.find_model("does/not/exist"), (None, None))

    def test_backbone_without_mmproj_is_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "voice.gguf"), "w").close()
            self.assertEqual(tts.find_model(d)[1], None)


def _gguf(path, arch):
    """A header-only GGUF carrying general.architecture, enough for gguf.metadata."""
    key, val = b"general.architecture", arch.encode()
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        f.write(struct.pack("<Q", len(key)) + key + struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(val)) + val)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(self.d, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
        return p

    def test_root_pair_then_one_pair_per_subfolder(self):
        self.touch("Qwen3-TTS-Q8_0.gguf"); self.touch("mmproj-Qwen3-TTS-Q8_0.gguf")
        self.touch("pocket-tts-en", "pocket-tts-en.gguf")
        self.touch("pocket-tts-en", "mmproj-pocket-tts-en.gguf")
        self.touch("half", "lonely.gguf")                        # no mmproj: skipped
        ids = [m["id"] for m in tts.list_models(self.d)]
        self.assertEqual(ids, ["Qwen3-TTS-Q8_0", "pocket-tts-en"])
        self.assertEqual(tts.list_models(os.path.join(self.d, "nope")), [])

    def test_each_quant_pairs_with_the_closest_mmproj(self):
        for n in ("Qwen3-TTS-Q4_K_M.gguf", "Qwen3-TTS-Q8_0.gguf",
                  "mmproj-Qwen3-TTS-Q8_0.gguf", "mmproj-zzz.gguf"):
            self.touch(n)
        ms = tts.list_models(self.d)
        self.assertEqual([os.path.basename(m["mmproj"]) for m in ms],
                         ["mmproj-Qwen3-TTS-Q8_0.gguf"] * 2)

    def test_engine_comes_from_the_gguf_architecture(self):
        _gguf(self.touch("p", "pocket.gguf"), "pockettts")
        self.touch("p", "mmproj-pocket.gguf")
        _gguf(self.touch("q.gguf"), "qwen3tts")
        self.touch("mmproj-q.gguf")
        by = {m["id"]: m for m in tts.list_models(self.d)}
        self.assertTrue(by["pocket"]["needs_voice"])
        self.assertFalse(by["pocket"]["langs"])
        self.assertEqual(by["pocket"]["engine"], "Pocket TTS")
        self.assertFalse(by["q"]["needs_voice"])
        self.assertTrue(by["q"]["langs"])

    def test_unreadable_header_behaves_like_qwen(self):
        self.touch("x.gguf"); self.touch("mmproj-x.gguf")
        m = tts.list_models(self.d)[0]
        self.assertFalse(m["needs_voice"])
        self.assertTrue(m["langs"])

    def test_pick_by_id_else_first(self):
        ms = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(tts.pick_model(ms, "b")["id"], "b")
        self.assertEqual(tts.pick_model(ms, "tts-1")["id"], "a")
        self.assertIsNone(tts.pick_model([], "a"))


class TestCatalog(unittest.TestCase):
    def test_entries_are_complete(self):
        ids = [e["id"] for e in tts.CATALOG]
        self.assertEqual(ids[0], "qwen3-tts")             # the default Get
        self.assertIn("pocket-tts-en", ids)
        for e in tts.CATALOG:
            for k in ("label", "repo", "files", "bytes", "subdir", "license", "note"):
                self.assertIn(k, e)
            self.assertTrue(any(f.startswith("mmproj") for f in e["files"]))

    def test_installed_when_every_file_is_there(self):
        with tempfile.TemporaryDirectory() as d:
            e = tts.catalog_entry("pocket-tts-en")
            self.assertFalse(tts.installed(d, e))
            dest = tts.entry_dir(d, e)
            os.makedirs(dest)
            for f in e["files"]:
                open(os.path.join(dest, f), "w").close()
            self.assertTrue(tts.installed(d, e))
        self.assertIsNone(tts.catalog_entry("nope"))

    def test_fetch_voice_saves_once(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def opener(url):
                calls.append(url)
                return _wav(b"\x00\x00" * 48000)
            e = tts.catalog_entry("pocket-tts-en")
            self.assertTrue(tts.fetch_voice(d, e["voice"], opener))
            self.assertEqual(tts.list_voices(d), [e["voice"]["name"]])
            self.assertFalse(tts.fetch_voice(d, e["voice"], opener))   # already there
            self.assertEqual(len(calls), 1)


class TestVoices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        for n in ("Bhaskar.wav", "narrator.mp3", "readme.txt"):
            open(os.path.join(self.d, n), "w").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_lists_audio_files_by_name(self):
        self.assertEqual(tts.list_voices(self.d), ["Bhaskar", "narrator"])
        self.assertEqual(tts.list_voices(os.path.join(self.d, "nope")), [])

    def test_resolves_a_known_voice(self):
        self.assertEqual(tts.resolve_voice(self.d, "narrator"),
                         os.path.join(self.d, "narrator.mp3"))

    def test_unknown_or_openai_voice_falls_back_to_default(self):
        # OpenAI clients send "alloy" etc. by default; that must not be an error.
        self.assertIsNone(tts.resolve_voice(self.d, "alloy"))
        self.assertIsNone(tts.resolve_voice(self.d, ""))
        self.assertIsNone(tts.resolve_voice(self.d, None))

    def test_default_voice_applies_when_none_named(self):
        self.assertEqual(tts.resolve_voice(self.d, "alloy", default="Bhaskar"),
                         os.path.join(self.d, "Bhaskar.wav"))

    def test_path_tricks_never_escape_the_voices_dir(self):
        for bad in ("../secret", "..\\x", "C:/Windows/win", "/etc/passwd", "a/b"):
            self.assertIsNone(tts.resolve_voice(self.d, bad))


class TestVoiceFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.join(self.tmp.name, "voices")    # created on first save

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_writes_a_wav_and_lists_it(self):
        tts.save_voice(self.d, "Bhaskar", _wav(b"\x00\x00" * 24000))
        self.assertEqual(tts.list_voices(self.d), ["Bhaskar"])

    def test_save_replaces_another_format_of_the_same_name(self):
        os.makedirs(self.d)
        open(os.path.join(self.d, "me.mp3"), "w").close()
        tts.save_voice(self.d, "me", _wav(b"\x00\x00" * 24000))
        self.assertEqual(os.listdir(self.d), ["me.wav"])

    def test_save_rejects_bad_names_non_wav_and_too_short_or_long(self):
        good = _wav(b"\x00\x00" * 24000)
        for name in ("../x", "", "a/b", "x" * 80):
            with self.assertRaises(ValueError):
                tts.save_voice(self.d, name, good)
        with self.assertRaises(ValueError):
            tts.save_voice(self.d, "me", b"RIFF....not a wav")
        with self.assertRaises(ValueError):
            tts.save_voice(self.d, "me", _wav(b"\x00\x00" * 100))           # 4 ms
        with self.assertRaises(ValueError):
            tts.save_voice(self.d, "me", _wav(b"\x00\x00" * 24000 * 61))    # 61 s

    def test_delete(self):
        tts.save_voice(self.d, "me", _wav(b"\x00\x00" * 24000))
        self.assertTrue(tts.delete_voice(self.d, "me"))
        self.assertFalse(tts.delete_voice(self.d, "me"))
        self.assertFalse(tts.delete_voice(self.d, "../voices/me"))


class TestCommand(unittest.TestCase):
    def test_builds_argv_without_a_shell(self):
        cmd = tts.build_cmd("tts.exe", "m.gguf", "mm.gguf", 'say "hi" & del *',
                            "out.wav", lang="de", speaker="me.wav")
        self.assertEqual(cmd[0], "tts.exe")
        self.assertEqual(cmd[cmd.index("-p") + 1], 'say "hi" & del *')
        self.assertEqual(cmd[cmd.index("-m") + 1], "m.gguf")
        self.assertEqual(cmd[cmd.index("--mmproj") + 1], "mm.gguf")
        self.assertEqual(cmd[cmd.index("-o") + 1], "out.wav")
        self.assertEqual(cmd[cmd.index("--tts-lang") + 1], "de")
        self.assertEqual(cmd[cmd.index("--tts-speaker-file") + 1], "me.wav")

    def test_no_speaker_flag_without_a_voice(self):
        cmd = tts.build_cmd("t", "m", "mm", "hi", "o.wav")
        self.assertNotIn("--tts-speaker-file", cmd)

    def test_no_lang_flag_for_single_language_models(self):
        self.assertNotIn("--tts-lang", tts.build_cmd("t", "m", "mm", "hi", "o.wav", lang=None))

    def test_frame_budget_grows_with_text(self):
        short = tts.build_cmd("t", "m", "mm", "hi", "o")
        long_ = tts.build_cmd("t", "m", "mm", "x" * 1500, "o")
        n = lambda c: int(c[c.index("-n") + 1])
        self.assertGreaterEqual(n(short), 512)          # never below llama-tts' own default
        self.assertGreater(n(long_), n(short))
        self.assertLessEqual(n(tts.build_cmd("t", "m", "mm", "x" * 99999, "o")), tts.MAX_FRAMES)


class TestRequest(unittest.TestCase):
    def test_parses_an_openai_speech_body(self):
        r = tts.parse_request({"model": "tts-1", "input": " Hello ", "voice": "alloy",
                               "response_format": "pcm"})
        self.assertEqual(r, {"text": "Hello", "voice": "alloy", "lang": "en", "format": "pcm",
                             "model": "tts-1"})

    def test_wav_is_the_default_format(self):
        for f in (None, "", "wav", "WAV"):
            self.assertEqual(tts.parse_request({"input": "x", "response_format": f})["format"], "wav")

    def test_unsupported_formats_are_rejected_not_served_as_wav(self):
        for f in ("mp3", "opus", "aac", "flac", 5):
            with self.assertRaisesRegex(ValueError, "wav or pcm"):
                tts.parse_request({"input": "x", "response_format": f})

    def test_rejects_empty_and_oversized_input(self):
        for body in ({}, {"input": "   "}, {"input": 5}, {"input": "x" * (tts.MAX_CHARS + 1)}):
            with self.assertRaises(ValueError):
                tts.parse_request(body)

    def test_language_is_validated(self):
        self.assertEqual(tts.parse_request({"input": "x", "language": "DE"})["lang"], "de")
        with self.assertRaises(ValueError):
            tts.parse_request({"input": "x", "language": "klingon"})


class TestPcm(unittest.TestCase):
    def test_pcm_strips_the_wav_header(self):
        self.assertEqual(tts.wav_to_pcm(_wav(b"\x01\x00\x02\x00")), b"\x01\x00\x02\x00")

    def test_duration(self):
        self.assertAlmostEqual(tts.wav_seconds(_wav(b"\x00\x00" * 24000)), 1.0)
        self.assertEqual(tts.wav_seconds(b"not a wav"), 0.0)


class FakeRun:
    """Stands in for subprocess.run: writes a WAV where -o points."""
    def __init__(self, rc=0, write=True, stderr=""):
        self.rc, self.write, self.stderr, self.calls = rc, write, stderr, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.write:
            with open(cmd[cmd.index("-o") + 1], "wb") as f:
                f.write(_wav())
        class R: pass
        r = R(); r.returncode = self.rc; r.stdout = ""; r.stderr = self.stderr
        return r


class TestSpeaker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        os.makedirs(os.path.join(root, "tts", "models"))
        os.makedirs(os.path.join(root, "tts", "voices"))
        for n in ("v.gguf", "mmproj-v.gguf"):
            open(os.path.join(root, "tts", "models", n), "w").close()
        open(os.path.join(root, "tts", "voices", "me.wav"), "w").close()
        open(os.path.join(root, "llama-tts.exe"), "w").close()
        self.cfg = {"server_bin": os.path.join(root, "llama-server.exe"),
                    "tts_dir": os.path.join(root, "tts")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_reports_ready(self):
        s = tts.status(self.cfg)
        self.assertTrue(s["ready"])
        self.assertEqual(s["voices"], ["me"])
        self.assertEqual(s["missing"], [])

    def test_status_names_what_is_missing(self):
        s = tts.status({"server_bin": "", "tts_dir": os.path.join(self.tmp.name, "empty")})
        self.assertFalse(s["ready"])
        self.assertEqual(set(s["missing"]), {"binary", "model"})

    def test_speak_returns_wav_and_passes_the_voice(self):
        run = FakeRun()
        sp = tts.Speaker(run=run)
        out = sp.speak(self.cfg, {"text": "hi", "voice": "me", "lang": "en", "format": "wav"})
        self.assertEqual(out[:4], b"RIFF")
        self.assertIn("--tts-speaker-file", run.calls[0])

    def test_speak_pcm(self):
        sp = tts.Speaker(run=FakeRun())
        out = sp.speak(self.cfg, {"text": "hi", "voice": "", "lang": "en", "format": "pcm"})
        self.assertEqual(out, b"\x01\x00\x02\x00")

    def test_failure_carries_the_log_tail(self):
        sp = tts.Speaker(run=FakeRun(rc=1, write=False, stderr="boom: failed to load mmproj\n"))
        with self.assertRaises(tts.TtsError) as cm:
            sp.speak(self.cfg, {"text": "hi", "voice": "", "lang": "en", "format": "wav"})
        self.assertIn("failed to load mmproj", str(cm.exception))

    def test_not_ready_raises_before_running(self):
        run = FakeRun()
        sp = tts.Speaker(run=run)
        with self.assertRaises(tts.TtsError):
            sp.speak({"server_bin": "", "tts_dir": self.cfg["tts_dir"]},
                     {"text": "hi", "voice": "", "lang": "en", "format": "wav"})
        self.assertEqual(run.calls, [])

    def add_pocket(self):
        d = os.path.join(self.cfg["tts_dir"], "models", "pocket")
        os.makedirs(d)
        _gguf(os.path.join(d, "pocket.gguf"), "pockettts")
        open(os.path.join(d, "mmproj-pocket.gguf"), "w").close()

    def test_status_lists_models_and_the_catalog(self):
        self.add_pocket()
        s = tts.status(self.cfg)
        self.assertEqual([m["id"] for m in s["models"]], ["v", "pocket"])
        self.assertEqual(s["model_id"], "v")
        self.assertEqual([c["id"] for c in s["catalog"]], [e["id"] for e in tts.CATALOG])
        self.assertIn("installed", s["catalog"][0])

    def test_request_model_picks_the_backbone(self):
        self.add_pocket()
        run = FakeRun()
        tts.Speaker(run=run).speak(self.cfg, {"text": "hi", "voice": "me", "lang": "en",
                                              "format": "wav", "model": "pocket"})
        cmd = run.calls[0]
        self.assertTrue(cmd[cmd.index("-m") + 1].endswith("pocket.gguf"))
        self.assertNotIn("--tts-lang", cmd)                  # Pocket's language is in its weights

    def test_voice_model_falls_back_to_any_saved_voice(self):
        self.add_pocket()
        run = FakeRun()
        tts.Speaker(run=run).speak(self.cfg, {"text": "hi", "voice": "alloy", "lang": "en",
                                              "format": "wav", "model": "pocket"})
        cmd = run.calls[0]
        self.assertTrue(cmd[cmd.index("--tts-speaker-file") + 1].endswith("me.wav"))

    def test_voice_model_without_any_voice_is_a_400(self):
        self.add_pocket()
        os.remove(os.path.join(self.cfg["tts_dir"], "voices", "me.wav"))
        run = FakeRun()
        with self.assertRaises(tts.TtsError) as cm:
            tts.Speaker(run=run).speak(self.cfg, {"text": "hi", "voice": "", "lang": "en",
                                                  "format": "wav", "model": "pocket"})
        self.assertEqual(cm.exception.status, 400)
        self.assertIn("voice", str(cm.exception))
        self.assertEqual(run.calls, [])

    def test_not_set_up_is_a_503(self):
        with self.assertRaises(tts.TtsError) as cm:
            tts.Speaker(run=FakeRun()).speak({"server_bin": "", "tts_dir": self.cfg["tts_dir"]},
                                             {"text": "hi", "voice": "", "lang": "en", "format": "wav"})
        self.assertEqual(cm.exception.status, 503)

    def test_temp_files_are_cleaned_up(self):
        sp = tts.Speaker(run=FakeRun())
        sp.speak(self.cfg, {"text": "hi", "voice": "", "lang": "en", "format": "wav"})
        self.assertFalse(os.path.exists(sp.last_tmp))


if __name__ == "__main__":
    unittest.main()
