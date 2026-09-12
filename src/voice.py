import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

from .config import ROOT

SAMPLE_RATE = 16000
SILERO_RATE = 48000
TTS_ALLOWED = re.compile(r"[^\w\s.,!?;:()\-—«»'\"]+", flags=re.UNICODE)

FILLERS = {
    "hmm": "Хм.",
    "aga": "Ага.",
    "ponimau": "Понимаю.",
    "da": "Да-да.",
}


class Voice:
    def __init__(self, cfg):
        v = cfg.data.get("voice", {})
        self.profiles = v.get("profiles", {})
        self.profile = v.get("default_profile", "leta")
        self.voices_dir = ROOT / "tools" / "piper" / "voices"
        self.stt_bin = ROOT / v.get("stt_bin", "")
        self.stt_model = ROOT / v.get("stt_model", "")
        self.input_device = v.get("input_device")
        self.stt_language = v.get("stt_language", "ru")
        self.piper = self._find_piper()
        self._silero_model = None
        self._accentor = None
        self._speak_done = threading.Event()
        self._speak_done.set()
        self._player = None
        self._cancel = False
        self._fillers_dir = ROOT / "tools" / "cache" / "fillers"
        self.muted = False

    def _find_piper(self):
        candidate = Path(sys.executable).parent / "piper"
        if candidate.exists():
            return str(candidate)
        return shutil.which("piper")

    def _parse_profile(self, profile=None):
        spec = self.profiles.get(profile or self.profile, "")
        if ":" in spec:
            engine, name = spec.split(":", 1)
            return engine.strip(), name.strip()
        return "piper", spec

    def set_profile(self, profile):
        if profile in self.profiles:
            self.profile = profile
            return True
        return False

    @staticmethod
    def _clean_for_tts(text):
        text = TTS_ALLOWED.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:900]

    @staticmethod
    def _split_sentences(text):
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _silero(self):
        if self._silero_model is None:
            import silero
            from silero_stress.accentor import load_accentor

            model, _ = silero.silero_tts(
                language="ru", speaker="v5_cis_base", sample_rate=SILERO_RATE
            )
            model.to("cpu")
            self._silero_model = model
            self._accentor = load_accentor("ru")
        return self._silero_model

    @staticmethod
    def _write_wav(path, data, rate):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())

    def _synth_silero(self, text, speaker, out_path):
        import numpy as np

        model = self._silero()
        chunks = []
        for sentence in self._split_sentences(text) or [text]:
            accented = self._accentor(sentence)
            audio = model.apply_tts(
                text=accented, speaker=speaker, sample_rate=SILERO_RATE
            )
            chunks.append((audio.numpy() * 32767).astype(np.int16))
            chunks.append(np.zeros(SILERO_RATE // 4, dtype=np.int16))
        data = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16)
        self._write_wav(out_path, data, SILERO_RATE)

    def _synth_piper(self, text, model_name, out_path):
        onnx = self.voices_dir / f"{model_name}.onnx"
        subprocess.run(
            [self.piper, "--model", str(onnx), "--output_file", out_path],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
            timeout=120,
        )

    @staticmethod
    def _pad_wav(path, before=0.7, after=0.3):
        with wave.open(path, "rb") as r:
            rate = r.getframerate()
            frames = r.readframes(r.getnframes())
        silence_b = b"\x00" * int(rate * before) * 2
        silence_a = b"\x00" * int(rate * after) * 2
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(silence_b + frames + silence_a)

    def can_speak(self):
        engine, name = self._parse_profile()
        if engine == "silero":
            try:
                import silero  # noqa: F401

                return True
            except Exception:
                return False
        return self.piper is not None and (self.voices_dir / f"{name}.onnx").exists()

    def speak(self, text, profile=None):
        text = self._clean_for_tts(text)
        if not text or self.muted:
            return False
        engine, name = self._parse_profile(profile)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            self._synth(text, engine, name, out)
            self._pad_wav(out)
            self._play(out)
            return True
        except Exception:
            return False

    def _synth(self, text, engine, name, out_path):
        if engine == "silero":
            try:
                self._synth_silero(text, name, out_path)
                return
            except Exception:
                if self.piper is None:
                    raise
                name = "ru_RU-irina-medium"
        if self.piper is None:
            raise RuntimeError("no tts engine")
        self._synth_piper(text, name, out_path)

    def speak_async(self, text, profile=None):
        text = self._clean_for_tts(text)
        if not text or self.muted:
            self._speak_done.set()
            return
        engine, name = self._parse_profile(profile)
        self._speak_done.clear()
        self._cancel = False

        def run():
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as f:
                    out = f.name
                self._synth(text, engine, name, out)
                if self._cancel:
                    return
                self._pad_wav(out)
                player = self._player_cmd()
                if player is None:
                    return
                self._player = subprocess.Popen(list(player) + [out])
                self._player.wait()
            except Exception:
                pass
            finally:
                self._speak_done.set()

        threading.Thread(target=run, daemon=True).start()

    def speaking(self):
        return not self._speak_done.is_set()

    def stop_speaking(self):
        self._cancel = True
        player = self._player
        if player is not None and player.poll() is None:
            try:
                player.terminate()
                player.wait(timeout=2)
            except Exception:
                try:
                    player.kill()
                except Exception:
                    pass
        self._speak_done.set()

    @staticmethod
    def _player_cmd():
        for cmd in (("afplay",), ("aplay", "-q"), ("paplay",)):
            if shutil.which(cmd[0]):
                return cmd
        return None

    def warm_fillers(self):
        try:
            self._fillers_dir.mkdir(parents=True, exist_ok=True)
            for key, text in FILLERS.items():
                path = self._fillers_dir / f"{self.profile}_{key}.wav"
                if path.exists():
                    continue
                engine, name = self._parse_profile()
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as f:
                    out = f.name
                self._synth(text, engine, name, out)
                with open(out, "rb") as src, open(path, "wb") as dst:
                    dst.write(src.read())
            return True
        except Exception:
            return False

    def play_filler(self, key):
        path = self._fillers_dir / f"{self.profile}_{key}.wav"
        if not path.exists() or self.muted:
            return 0.0
        try:
            player = self._player_cmd()
            if player is None:
                return 0.0
            subprocess.Popen(list(player) + [str(path)])
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / float(w.getframerate())
        except Exception:
            return 0.0

    def _play(self, wav_path):
        for cmd in (("afplay",), ("aplay", "-q"), ("paplay",)):
            if shutil.which(cmd[0]):
                subprocess.run(list(cmd) + [wav_path], capture_output=True)
                return

    def can_listen(self):
        try:
            import sounddevice  # noqa: F401

            ok_mic = True
        except Exception:
            ok_mic = False
        return ok_mic and self.stt_bin.exists() and self.stt_model.exists()

    def _find_input_device(self):
        try:
            import sounddevice as sd

            if not self.input_device:
                return None
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if (
                    d["max_input_channels"] > 0
                    and self.input_device.lower() in d["name"].lower()
                ):
                    return i
        except Exception:
            pass
        return None

    def transcribe_chunks(self, chunks):
        import numpy as np

        if not chunks:
            return None
        audio = np.concatenate(chunks).flatten().astype("int16")
        if float(np.abs(audio).mean()) < 80:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        self._write_wav(wav_path, audio, SAMPLE_RATE)
        return self.transcribe_file(wav_path)

    def transcribe_file(self, wav_path):
        try:
            out = subprocess.run(
                [
                    str(self.stt_bin),
                    "-m",
                    str(self.stt_model),
                    "-f",
                    wav_path,
                    "-l",
                    self.stt_language,
                    "-nt",
                    "-np",
                ],
                capture_output=True,
                timeout=120,
            )
            text = out.stdout.decode("utf-8", errors="ignore").strip()
            text = re.sub(
                r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]",
                " ",
                text,
            )
            for junk in ("\r", "\n[BLANK_AUDIO]", "[BLANK_AUDIO]"):
                text = text.replace(junk, " ")
            text = re.sub(r"\s+", " ", text).strip()
            return text or None
        except Exception:
            return None
