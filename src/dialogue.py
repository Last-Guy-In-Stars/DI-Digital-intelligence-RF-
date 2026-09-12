import threading

import numpy as np

CYAN, YELLOW, DIM, RESET = "\033[36m", "\033[33m", "\033[2m", "\033[0m"
SAMPLE_RATE = 16000


class MicStream:
    def __init__(self, voice, dcfg):
        self.voice = voice
        self.block = dcfg.get("block_seconds", 0.3)
        self.device = voice._find_input_device()
        self.noise_floor = 250.0
        self._bc_playing = threading.Event()

    def update_floor(self, level):
        self.noise_floor = 0.7 * self.noise_floor + 0.3 * max(level, 150.0)

    def read(self):
        import sounddevice as sd

        chunk = sd.rec(
            int(SAMPLE_RATE * self.block),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=self.device,
        )
        sd.wait()
        return chunk, float(np.abs(chunk).mean())

    def onset_level(self):
        return max(350.0, self.noise_floor * 1.8)

    def hold_level(self):
        return max(220.0, self.noise_floor * 1.25)


class TurnDetector:
    def __init__(self, dcfg):
        self.short = dcfg.get("silence_short", 0.8)
        self.medium = dcfg.get("silence_medium", 1.8)
        self.long = dcfg.get("silence_long", 2.5)

    def silence_timeout(self, speech_sec):
        if speech_sec < 2.0:
            return self.short
        if speech_sec < 5.0:
            return self.medium
        return self.long


class DialogueEngine:
    def __init__(self, brain, voice, cfg):
        self.brain = brain
        self.voice = voice
        self.dcfg = cfg.data.get("dialogue", {})
        self.mic = MicStream(voice, self.dcfg)
        self.detector = TurnDetector(self.dcfg)

    @property
    def block(self):
        return self.mic.block

    def listen_turn(self, wait_speech=10.0, max_speech=None, interrupt=None):
        max_speech = max_speech or self.dcfg.get("max_speech_sec", 95.0)
        bc_after = self.dcfg.get("backchannel_after_sec", 20.0)
        bc_every = self.dcfg.get("backchannel_every_sec", 15.0)
        chunks = []
        pre = None
        speech = False
        speech_sec = 0.0
        silence = 0.0
        waited = 0.0
        last_bc = 0.0
        while True:
            if interrupt is not None and interrupt():
                return None, True
            chunk, level = self.mic.read()
            if not speech:
                self.mic.update_floor(level)
                if level > self.mic.onset_level():
                    speech = True
                    if pre is not None:
                        chunks.append(pre)
                    chunks.append(chunk)
                    silence = 0.0
                else:
                    pre = chunk
                    waited += self.block
                    if waited >= wait_speech:
                        return None, False
                continue
            if not self.mic._bc_playing.is_set():
                chunks.append(chunk)
            speech_sec += self.block
            if level > self.mic.hold_level():
                silence = 0.0
            else:
                silence += self.block
                if silence >= self.detector.silence_timeout(speech_sec):
                    break
            if (
                bc_every > 0
                and speech_sec > bc_after
                and speech_sec - last_bc >= bc_every
            ):
                last_bc = speech_sec
                dur = self.voice.play_filler("aga")
                if dur > 0:
                    self.mic._bc_playing.set()
                    threading.Timer(dur, self.mic._bc_playing.clear).start()
            if speech_sec >= max_speech:
                break
        return self.voice.transcribe_chunks(chunks), False

    def speak_turn(self, text):
        self.voice.speak_async(text)
        holdoff = self.dcfg.get("bargein_holdoff", 0.4)
        min_speech = self.dcfg.get("bargein_min_speech", 0.7)
        echo_level = 0.0
        run = 0.0
        elapsed = 0.0
        collecting = False
        barg_chunks = []
        while self.voice.speaking():
            chunk, level = self.mic.read()
            elapsed += self.block
            if elapsed < holdoff:
                echo_level = max(echo_level, level)
                continue
            threshold = max(self.mic.onset_level(), echo_level * 1.6)
            if collecting or level > threshold:
                collecting = True
                barg_chunks.append(chunk)
            if level > threshold:
                run += self.block
                if run >= min_speech:
                    self.voice.stop_speaking()
                    barg_chunks.extend(self._tail_until_silence())
                    return self.voice.transcribe_chunks(barg_chunks)
            else:
                run = 0.0
        return None

    def _tail_until_silence(self, silence_timeout=1.5, max_sec=8.0):
        chunks = []
        silence = 0.0
        total = 0.0
        hold = self.mic.hold_level()
        while total < max_sec:
            chunk, level = self.mic.read()
            total += self.block
            if level > hold:
                silence = 0.0
                chunks.append(chunk)
            else:
                silence += self.block
                if silence >= silence_timeout and chunks:
                    break
                if silence >= silence_timeout + 1.0:
                    break
        return chunks

    def think_with_filler(self, user_text, bargein=None):
        filler_after = self.dcfg.get("filler_after_sec", 4.0)
        box = {}

        def work():
            if bargein is None:
                box["r"] = self.brain.respond(user_text)
            else:
                box["r"] = self.brain.respond_bargein(bargein, user_text)

        t = threading.Thread(target=work)
        t.start()
        t.join(filler_after)
        if t.is_alive():
            self.voice.play_filler("hmm")
            print(f"{DIM}…думает…{RESET}")
            t.join()
        return box["r"]

    @staticmethod
    def _sane_transcript(text):
        if not text:
            return False
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False
        return all(
            ch.isascii() or "\u0400" <= ch <= "\u04FF" for ch in letters
        )

    def voice_session(self, lines=None):
        if not self.voice.can_listen():
            print(f"{DIM}голос недоступен (слух не настроен){RESET}")
            return
        self.voice.warm_fillers()
        print(
            f"{DIM}голосовой диалог: говори после «…слушаю…»; "
            f"10 сек тишины — выход в текст{RESET}"
        )
        import queue as _queue

        def has_typed_text():
            if lines is None:
                return False
            keep = []
            found = False
            while True:
                try:
                    item = lines.get_nowait()
                except _queue.Empty:
                    break
                if item is not None and item.strip():
                    found = True
                keep.append(item)
            for item in keep:
                lines.put(item)
            return found

        while True:
            print(f"{DIM}…слушаю…{RESET}")
            try:
                heard, interrupted = self.listen_turn(
                    wait_speech=10.0, interrupt=has_typed_text
                )
            except KeyboardInterrupt:
                print()
                return
            if interrupted:
                print(f"{DIM}(ты начал печатать — слушаю текстом){RESET}")
                return
            if not heard:
                break
            if not self._sane_transcript(heard):
                print(f"{DIM}не расслышала, повтори{RESET}")
                continue
            print(f"{YELLOW}Ты (голос):{RESET} {heard}")
            answer, meta = self.think_with_filler(heard)
            print(f"{CYAN}Leta:{RESET} {answer}\n")
            barged = self.speak_turn(answer)
            if barged:
                print(f"{YELLOW}Ты (перебил):{RESET} {barged}")
                answer2, _ = self.think_with_filler(barged, bargein=answer)
                print(f"{CYAN}Leta:{RESET} {answer2}\n")
                self.speak_turn(answer2)
        if lines is not None:
            drained = []
            while True:
                try:
                    item = lines.get_nowait()
                except _queue.Empty:
                    break
                if item is not None and item.strip():
                    drained.append(item)
            for item in drained:
                lines.put(item)
        print(f"{DIM}(голосовая пауза — она рядом, Enter — снова говорить){RESET}")
