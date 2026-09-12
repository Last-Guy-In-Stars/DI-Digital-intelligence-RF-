import time

BASE = {"joy": 0.35, "anger": 0.08, "curiosity": 0.55, "boredom": 0.25, "trust": 0.5}
KEYS = ["joy", "anger", "curiosity", "boredom", "trust"]

POSITIVE = {
    "спасибо", "благодарю", "молодец", "класс", "отлично", "супер", "нравишься",
    "хорошо", "good", "great", "thanks", "love", "nice", "perfect",
}
NEGATIVE = {
    "дурак", "тупой", "бесишь", "ненавижу", "ужасно", "плохо", "бред", "чушь",
    "ерунда", "stupid", "bad", "hate", "wrong", "nonsense",
}
QUESTION_MARKS = ("почему", "как ", "зачем", "что если", "объясни", "расскажи", "why", "how", "what if")


class Limbic:
    def __init__(self, conn, homeostasis_rate):
        self.conn = conn
        self.rate = homeostasis_rate
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS emotion_log(
              ts REAL, joy REAL, anger REAL, curiosity REAL,
              boredom REAL, trust REAL, trigger TEXT
            );
            """
        )
        self.state = dict(BASE)
        self._load_last()

    def _load_last(self):
        row = self.conn.execute(
            "SELECT joy, anger, curiosity, boredom, trust FROM emotion_log"
            " ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row:
            self.state = dict(zip(KEYS, row))

    def react(self, user_text, is_novel, is_repetitive):
        words = set(user_text.lower().split())
        lower = user_text.lower()
        trigger = []
        if words & POSITIVE:
            self.state["joy"] += 0.15
            self.state["trust"] += 0.1
            trigger.append("praise")
        if words & NEGATIVE:
            self.state["anger"] += 0.18
            self.state["trust"] -= 0.12
            trigger.append("insult")
        if is_novel:
            self.state["curiosity"] += 0.2
            self.state["boredom"] -= 0.15
            trigger.append("novelty")
        if is_repetitive:
            self.state["boredom"] += 0.12
            trigger.append("repetition")
        if any(q in lower for q in QUESTION_MARKS):
            self.state["curiosity"] += 0.08
            trigger.append("question")
        self._homeostasis()
        self._clamp()
        self._log(",".join(trigger))
        return dict(self.state)

    def stimulate(self, kind, amount):
        if kind in self.state:
            self.state[kind] = min(1.0, self.state[kind] + amount)
            self._clamp()
            self._log("stimulate:" + kind)

    def drift(self):
        for _ in range(5):
            self._homeostasis()
        self._clamp()
        self._log("sleep")

    def _homeostasis(self):
        for k in KEYS:
            self.state[k] += (BASE[k] - self.state[k]) * self.rate

    def _clamp(self):
        for k in KEYS:
            self.state[k] = max(0.0, min(1.0, self.state[k]))

    def _log(self, trigger):
        self.conn.execute(
            "INSERT INTO emotion_log VALUES (?,?,?,?,?,?,?)",
            (time.time(), *[self.state[k] for k in KEYS], trigger),
        )
        self.conn.commit()

    def signals(self):
        s = self.state
        return (
            f"радость {s['joy']:.2f}, гнев {s['anger']:.2f}, "
            f"любопытство {s['curiosity']:.2f}, скука {s['boredom']:.2f}, "
            f"доверие {s['trust']:.2f}"
        )

    def describe(self):
        s = self.state
        parts = []
        if s["anger"] > 0.4:
            parts.append("раздражена")
        if s["joy"] > 0.5:
            parts.append("радостна")
        if s["curiosity"] > 0.6:
            parts.append("очень любопытна")
        if s["boredom"] > 0.5:
            parts.append("тебе скучно")
        if s["trust"] > 0.7:
            parts.append("чувствуешь доверие к собеседнику")
        return ", ".join(parts) if parts else "спокойна"

    def temperature(self):
        s = self.state
        return 0.65 + s["anger"] * 0.25 + s["curiosity"] * 0.1 - s["boredom"] * 0.1
