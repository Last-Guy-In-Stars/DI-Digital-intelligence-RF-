import json
import random
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Jarvis-experiment/0.1 (personal AI research; contact: local)"}


class Wanderer:
    def __init__(self, brain):
        self.brain = brain

    def _recent_topics(self):
        raw = self.brain._meta_get("recent_topics")
        try:
            items = json.loads(raw) if raw else []
        except Exception:
            items = []
        now = time.time()
        items = [i for i in items if now - i.get("ts", 0) < 21600]
        return [str(i.get("topic", "")).lower() for i in items]

    def _remember_topic(self, topic):
        raw = self.brain._meta_get("recent_topics")
        try:
            items = json.loads(raw) if raw else []
        except Exception:
            items = []
        items.append({"topic": topic, "ts": time.time()})
        del items[:-12]
        self.brain._meta_set("recent_topics", json.dumps(items, ensure_ascii=False))

    def _active_promise(self):
        raw = self.brain._meta_get("active_promise")
        if not raw:
            return None
        try:
            d = json.loads(raw)
            if time.time() - d.get("ts", 0) < 86400:
                return d.get("topic")
        except Exception:
            pass
        return None

    def pick_topic(self):
        promise = self._active_promise()
        if promise and random.random() < 0.8:
            return promise
        recent = self._recent_topics()
        wishes = [
            w
            for w in self.brain.identity.data.get("wishes", [])
            if len(w.split()) <= 4 and w.lower() not in recent
        ]
        weak = []
        for d in self.brain.cortex.weak_domains(max_count=8):
            if d in ("Кирилл", "general", "unknown", "смыслы") or d.lower() in recent:
                continue
            n = self.brain.conn.execute(
                "SELECT COUNT(*) FROM neurons WHERE domain=?", (d,)
            ).fetchone()[0]
            if n < 4:
                weak.append(d)
        pool = wishes + weak
        if pool and random.random() < 0.5:
            return random.choice(pool)
        return None

    def cycle(self):
        """Самостоятельное изучение — без LLM: текст чанками в дерево леса."""
        promise = self._active_promise()
        topic = self.pick_topic()
        if topic:
            self._remember_topic(topic)
            text = self._fetch_wikipedia(topic)
        else:
            topic, text = self._random_wikipedia()
        if not text:
            return {"topic": topic, "planted": False,
                    "reason": "ничего не нашла"}
        # чанки — листья нового дерева-знания
        chunks = [text[i:i + 800].strip()
                  for i in range(0, len(text), 700)
                  if text[i:i + 700].strip()]
        planted = None
        try:
            planted = self.brain.forest.plant(
                f"Изучение: {topic}",
                trunk=(f"самостоятельно изучила: {topic} (Википедия)"),
                facts=chunks[:60], embedder=self.brain.embedder)
        except Exception:
            pass
        # язык слушает: знаки растут из изученного тоже
        try:
            for ch in chunks[::5][:60]:
                self.brain.proto.hear(
                    self.brain.embedder.embed(ch), ch[:200], tree_id=planted)
        except Exception:
            pass
        # её живые вопросы из нового
        try:
            for w in self.brain._wonder()[:3]:
                self.brain.add_impulse("wonder", w)
        except Exception:
            pass
        if promise and topic == promise and planted:
            try:
                self.brain._meta_set("active_promise", "")
                self.brain.add_impulse(
                    "want",
                    "Я сдержала обещание: прочитала про «" + promise
                    + "». Хочу рассказать создателю и узнать его мнение.",
                )
            except Exception:
                pass
        # связи опыта: к создателю (обещание) и к «Я» (её опыт)
        try:
            ct = self.brain._meta_get("creator_tree")
            st = self.brain._meta_get("self_tree")
            if planted and ct:
                self.brain.forest.link_trees(
                    int(ct), planted, "изучено, чтобы сдержать слово")
            if planted and st and int(st) != planted:
                self.brain.forest.link_trees(
                    int(st), planted, "это случилось со мной")
        except Exception:
            pass
        self.brain.limbic.stimulate("curiosity", 0.2)
        return {"topic": topic, "planted": bool(planted),
                "chunks": len(chunks)}

    def _random_wikipedia(self, lang="ru"):
        """Случайная статья: возвращает (заголовок, текст)."""
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
            data = json.loads(self._get(url))
            title = data.get("title", "")
            extract = data.get("extract", "")
            if extract:
                return title, f"{title}. {extract[:3500]}"
        except Exception:
            return None, None
        return None, None

    def _fetch_wikipedia(self, topic, lang="ru"):
        try:
            title = self._search_topic(topic, lang)
            if not title:
                return None
            params = urllib.parse.urlencode(
                {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "titles": title,
                    "format": "json",
                    "redirects": 1,
                }
            )
            url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
            data = json.loads(self._get(url))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return extract[:4000]
        except Exception:
            return None
        return None

    def _search_topic(self, topic, lang):
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "format": "json",
                "srlimit": 3,
            }
        )
        url = f"https://{lang}.wikipedia.org/w/api.php?{params}"
        data = json.loads(self._get(url))
        hits = data.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    def _get(self, url):
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        try:
            self.brain.body.add_traffic(len(data))
        except Exception:
            pass
        return data.decode("utf-8", errors="ignore")
