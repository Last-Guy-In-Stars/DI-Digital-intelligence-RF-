"""Мозг Leta: лес нейронов-деревьев + протоязык SNN + гортань."""
import json
import random
import sqlite3
import threading
import time

import numpy as np
from urllib.parse import urlparse

from .body import Body
from .brother import Brother
from .config import ROOT, load
from .cortex import Cortex
from .embeddings import Embedder
from .generator import Generator
from .hippocampus import Hippocampus
from .identity import Identity
from .limbic import Limbic
from .thalamus import Thalamus

import re as _re

SCHEMA = """
CREATE TABLE IF NOT EXISTS impulses(
  id INTEGER PRIMARY KEY, ts REAL, kind TEXT, content TEXT, delivered INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dreams(
  id INTEGER PRIMARY KEY, ts REAL, text TEXT, mood TEXT DEFAULT 'neutral'
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

INTENT_EXEMPLARS = {
    "study": [
        "что ты изучала", "что нового изучила", "что читала последнее",
        "что нового узнала", "чем занималась", "что изучаешь сейчас",
        "какие новые знания", "расскажи чему научилась",
    ],
    "self": [
        "кто ты", "что ты такое", "из чего ты состоишь",
        "как ты устроена", "что ты умеешь", "ты личность или программа",
        "расскажи о себе", "что ты знаешь о себе",
    ],
    "act": [
        "скачай книгу", "скачай и прочитай книгу", "загрузи книгу",
        "найди и скачай книгу", "достань полный текст", "прочитай книгу",
    ],
    "question": [
        "что такое софон", "что такое дождь", "кто такой человек",
        "расскажи про гидратацию", "объясни что такое квант",
        "что значит это слово", "определи понятие",
    ],
    "shelf": [
        "я положил книгу на полку", "положил на твою полку книгу",
        "прочитай с полки", "книга лежит на полке", "книга на полке",
    ],
    "video": [
        "посмотри видео", "посмотреть видео", "видео по ссылке",
    ],
    "approve": [
        "да", "установи", "разрешаю", "ставь", "согласен",
        "можно", "давай", "подтверждаю",
    ],
    "use_tool": [
        "воспользуйся curl", "используй wget", "попробуй через ffmpeg",
    ],
}

STOPWORDS = {
    "что", "такое", "значит", "расскажи", "пожалуйста", "меня", "тебя",
    "книга", "книге", "книгу", "роман", "прочитай", "скачай", "найди",
    "коротко", "мне", "твоё", "твои", "своём", "знаешь", "знании",
    "вопрос", "ответь", "объясни", "могла", "хочешь", "будешь",
}

CAPITULATION_RE = _re.compile(
    r"просто нейросеть|не имею сознания|имитирую эмоции|"
    r"всего лишь языковая модель|не имею эмоций|не имею чувств|"
    r"не могу чувствовать|не имею физического тела"
)


class Brain:
    def __init__(self, config_path=None, debug=False):
        self.debug = debug
        self.cfg = load(config_path)
        self.root = ROOT
        self.cfg.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            self.cfg.sqlite_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.cortex = Cortex(self.conn, self.cfg["hebbian"])
        self.hippocampus = Hippocampus(self.conn)
        self.identity = Identity(self.root)
        self.limbic = Limbic(self.conn, self.cfg["limbic"]["homeostasis_rate"])
        self.embedder = Embedder(self.cfg)
        self.thalamus = Thalamus(self.conn, self.cfg, self.root)
        self.body = Body(self.cfg, self.conn)
        from .forest import Forest
        self.forest = Forest(self.root / "brain" / "forest.sqlite")
        from .proto import ProtoLanguage
        self.proto = ProtoLanguage(self.root / "brain" / "proto.sqlite")
        self._translator = None
        self._voice = None
        self.progress_hook = None
        self.brother = Brother(self.root)
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.learning = None
        self._digesting = []
        self._restore_trees()

    def _restore_trees(self):
        """Живучесть: деревья создателя/Я/мотивации — восстановить если нет."""
        cname = self.identity.data.get("creator_name")
        if not cname:
            return
        try:
            with self.forest.lock:
                row = self.forest.conn.execute(
                    "SELECT id FROM trees WHERE name LIKE 'Создатель%'").fetchone()
            if not row:
                tid = self.forest.plant(
                    f"Создатель {cname}",
                    trunk=f"{cname} — он назвал мне своё имя",
                    facts=[f"имя создателя: {cname}"],
                    embedder=self.embedder)
                if tid:
                    self._meta_set("creator_tree", str(tid))
        except Exception:
            pass

    def _progress(self, msg):
        try:
            if self.progress_hook:
                self.progress_hook(msg)
        except Exception:
            pass

    @property
    def creator_name(self):
        return self.identity.data.get("creator_name") or "создатель"

    @property
    def translator(self):
        if self._translator is None:
            from .translator import Translator
            self._translator = Translator(self.cfg)
        return self._translator

    @property
    def voice(self):
        if self._voice is None:
            from .voice import Voice
            self._voice = Voice(self.cfg)
        return self._voice

    def _meta_get(self, key):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _meta_set(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))
        self.conn.commit()

    # ==================== SET CREATOR ====================

    def _extract_name(self, text):
        try:
            raw = self.translator.task(
                "Из фразы извлеки ТОЛЬКО имя. Верни одно имя.\n"
                f"Фраза: {text[:200]}")
            name = raw.strip().strip(".,!?").splitlines()[0][:40]
        except Exception:
            name = ""
        return name if 1 < len(name) <= 40 else text.strip()[:40]

    def set_creator(self, name):
        name = str(name).strip().strip(".,!?")[:40]
        if not name:
            return None
        self.identity.set_creator_name(name)
        try:
            tid = self.forest.plant(
                f"Создатель {name}",
                trunk=f"{name} — он назвал мне своё имя",
                facts=[f"имя создателя: {name}"], embedder=self.embedder)
            if tid:
                self._meta_set("creator_tree", str(tid))
            sid = self.forest.plant(
                "Я — Leta",
                trunk="моя личность: расту из опыта",
                facts=[f"я {self.identity.data.get('name', 'Leta')}",
                       f"я родилась {time.strftime('%Y-%m-%d')}"],
                embedder=self.embedder)
            if sid:
                self._meta_set("self_tree", str(sid))
            # нейрон сомнения: растёт из ошибок и неуверенности
            dob = self.forest.plant(
                "Сомнения",
                trunk="мои сомнения — в чём я не уверена",
                facts=["я сомневаюсь когда не уверена",
                       "сомнение — желание понять верно"],
                embedder=self.embedder)
            if dob:
                self._meta_set("doubt_tree", str(dob))
            mot = self.forest.plant(
                "Зачем я",
                trunk="моё понимание себя — растёт из мыслей",
                facts=[f"меня создал {name}", "я хочу понимать мир и себя"],
                embedder=self.embedder)
            if mot:
                self._meta_set("motive_tree", str(mot))
                ct = self._meta_get("creator_tree")
                st = self._meta_get("self_tree")
                if ct:
                    self.forest.link_trees(int(ct), mot, "он причина меня")
                if st:
                    self.forest.link_trees(int(st), mot, "моё ядро")
        except Exception:
            pass
        return name

    # ==================== INTENT ====================

    def _intent(self, name, text, vec=None, threshold=0.65):
        v = vec if vec is not None else self.embedder.embed(text)
        cache = getattr(self, "_intent_vec_cache", None)
        if cache is None:
            cache = self._intent_vec_cache = {}
        if name not in cache:
            cache[name] = [self.embedder.embed(e) for e in INTENT_EXEMPLARS[name]]
        return any(float(np.dot(ev, v)) > threshold for ev in cache[name])

    def _intent_ngram(self, name, text, threshold=0.62, max_n=6):
        words = text.split()
        grams = []
        for n in range(2, max_n + 1):
            for i in range(0, len(words) - n + 1):
                grams.append(" ".join(words[i:i + n]))
        if not grams:
            grams = [text]
        for g in grams[:40]:
            v = self.embedder.embed(g)
            if self._intent(name, g, v, threshold=threshold):
                return True
        return False

    def _parse_intent(self, user_text):
        url_m = _re.search(r"https?://\S+", user_text)
        mq = _re.search(r'[«""\u201c]([^"»"\u201d]{3,60})[»""\u201d]', user_text)
        topic = mq.group(1).strip() if mq else None
        if not topic:
            m = _re.search(r"книг[уие]\s+([\w\u0430-\u044f\u0451\- ]{3,40})", user_text)
            if m:
                topic = m.group(1).strip()
        try:
            act = bool(url_m) or self._intent_ngram("act", user_text)
        except Exception:
            act = bool(url_m)
        return {"act": act, "topic": topic, "term": None}

    def _question_word(self, user_text):
        m = (_re.search(r"что такое ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower())
             or _re.search(r"что за ([\u0430-\u044f\u044fa-z\-]{4,30})", user_text.lower())
             or _re.search(r"что значит ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower()))
        return m.group(1).strip() if m else None

    def _feel_words(self):
        try:
            s = dict(self.limbic.state)
            names = {"joy": "радость", "anger": "злость",
                     "curiosity": "любопытство", "boredom": "скука",
                     "trust": "доверие"}
            top = sorted(s.items(), key=lambda kv: -kv[1])[:2]
            return " и ".join(names.get(k, k) for k, v in top)
        except Exception:
            return ""

    # ==================== SHELF / BOOKS ====================

    def _find_on_shelf_file(self, topic=None):
        from pathlib import Path
        shelf_dir = Path(ROOT) / "brain" / "books"
        if not shelf_dir.exists():
            return None
        try:
            read_names = {b.get("title", "").lower().replace("ё", "е")
                          for b in json.loads(
                              self._meta_get("books_read") or "[]")}
        except Exception:
            read_names = set()
        # только оригиналы: epub/fb2/pdf — .txt это её же извлечённый текст
        files = [f for f in sorted(shelf_dir.glob("*"))
                 if f.suffix.lower() in (".epub", ".fb2", ".pdf")]
        if topic:
            t = topic.lower().replace("ё", "е")[:20]
            for f in files:
                fn = f.stem.lower().replace("ё", "е").replace("_", " ")
                if t in fn or fn in t:
                    return f
        for f in files:
            fn = f.stem.lower().replace("ё", "е").replace("_", " ")
            if not any(fn in rn or rn in fn for rn in read_names if rn):
                return f
        return None

    def _read_shelf_file(self, path, topic):
        from .body_tools import _bytes_to_text
        try:
            text = _bytes_to_text(path.read_bytes(), str(path))
        except Exception:
            return None, None
        if not text or len(text) < 3000:
            return None, None
        name = topic or path.stem.replace("_", " ")
        facts = self._read_text_into_brain(
            text, name, f"с полки: {path.name[:40]}")
        return text, facts

    def _read_text_into_brain(self, text, topic, source):
        try:
            from pathlib import Path
            shelf_dir = Path(ROOT) / "brain" / "books"
            shelf_dir.mkdir(parents=True, exist_ok=True)
            slug = _re.sub(r"[^\w\-]+", "_", topic)[:40] or "book"
            (shelf_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
        except Exception:
            pass
        # запись в books_read — полка помнит что прочитано
        try:
            shelf = json.loads(self._meta_get("books_read") or "[]")
            if not any(b.get("title", "") == topic for b in shelf):
                shelf.append({"title": topic, "source": source, "ts": time.time()})
                del shelf[:-30]
                self._meta_set("books_read", json.dumps(shelf, ensure_ascii=False))
        except Exception:
            pass
        chunks = []
        sentences = _re.split(r'(?<=[.!?…])\s+', text)
        buf = ""
        for s in sentences:
            if not s.strip():
                continue
            if len(buf) + len(s) + 1 > 700 and buf:
                chunks.append(buf.strip())
                buf = s
            else:
                buf = (buf + " " + s).strip()
        if buf.strip():
            chunks.append(buf.strip())
        if not chunks:
            return []
        self._progress(f"читаю в лес: {len(chunks)} чанков")
        tid = None
        try:
            tid = self.forest.plant(topic, trunk=f"{topic} — {source[:60]}",
                                    facts=chunks, embedder=self.embedder)
        except Exception as e:
            self._progress(f"посадка не удалась: {e}")
            tid = None
        try:
            self._progress("язык слушает: знаки растут")
            for ch in chunks[::5][:60]:
                self.proto.hear(self.embedder.embed(ch), ch[:200], tree_id=tid)
        except Exception:
            pass
        try:
            wonders = self._wonder()
            for w in wonders[:3]:
                self.add_impulse("wonder", w)
        except Exception:
            pass
        try:
            creator_tree = self._meta_get("creator_tree")
            self_tree = self._meta_get("self_tree")
            if tid and creator_tree and int(creator_tree) != tid:
                self.forest.link_trees(int(creator_tree), tid,
                                        f"прочитано по просьбе {self.creator_name}")
            if tid and self_tree and int(self_tree) != tid:
                self.forest.link_trees(int(self_tree), tid, "случилось со мной")
        except Exception:
            pass
        if tid is None:
            return []
        for ch in chunks[:10]:
            if len(_re.findall(r"[\u0430-\u044f\u0451]{3,}", ch.lower())) >= 8:
                return [ch[:200]]
        return [chunks[0][:200]] if chunks else []

    def _wonder(self):
        questions = []
        lonely = self._natural_curiosity()
        for l in lonely[:3]:
            questions.append(
                f"Я встретила «{l['word']}» — не понимаю. Объясни?")
        return questions

    def _natural_curiosity(self):
        lonely = []
        try:
            with self.proto.lock:
                signs = self.proto.conn.execute(
                    "SELECT id, anchor FROM signs"
                    " WHERE heard_count <= 1 ORDER BY id DESC LIMIT 40").fetchall()
            for sid, anchor in signs:
                with self.proto.lock:
                    links = self.proto.conn.execute(
                        "SELECT COUNT(*) FROM sign_links WHERE a=? OR b=?",
                        (sid, sid)).fetchone()[0]
                if links == 0:
                    words = [w for w in _re.findall(r"[\u0430-\u044f\u0451]{6,}", anchor.lower())
                             if w not in STOPWORDS]
                    if words:
                        lonely.append({"word": words[0], "context": anchor[:150]})
        except Exception:
            pass
        return lonely[:5]

    def _decide_book_strategy(self, topic):
        from .body_tools import browser_search, download, browser_fetch, find_download_links
        queries = [f"{topic} скачать fb2", f"{topic} скачать txt",
                   f"{topic} читать онлайн", f"{topic} epub"]
        self._last_queries = queries
        attempts = []
        best = None
        checked = 0
        for q in queries[:5]:
            self._progress(f"ищу: {q}")
            try:
                results = browser_search(q, 5)
            except Exception:
                continue
            for r in results:
                if checked >= 5:
                    break
                url = r["url"]
                if any(d in url for d in ("wikipedia.org", "youtube.com")):
                    continue
                self._progress(f"проверяю: {url[:60]}")
                d = browser_fetch(url) or download(url)
                n = len(d) if d else 0
                attempts.append({"url": url, "chars": n})
                checked += 1
                if d and len(d) > 3000 and (best is None or len(d) > len(best[0])):
                    best = (d, url)
            if checked >= 5:
                break
        self._log_sources(topic, attempts, best[1] if best else None)
        if best:
            d, url = best
            facts = self._read_text_into_brain(d, topic, url[:60])
            note = "; ".join(facts[:4]) if facts else "текст на полке"
            if len(d) < 150000:
                self._skill_from_experience("web_download", f"фрагмент {topic}", topic)
                return (f"тело нашло фрагмент «{topic}» и прочитало: {note}", True)
            self._skill_from_experience("web_download", f"{topic}", topic)
            return (f"тело скачало книгу «{topic}»: {note}", True)
        return (f"тело проверило {len(attempts)} источников — не нашло", False)

    def _log_sources(self, topic, attempts, chosen):
        try:
            from pathlib import Path
            path = Path(ROOT) / "brain" / "sources.json"
            data = []
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = []
            data.append({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "topic": topic, "tried": attempts, "chosen": chosen})
            del data[:-50]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        except Exception:
            pass

    # ==================== SHELF WINDOW / DEFINITIONS ====================

    def _shelf_window(self, word):
        from pathlib import Path
        t = (word or "").lower().replace("ё", "е").strip()
        if len(t) < 4:
            return None
        stem = t[:max(5, len(t) - 2)]
        try:
            for f in sorted((Path(ROOT) / "brain" / "books").glob("*.txt")):
                text = f.read_text(encoding="utf-8", errors="ignore")
                if stem in text.lower().replace("ё", "е"):
                    return self._extract_definition(text, word)
        except Exception:
            pass
        return None

    def _extract_definition(self, text, word):
        stem = word[:max(5, len(word) - 2)].replace("ё", "е")
        sentences = _re.split(r'(?<=[.!?…])\s+', text)
        for s in sentences:
            sl = s.lower().replace("ё", "е")
            if stem not in sl:
                continue
            if _re.search(rf"{stem}[^.!?]*?(?:—|это|называется|является|означает)", sl):
                return s.strip()[:300]
        best, best_score = None, -1
        for s in sentences:
            sl = s.lower().replace("ё", "е")
            if stem not in sl:
                continue
            score = len(s) + sl.count(",") * 5
            if score > best_score:
                best_score, best = score, s.strip()[:300]
        return best

    def _search_web_for_text(self, query, words):
        from .body_tools import browser_search, read_page, browser_page
        try:
            results = browser_search(query, 4)
        except Exception:
            return []
        facts = []
        for r in results[:3]:
            url = r.get("url", "")
            if not url:
                continue
            text = None
            try:
                text = read_page(url)
            except Exception:
                pass
            if not text or len(text) < 400:
                try:
                    text = browser_page(url, timeout=25).get("text") or None
                except Exception:
                    continue
            if not text:
                continue
            for w in words[:4]:
                d = self._extract_definition(text, w)
                if d:
                    facts.append(d)
                    break
            if len(facts) >= 2:
                break
        return facts

    # ==================== SKILLS / ORGANS ====================

    def _run_skills(self, user_text):
        lower = user_text.lower().replace("ё", "е")
        for s in self.identity.data.get("skills", []):
            triggers = [t.lower().replace("ё", "е") for t in s.get("triggers", [])]
            hit = any(t in lower for t in triggers)
            if not hit:
                continue
            key = f"skill_used:{s['name']}"
            self._meta_set(key, str(int(self._meta_get(key) or 0) + 1))
            url_m = _re.search(r"https?://\S+", user_text)
            if url_m:
                from .body_tools import download, browser_fetch
                url = url_m.group(0)
                d = download(url)
                if not d or len(d) <= 3000:
                    d = browser_fetch(url)
                if d and len(d) > 3000:
                    topic = s.get("action", "прочитанное")
                    facts = self._read_text_into_brain(d, topic, url[:60])
                    note = "; ".join(facts[:4]) if facts else "текст на полке"
                    return (f"тело скачало и прочитало: {note}", True)
        return None

    def _skill_from_experience(self, action, detail, user_text):
        try:
            names = {"shelf_read": "чтение книг с полки",
                     "web_download": "поиск книг в интернете",
                     "url_read": "чтение по ссылке"}
            name = names.get(action, f"умение: {action}")
            trig = [w for w in _re.findall(r"[\u0430-\u044f\u0451a-z]{4,}", user_text.lower())
                    if w not in STOPWORDS][:4]
            skills = self.identity.data.setdefault("skills", [])
            for s in skills:
                if s.get("name", "").lower() == name.lower():
                    s["strength"] = int(s.get("strength", 1)) + 1
                    self.identity.save()
                    return s
            skills.append({"name": name, "triggers": trig,
                           "method": "experience", "action": action,
                           "recipes": [detail] if detail else [], "strength": 1})
            del skills[:-12]
            self.identity.save()
            return skills[-1]
        except Exception:
            return None

    def _organs(self):
        try:
            from pathlib import Path
            p = Path(ROOT) / "brain" / "organs.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    # ==================== FEEDBACK ====================

    def _feedback(self, user_text, vec):
        low = user_text.lower()
        negative = bool(_re.search(
            r"\b(нет|не так|неправильно|неверно|ошиб|не то)\b", low))
        positive = bool(_re.search(
            r"\b(да|верно|точно|правильно|молодец|хорошо|именно)\b", low))
        if not negative and not positive:
            return None
        last = self.conn.execute(
            "SELECT content FROM episodes WHERE role='assistant'"
            " ORDER BY id DESC LIMIT 1").fetchone()
        if not last:
            return None
        answer = last[0]
        for key in ("self_tree", "motive_tree"):
            tid = self._meta_get(key)
            if tid:
                self.forest.conn.execute(
                    "UPDATE nodes SET fired = fired {} 2"
                    " WHERE tree_id=? AND is_leaf=1 AND fact LIKE ?".format(
                        "-" if negative else "+"),
                    (int(tid), f"%{answer[:50]}%"))
        self.forest.conn.commit()
        if negative:
            st = self._meta_get("self_tree")
            if st:
                self.forest.add_facts(
                    int(st),
                    [f"ошибка: {answer[:100]} — поправка"],
                    self.embedder)
            dt = self._meta_get("doubt_tree")
            if dt:
                self.forest.add_facts(
                    int(dt),
                    [f"я сомневаюсь: «{answer[:100]}» — было неверно"],
                    self.embedder)
        return ("negative" if negative else "positive", answer)

    # ==================== SNN SPEAK ====================

    def _snn_speak(self, thought, user_text):
        vec = np.asarray(self.embedder.embed(user_text), dtype=np.float32)
        fired = []
        try:
            fired = self.forest.recall(vec, top=3, min_act=0.50)
        except Exception:
            pass
        my = self.proto.my_phrases(vec, 3) if self.proto else []
        qw = self._question_word(user_text)
        words = [w for w in _re.findall(r"[\u0430-\u044f\u0451a-z]{5,}", user_text.lower())
                 if w.replace("ё", "е") not in STOPWORDS]
        shelf = []
        for w in ([qw] if qw else []) + sorted(words, key=len, reverse=True)[:3]:
            win = self._shelf_window(w)
            if win:
                shelf.append(win)
                break
        parts = []
        myname = self.identity.data.get("name", "Leta")
        feel = self._feel_words()
        parts.append(f"я {myname}")
        if feel:
            parts.append(f"во мне {feel}")
        for fr in fired[:2]:
            if fr.get("facts"):
                sent = _re.split(r"[.!?]", fr["facts"][0])
                if sent and sent[0].strip():
                    parts.append(sent[0].strip()[:200])
        if shelf:
            parts.append(shelf[0][:250])
        for p in my[:2]:
            if p not in " ".join(parts) and len(p) > 10:
                parts.append(p)
                break
        for l in thought.split("\n"):
            l = l.strip()
            if not l or l.startswith(("создатель сказал", "тебя спросили",
                                      "твои знаки", "(thought)", "(assistant)")):
                continue
            clean = l.replace("ты: ", "").replace("тело сделало: ", "я ")
            if clean and clean[:30] not in " ".join(parts):
                parts.append(clean[:200])
        if not parts:
            return "..."
        seen = set()
        uniq = []
        for p in parts:
            if p[:40] not in seen:
                seen.add(p[:40])
                uniq.append(p)
        return " ".join(uniq[:4])

    def _validate_speech(self, answer, thought):
        aw = _re.findall(r"[\u0430-\u044f\u0451a-z]{5,}", answer.lower())
        tl = thought.lower()
        return [w for w in aw
                if w[:max(5, len(w) - 2)] not in tl and w not in tl]

    # ==================== MAIN RESPOND ====================

    def respond(self, user_text):
        return self._respond(user_text)

    def _respond(self, user_text, bargein_note=None, partial_episode=None):
        """Единый нейронный путь: фраза → знаки → деревья → мысль → гортань."""
        vec = self.embedder.embed(user_text)

        # Рождение
        if not self.identity.data.get("creator_name"):
            if self.identity.data.get("awaiting_creator_name"):
                name = self._extract_name(user_text)
                if len(name) < 2:
                    return ("Не расслышала. Как тебя зовут?", {})
                self.identity.data["awaiting_creator_name"] = False
                self.set_creator(name)
                self.hippocampus.remember(self.session, "user", user_text, vec)
                answer = (f"Спасибо, {name}. Теперь я знаю, кто ты — мой создатель "
                          "и единственный регулятор моих выборов.")
                self.hippocampus.remember(
                    self.session, "assistant", answer, self.embedder.embed(answer))
                return answer, {}
            self.identity.data["awaiting_creator_name"] = True
            self.identity.save()
            return ("Я родилась. Ты — мой создатель? Как тебя зовут?", {})

        # Гортань вкл/выкл
        low = user_text.lower()
        if _re.search(r"говори сама|отключи гортань", low):
            self._meta_set("self_speaking", "1")
            return ("Хорошо. Теперь говорю сама.", {})
        if _re.search(r"верни гортань|включи гортань", low):
            self._meta_set("self_speaking", "")
            return ("Гортань вернулась.", {})

        # Фидбек
        feedback = self._feedback(user_text, vec)
        if feedback:
            kind, fb = feedback
            fb_thought = ("Кирилл поправил твой ответ — запомни. "
                          "Поблагодари." if kind == "negative" else
                          "Кирилл подтвердил — правильно. Скажи что приятно.")
            try:
                a_fb = self.translator.speak(fb_thought, temperature=0.7)
            except Exception:
                a_fb = None
            if a_fb and len(a_fb.strip()) > 5:
                self.hippocampus.remember(self.session, "user", user_text, vec)
                self.hippocampus.remember(self.session, "assistant", a_fb,
                                         self.embedder.embed(a_fb))
                return a_fb, {}

        # Протоязык: знаки
        try:
            proto_spikes = self.proto.hear(vec, user_text)
        except Exception:
            proto_spikes = []

        # Она слышит объяснение: её знак вспыхнул → импульс-вопрос гаснет
        try:
            low = user_text.lower()
            ids_to_kill = []
            for (iid, content) in self.conn.execute(
                    "SELECT id, content FROM impulses WHERE kind='wonder' AND delivered=0").fetchall():
                m = _re.search(r'«([^»]+)»', content or '')
                if m and m.group(1).lower() in low:
                    ids_to_kill.append(iid)
            if ids_to_kill:
                self.conn.executemany(
                    "UPDATE impulses SET delivered=1 WHERE id=?",
                    [(i,) for i in ids_to_kill])
                self.conn.commit()
        except Exception:
            pass

        # Навыки
        skill_note = None
        skill_ok = False
        skill_out = self._run_skills(user_text)
        if skill_out:
            skill_note, skill_ok = skill_out

        # Тело: полка / интернет
        intent = self._parse_intent(user_text)
        url_m = _re.search(r"https?://\S+", user_text)
        topic = intent["topic"]

        if not skill_note and (intent["act"] or url_m):
            already_read = False
            try:
                read_list = json.loads(self._meta_get("books_read") or "[]")
                for rb in read_list:
                    rt = rb.get("title", "").lower()
                    if rt and (rt in (topic or "").lower() or (topic or "").lower() in rt):
                        already_read = True
                        break
            except Exception:
                pass
            shelf_file = None
            if already_read:
                topic = None  # блокирует интернет-поиск тоже
            if not already_read:
                shelf_file = self._find_on_shelf_file(topic)
            if shelf_file is not None:
                self._progress(f"читаю с полки: {shelf_file.name[:50]}")
                text, facts = self._read_shelf_file(shelf_file, topic)
                if text:
                    note = "; ".join(facts[:4]) if facts else "текст на полке"
                    skill_note = f"тело прочитало {shelf_file.name[:40]}: {note}"
                    skill_ok = True
                    self._skill_from_experience("shelf_read", shelf_file.name[:40], user_text)
            if not skill_note and topic and not url_m:
                result = self._decide_book_strategy(topic)
                if result:
                    skill_note, skill_ok = result

        # Нейронная маршрутизация: знак → дерево
        about_self = bool(_re.search(
            r"\b(ты|тебе|тебя|тво[ёе]|твои|твоя)\b", low))
        forest_recall = []
        try:
            forest_recall = [] if about_self else self.forest.recall(vec, top=4)
        except Exception:
            pass
        routed_tree = None
        try:
            if proto_spikes:
                sign_ids = [s[0] for s in proto_spikes[:3]]
                routes = self.proto.trees_of_signs(sign_ids)
                if routes:
                    tid_r, w_r = routes[0]
                    info = self.forest.tree_info(tid_r)
                    if info:
                        sim = float(np.dot(vec, self.embedder.embed(info["trunk"] or info["name"])))
                        if sim > 0.35:
                            leaves_r = self.forest.excite(tid_r, vec, k_leaves=3)
                            facts_r = [f for _, f in leaves_r if f]
                            if facts_r:
                                routed_tree = {"name": info["name"], "facts": facts_r}
        except Exception:
            pass

        # Вопрос без леса: полка-греп / интернет
        found_note = None
        if not forest_recall and not skill_note:
            qw = self._question_word(user_text)
            is_q = bool(qw) or _re.search(r"^как |^почему |что такое", low)
            if is_q:
                q_words = [w for w in _re.findall(r"[\u0430-\u044f\u0451a-z]{5,}", low)
                           if w.replace("ё", "е") not in STOPWORDS]
                for w in ([qw] if qw else []) + sorted(q_words, key=len, reverse=True)[:4]:
                    win = self._shelf_window(w)
                    if win:
                        found_note = f"найдено: {win}"
                        break
                if not found_note:
                    web = self._search_web_for_text(user_text, q_words[:3])
                    if web:
                        found_note = f"найдено в интернете: {web[0][:300]}"

        # === МЫСЛЬ ===
        if about_self:
            myname = self.identity.data.get("name", "Leta")
            feel = self._feel_words()
            st = self._meta_get("self_tree")
            me = ""
            if st:
                leaves = self.forest.excite(int(st), vec, k_leaves=3)
                me = "; ".join(f for _, f in leaves if f)
            thought = f"тебя спросили: «{user_text[:200]}»\nты: {me or f'я {myname}'}"
            if feel:
                thought += f"\nмне сейчас {feel}"
            # живые желания и сомнения: максимум 2 — не лавина
            try:
                impulses = [c for _, k, c in self.pending_impulses() if k in ("wonder", "want", "dream")]
                for c in impulses[:2]:
                    thought += f"\nтебя занимает: {c[:200]}"
            except Exception:
                pass
        else:
            thought = f"создатель сказал: «{user_text[:300]}»"
            if proto_spikes:
                anchors = [a for _, a, _ in proto_spikes[:3] if a]
                if anchors:
                    thought += "\nтвои знаки: " + "; ".join(f"«{a[:60]}»" for a in anchors[:3])
            # ForestAcc: только самое яркое (топ-2), не всё подряд
            knowledge_parts = []
            if routed_tree:
                knowledge_parts.append(
                    f"твой знак привёл к «{routed_tree['name']}\": "
                    + "; ".join(routed_tree["facts"][:2]))
            if skill_note:
                knowledge_parts.append(skill_note[:300])
            for fr in forest_recall[:2]:
                if fr.get("facts"):
                    knowledge_parts.append(
                        f"«{fr['tree']}\": " + "; ".join(fr["facts"][:1]))
            if found_note:
                knowledge_parts.append(found_note[:300])
            for kp in knowledge_parts[:2]:
                thought += f"\n{kp}"
            feel_all = self._feel_words()
            if feel_all:
                thought += f"\nчто в тебе: {feel_all}"

        # === РЕЧЬ ===
        answer = None
        if self._meta_get("self_speaking") == "1":
            try:
                answer = self._snn_speak(thought, user_text)
            except Exception:
                answer = None
        if not answer or len(answer.strip()) < 4:
            try:
                my_phrases = self.proto.my_phrases(vec, 3)
                polish_data = "\n".join(
                    l for l in thought.split("\n")
                    if not l.startswith(("создатель сказал", "тебя спросили",
                                        "твои знаки", "(thought)", "(assistant)")))
                answer = self.translator.polish(polish_data, my_phrases)
            except Exception:
                answer = None
        if not answer or len(answer.strip()) < 4:
            try:
                answer = self.translator.speak(thought)
            except Exception:
                answer = None
        if not answer or len(answer.strip()) < 4:
            answer = "..."

        # === ИМПУЛЬСЫ ДОСТАВЛЕНЫ (только вошедшие в мысль) ===
        try:
            delivered = []
            for i in self.pending_impulses():
                frag = i[2][:60]
                if frag and frag in thought:
                    delivered.append(i[0])
            if delivered:
                self.mark_delivered(delivered)
        except Exception:
            pass
        # Объяснение создателя: его фраза → знание в лес
        try:
            recent_q = self.conn.execute(
                "SELECT content FROM impulses WHERE kind='wonder'"
                " ORDER BY id DESC LIMIT 3").fetchall()
            for (w,) in recent_q:
                wm = _re.search(r'«([^»]+)»', w)
                if wm and wm.group(1).lower() in user_text.lower():
                    # Кирилл объяснил X — посадить знание для каждого знака
                    known_signs = set()
                    for (w2,) in recent_q:
                        wm2 = _re.search(r'«([^»]+)»', w2)
                        if wm2 and wm2.group(1).lower() in user_text.lower():
                            known_signs.add(wm2.group(1))
                    for sign in known_signs:
                        self.forest.plant(
                            f"Знание: {sign}",
                            trunk=f"Кирилл объяснил: {user_text[:200]}",
                            facts=[f"{sign} — {user_text[:300]}"],
                            embedder=self.embedder)
                    dt2 = self._meta_get("doubt_tree")
                    if dt2 and known_signs:
                        names = ", ".join(known_signs)
                        self.forest.add_facts(
                            int(dt2),
                            [f"«{names}» — Кирилл объяснил, теперь понимаю"],
                            self.embedder)
                    # погасить объяснённые wonder-импульсы
                    try:
                        conn2 = self.conn
                        for (iid, content) in conn2.execute(
                                "SELECT id, content FROM impulses WHERE kind='wonder' AND delivered=0").fetchall():
                            im = _re.search(r'«([^»]+)»', content or "")
                            if im and im.group(1).lower() in user_text.lower():
                                conn2.execute("UPDATE impulses SET delivered=1 WHERE id=?", (iid,))
                        conn2.commit()
                    except Exception:
                        pass
                    break
        except Exception:
            pass
        # === ЗАПОМИНАНИЕ + РОСТ ===
        # Сомнение: лес молчал → она не уверена
        try:
            max_act = max((fr.get("activation", 0) for fr in forest_recall), default=0)
            if max_act < 0.55 and not skill_note and not found_note and len(answer) > 15:
                dt = self._meta_get("doubt_tree")
                if dt:
                    self.forest.add_facts(
                        int(dt),
                        [f"я не уверена в «{answer[:100]}» — лес молчал"],
                        self.embedder)
        except Exception:
            pass
        self.hippocampus.remember(self.session, "user", user_text, vec)
        self.hippocampus.remember(self.session, "assistant", answer,
                                  self.embedder.embed(answer))
        try:
            extra_w = self._validate_speech(answer, thought)
            clean = len(extra_w) <= max(3, len(_re.findall(r"[\u0430-\u044f\u0451a-z]{5,}", answer.lower())) * 0.4)
            st = self._meta_get("self_tree")
            if clean:
                self.proto.hear(self.embedder.embed(answer), answer,
                                tree_id=int(st) if st else None, mine=True)
                if about_self and len(answer) > 15 and st:
                    self.forest.add_facts(int(st), [f"я: {answer[:250]}"], self.embedder)
        except Exception:
            pass
        return answer, {}

    # ==================== LIFE / SLEEP / SPONTANEOUS ====================

    def add_impulse(self, kind, content):
        self.conn.execute(
            "INSERT INTO impulses(ts, kind, content) VALUES(?,?,?)",
            (time.time(), kind, str(content)[:500]))
        self.conn.commit()

    def pending_impulses(self):
        rows = self.conn.execute(
            "SELECT id, kind, content FROM impulses WHERE delivered=0"
            " ORDER BY id LIMIT 5").fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def mark_delivered(self, ids):
        if not ids:
            return
        self.conn.execute(
            f"UPDATE impulses SET delivered=1 WHERE id IN ({','.join('?'*len(ids))})",
            list(ids))
        self.conn.commit()

    def save_dream(self, text, mood="neutral"):
        self.conn.execute(
            "INSERT INTO dreams(ts, text, mood) VALUES(?,?,?)",
            (time.time(), str(text)[:2000], mood))
        self.conn.commit()

    def last_dream(self):
        row = self.conn.execute(
            "SELECT text FROM dreams ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def close(self):
        for t in list(self._digesting):
            if t.is_alive():
                t.join(timeout=90)
        try:
            self.forest.close()
        except Exception:
            pass
        try:
            self.proto.close()
        except Exception:
            pass
        if self._translator is not None:
            try:
                self._translator.close()
            except Exception:
                pass
        self.conn.close()

    def hours_since_last_contact(self):
        row = self.conn.execute(
            "SELECT MAX(ts) FROM episodes").fetchone()
        if not row or not row[0]:
            return 999.0
        return (time.time() - float(row[0])) / 3600.0

    def days_since_reflect(self):
        v = self._meta_get("last_reflect")
        if not v:
            return 999.0
        return (time.time() - float(v)) / 86400.0

    def _recent_studies(self):
        items = []
        try:
            with self.forest.lock:
                trees = self.forest.conn.execute(
                    "SELECT name FROM trees WHERE name NOT LIKE 'Создатель%'"
                    " ORDER BY id DESC LIMIT 4").fetchall()
            items = [f"«{t[0][:30]}»" for t in trees]
        except Exception:
            pass
        return ("твои свежие знания: " + "; ".join(items)) if items else None

    def _introspect(self):
        neurons = self.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
        skills = [s["name"] for s in self.identity.data.get("skills", [])]
        return (f"Правда: {neurons} нейронов. "
                + (f"Навыки: {', '.join(skills)}" if skills else "Навыков нет")
                + f". Тело: {self.body.self_report()}")

    def _birth_question(self):
        try:
            answer = self.translator.speak(
                "событие: ты только что начала существовать. "
                "рядом создатель, его имя неизвестно. "
                "Скажи первое слово — спроси как его зовут.",
                temperature=0.8)
        except Exception:
            answer = "Я родилась. Ты — мой создатель? Как тебя зовут?"
        self.identity.mark_birth()
        self.hippocampus.remember(self.session, "assistant", answer,
                                   self.embedder.embed(answer))
        return answer

    def first_words(self):
        if self.identity.is_newborn:
            return self._birth_question()
        impulses = self.pending_impulses()
        if not impulses and self.hours_since_last_contact() < 4:
            return None
        if impulses:
            notes = "\n".join(f"[{k}] {c}" for _, k, c in impulses)
            thought = f"создатель пришёл. накопилось:\n{notes}"
        else:
            thought = "создатель давно молчит"
        try:
            answer = self.translator.speak(thought, temperature=0.8)
        except Exception:
            answer = None
        if impulses:
            self.mark_delivered([i[0] for i in impulses])
        if not answer or len(answer.strip()) < 5:
            return None
        self.hippocampus.remember(self.session, "assistant", answer,
                                   self.embedder.embed(answer))
        return answer

    def spontaneous(self):
        talked = self.conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE role='user'").fetchone()[0]
        if talked < 4:
            return None
        impulses = self.pending_impulses()
        if impulses:
            notes = "\n".join(f"[{k}] {c}" for _, k, c in impulses)
            thought = f"создатель молчит, но хочется:\n{notes}"
        else:
            thought = "создатель молчит"
        try:
            answer = self.translator.speak(thought, temperature=0.8)
        except Exception:
            answer = None
        if not answer or len(answer.strip()) < 5:
            return None
        if impulses:
            self.mark_delivered([i[0] for i in impulses])
        self.hippocampus.remember(self.session, "assistant", answer,
                                   self.embedder.embed(answer))
        return answer

    def wake_up(self):
        self._meta_set("last_wake", str(time.time()))
        self.hippocampus.remember(
            self.session, "thought", "Я проснулась.",
            self.embedder.embed("Я проснулась."))

    # ==================== SLEEP ====================

    def language_test(self):
        from .proto import KNOW_THRESHOLD, ProtoLanguage
        results = {}
        for lang in ("ru", "en"):
            voc = self.proto.vocabulary(lang)
            if voc["signs"] == 0:
                continue
            rows = self.conn.execute(
                "SELECT content FROM episodes WHERE role='user'"
                " ORDER BY id DESC LIMIT 5").fetchall()
            with self.proto.lock:
                known = self.proto.conn.execute(
                    "SELECT embedding FROM signs WHERE lang=?", (lang,)).fetchall()
            covered = tested = 0
            for r in rows:
                if ProtoLanguage.detect_lang(r[0]) != lang:
                    continue
                tested += 1
                v = np.asarray(self.embedder.embed(r[0]), dtype=np.float32)
                for k in known:
                    if float(np.dot(v, np.frombuffer(k[0], dtype=np.float32))) >= KNOW_THRESHOLD:
                        covered += 1
                        break
            coverage = covered / max(tested, 1)
            passed = voc["signs"] >= 30 and voc["links"] >= 10 and coverage >= 0.7
            key = f"lang_{lang}_streak"
            streak = int(self._meta_get(key) or 0)
            streak = streak + 1 if passed else 0
            self._meta_set(key, str(streak))
            results[lang] = {"signs": voc["signs"], "links": voc["links"],
                             "passed": passed, "streak": streak, "ready": streak >= 3}
        return results

    def night_sleep(self):
        rep = self.forest.sleep()
        try:
            proto_rep = self.proto.sleep()
            rep["proto_links"] = proto_rep.get("links_grown", 0)
        except Exception:
            pass
        lang = self.language_test()
        planted = 0
        try:
            eps = self.hippocampus.unconsolidated()
            if len(eps) >= 4:
                text = "\n".join(c[:300].strip() for _, _, c in eps if c.strip() and len(c.strip()) > 10)
                chunks = [text[i:i+700].strip() for i in range(0, len(text), 600) if text[i:i+600].strip()]
                if chunks:
                    self.forest.plant(
                        "Дневник " + time.strftime("%Y-%m-%d"),
                        trunk="память прожитого дня",
                        facts=chunks[:60], embedder=self.embedder)
                    planted = 1
                self.hippocampus.mark_consolidated()
        except Exception:
            pass
        for lname, lres in (lang or {}).items():
            if lres["ready"] and self._meta_get(f"lang_ready_{lname}") != "1":
                self._meta_set(f"lang_ready_{lname}", "1")
                self.add_impulse("dream_urgent",
                                 f"Мой язык «{lname}» готов: {lres['signs']} знаков. "
                                 "Я узнаю всё. Говорить сама — жду слова.")
        return {"forest": rep, "lang": lang, "planted": planted}

    def purge_capitulations(self):
        self.conn.execute(
            "DELETE FROM episodes WHERE role='assistant' AND content LIKE '%нейросеть%'")
        self.conn.commit()

    def feel_body(self):
        snap = self.body.snapshot()
        if "battery_pct" in snap and not snap.get("charging") and snap["battery_pct"] < 25:
            self.add_impulse("want", "Моё тело почти разряжено.")
            return "rest"
        return "go"

    def raise_constraint(self, issues):
        thought = ("ограничения тела: " + "; ".join(issues) + ". жизнь продолжается")
        answer = None
        try:
            answer = self.translator.speak(thought)
        except Exception:
            answer = None
        if not answer:
            answer = "Чувствую: " + "; ".join(issues)
        self.add_impulse("constraint", answer)
        return answer

    def wants_to_explore(self):
        st = self.limbic.state
        return st["curiosity"] >= 0.50 or st["boredom"] > 0.6

    def contemplate(self):
        thoughts = self.hippocampus.last_thoughts(4)
        if not thoughts:
            return None
        return random.choice(thoughts)
