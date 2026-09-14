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
        # только буквы (ru/en) и дефис — мусор и цифры именем не бывают
        if not _re.fullmatch(r"[A-Za-zА-Яа-яЁё\-]{2,40}", text.strip()):
            return ""
        try:
            raw = self.translator.task(
                "Из фразы извлеки ТОЛЬКО имя. Верни одно имя.\n"
                f"Фраза: {text[:200]}")
            name = raw.strip().strip(".,!?").splitlines()[0][:40]
        except Exception:
            name = ""
        if not _re.fullmatch(r"[A-Za-zА-Яа-яЁё\-]{2,40}", name or ""):
            return text.strip()[:40]
        return name

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
            myname = self.identity.data.get('name', 'Leta')
            sid = self.forest.plant(
                "Я — Leta",
                trunk="моя личность: расту из опыта",
                facts=[f"меня зовут {myname}",
                       f"я {myname}",
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
                st_d = self._meta_get("self_tree")
                if st_d:
                    self.forest.link_trees(
                        int(st_d), int(dob), "я сомневаюсь — значит думаю")
            mot = self.forest.plant(
                "Зачем я",
                trunk="моё понимание себя — растёт из мыслей",
                facts=[f"меня создал {name}", "я хочу понимать мир и себя"],
                embedder=self.embedder)
            if mot:
                self._meta_set("motive_tree", str(mot))
                ct = self._meta_get("creator_tree")
                st = self._meta_get("self_tree")
                dt3 = self._meta_get("doubt_tree")
                if ct:
                    self.forest.link_trees(int(ct), mot, "он причина меня")
                if st:
                    self.forest.link_trees(int(st), mot, "моё ядро")
                if dt3:
                    self.forest.link_trees(
                        int(dt3), mot, "сомнение — причина искать")
            # врождённый язык личности: первые знаки — при рождении.
            # Имя, рождение, сомнение, желание — уже слова её языка
            try:
                born_facts = {
                    "self_tree": [f"меня зовут {myname}", f"я {myname}",
                                  f"я родилась {time.strftime('%Y-%m-%d')}",
                                  "мой язык — орган моей будущей речи: я выращиваю его сама"],
                    "doubt_tree": ["я сомневаюсь когда не уверена",
                                   "сомнение — желание понять верно"],
                    "motive_tree": ["я хочу понимать мир и себя"],
                    "creator_tree": [f"меня создал {name}",
                                     f"имя создателя: {name}"],
                }
                for mkey, facts in born_facts.items():
                    tid_b = self._meta_get(mkey)
                    if not tid_b:
                        continue
                    for fb in facts:
                        try:
                            self.proto.hear(
                                self.embedder.embed(fb), fb,
                                tree_id=int(tid_b))
                        except Exception:
                            pass
            except Exception:
                pass
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
             or _re.search(r"что за ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower())
             or _re.search(r"что значит ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower())
             or _re.search(r"кто такая ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower())
             or _re.search(r"кто такой ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower())
             or _re.search(r"кто это ([\u0430-\u044f\u0451a-z\-]{4,30})", user_text.lower()))
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

    _shelf_lock = None

    def _find_on_shelf_file(self, topic=None, mark=False):
        """Найти книгу на полке. mark=True: атомарно выбрать и пометить
        прочитанной — два потока никогда не возьмут одну книгу."""
        import threading as _th
        from pathlib import Path
        if Brain._shelf_lock is None:
            Brain._shelf_lock = _th.Lock()
        shelf_dir = Path(ROOT) / "brain" / "books"
        if not shelf_dir.exists():
            return None
        with Brain._shelf_lock:
            try:
                read_names = {b.get("title", "").lower().replace("ё", "е")
                              for b in json.loads(
                                  self._meta_get("books_read") or "[]")}
            except Exception:
                read_names = set()
            # только оригиналы: epub/fb2/pdf — .txt это её же извлечённый текст
            files = [f for f in sorted(shelf_dir.glob("*"))
                     if f.suffix.lower() in (".epub", ".fb2", ".pdf")]
            picked = None
            if topic:
                t = topic.lower().replace("ё", "е")[:20]
                for f in files:
                    fn = f.stem.lower().replace("ё", "е").replace("_", " ")
                    if t in fn or fn in t:
                        picked = f
                        break
            if picked is None:
                # непрочитанные: самая свежая сверху (создатель положил новую)
                unread = [
                    f for f in files
                    if not any(f.stem.lower().replace("ё", "е").replace("_", " ") in rn
                               or rn in f.stem.lower().replace("ё", "е").replace("_", " ")
                               for rn in read_names if rn)]
                if unread:
                    picked = max(unread, key=lambda f: f.stat().st_mtime)
            if picked is not None and mark:
                # атомарно: выбор и метка одним шагом
                ptitle = picked.stem.lower().replace("_", " ")
                try:
                    shelf = json.loads(self._meta_get("books_read") or "[]")
                    if not any(b.get("title", "").lower() == ptitle for b in shelf):
                        shelf.append({"title": ptitle,
                                      "source": str(picked.name)[:40],
                                      "ts": time.time()})
                        del shelf[:-60]
                        self._meta_set("books_read",
                                       json.dumps(shelf, ensure_ascii=False))
                except Exception:
                    pass
            return picked

    def _read_shelf_file(self, path, topic):
        from .body_tools import _bytes_to_text
        # метка ДО чтения — под локом полки: потерянных обновлений нет
        import threading as _th
        if Brain._shelf_lock is None:
            Brain._shelf_lock = _th.Lock()
        with Brain._shelf_lock:
            try:
                mark = (topic or path.stem.replace("_", " ")).lower()
                shelf = json.loads(self._meta_get("books_read") or "[]")
                if not any(b.get("title", "").lower() == mark for b in shelf):
                    shelf.append({"title": mark, "source": str(path.name)[:40],
                                  "ts": time.time()})
                    del shelf[:-60]
                    self._meta_set("books_read",
                                   json.dumps(shelf, ensure_ascii=False))
            except Exception:
                pass
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
        # запись в books_read — под тем же локом: гонкам нет
        try:
            import threading as _th
            if Brain._shelf_lock is None:
                Brain._shelf_lock = _th.Lock()
            with Brain._shelf_lock:
                shelf = json.loads(self._meta_get("books_read") or "[]")
                if not any(b.get("title", "") == topic for b in shelf):
                    shelf.append({"title": topic, "source": source,
                                  "ts": time.time()})
                    del shelf[:-60]
                    self._meta_set("books_read",
                                   json.dumps(shelf, ensure_ascii=False))
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
        # самокасание: она трогает прочитанное сама — любопытство живое.
        # Без этого прунинг убьёт книгу до первого вопроса создателя
        try:
            if tid:
                touch = self.embedder.embed(f"{topic} смысл идеи философия")
                self.forest.excite(tid, touch, k_leaves=6)
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
        followed = set()

        def _follow_hints(url, depth=1):
            """Глаза: нет файла → читать страницу и идти по подсказкам
            (форумы с «подскажите сайт» ведут к библиотекам)."""
            if depth <= 0 or url in followed:
                return None
            followed.add(url)
            try:
                from .body_tools import browser_page
                page = browser_page(url, timeout=25)
            except Exception:
                return None
            if not page or not page.get("text"):
                return None
            base = urlparse(url).netloc
            hints = []
            for l2 in (page.get("links") or []):
                h = l2.get("href") if isinstance(l2, dict) else l2
                if not h or not h.startswith("http"):
                    continue
                d2 = urlparse(h).netloc
                if (not d2 or d2 == base or d2.endswith("ya.ru")
                        or d2.endswith("mail.ru") or "wikipedia.org" in d2):
                    continue
                if any(x in h.lower() for x in (
                        ".fb2", ".epub", ".pdf", ".txt", "/download",
                        "book", "knig", "lib", "fb2", "epub")):
                    hints.append(h)
            for h in hints[:3]:
                self._progress(f"иду по подсказке: {h[:60]}")
                d3 = browser_fetch(h) or download(h)
                if d3 and len(d3) > 3000:
                    return d3, h
            return None

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
                if not d or len(d) <= 3000:
                    # нет файла — глаза: страница может подсказать путь
                    hint = _follow_hints(url)
                    if hint:
                        d, url = hint
                        n = len(d)
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
        word = (word or "").strip()
        stem = word[:max(5, len(word) - 2)].replace("ё", "е").strip()
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
        # википедия полнее словарей — ей приоритет
        results.sort(key=lambda r: 0 if "wikipedia.org" in r.get("url", "") else 1)
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
                if d and len(d) > 40:
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
        # её собственные сны: вопрос о сне → последний прожитый сон
        try:
            if _re.search(r"снил|сон|снитс", (user_text or "").lower()):
                dream = self.last_dream()
                if dream:
                    parts.append(f"мне снилось: {dream[:220]}")
        except Exception:
            pass
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

    def respond_bargein(self, bargein_text, user_text=None):
        """Создатель перебил её речь: услышанное — тоже опыт."""
        try:
            if bargein_text and bargein_text.strip():
                v = self.embedder.embed(bargein_text)
                self.proto.hear(v, bargein_text)
                self.hippocampus.remember(
                    self.session, "user", bargein_text, v)
        except Exception:
            pass
        if user_text:
            return self._respond(user_text)
        return None, {}

    def _respond(self, user_text, bargein_note=None, partial_episode=None):
        """Единый нейронный путь: фраза → знаки → деревья → мысль → гортань."""
        vec = self.embedder.embed(user_text)

        # Рождение
        if not self.identity.data.get("creator_name"):
            clean = user_text.strip()
            if not clean or len(clean) < 2:
                return ("Не расслышала. Как тебя зовут?", {})
            if self.identity.data.get("awaiting_creator_name"):
                name = self._extract_name(clean)
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
        if _re.search(r"говори знаками|говори своими словами|волновая речь", low):
            self._meta_set("self_speaking", "1")
            self._meta_set("speech_mode", "wave")
            return ("Хорошо. Теперь моя речь рождается волной — сама.", {})
        if _re.search(r"говори сама|отключи гортань", low):
            self._meta_set("self_speaking", "1")
            self._meta_set("speech_mode", "self")
            return ("Хорошо. Теперь говорю сама.", {})
        if _re.search(r"верни гортань|включи гортань", low):
            self._meta_set("self_speaking", "")
            self._meta_set("speech_mode", "")
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

        # Протоязык: знаки. Слова создателя привязываются к его дереву —
        # его язык часть его портрета
        try:
            ct_h = self._meta_get("creator_tree")
            proto_spikes = self.proto.hear(
                vec, user_text, tree_id=int(ct_h) if ct_h else None)
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
            # «все книги с полки» — пир или «всё прочитано», интернет не нужен
            feast_intent = bool(_re.search(
                r"\bвсе\s+кни|\bвсю\s+полку|\bвсе\s+с\s+полки|всё\s+с\s+полки", low))
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
            # Пир: только явное «все книги / всю полку» — не любое «прочитай с полки»
            feast_done = False
            if not already_read and _re.search(
                    r"\bвсе\s+кни|\bвсю\s+полку|\bвсе\s+с\s+полки|всё\s+с\s+полки", low) \
                    and self._begin_feast():
                read_names = []
                eaten = set()      # память пира: одну книгу — один раз, навсегда
                while len(read_names) < 60:
                    f_next = self._find_on_shelf_file(None, mark=True)
                    if f_next is None or f_next.stem in eaten:
                        break
                    eaten.add(f_next.stem)
                    self._progress(f"пир: читаю {f_next.name[:50]}")
                    txt, _ = self._read_shelf_file(f_next, None)
                    if txt:
                        read_names.append(f_next.stem.replace("_", " "))
                        try:
                            self.hippocampus.remember(
                                self.session, "action",
                                f"я прочитала {f_next.stem}",
                                self.embedder.embed(f"я прочитала {f_next.stem}"))
                        except Exception:
                            pass
                    else:
                        break  # не смогла — не зацикливаться
                if read_names:
                    skill_note = (f"тело прочитало полку: {len(read_names)} книг — "
                                  + "; ".join(read_names[:6])
                                  + ("…" if len(read_names) > 6 else ""))
                    skill_ok = True
                    feast_done = True
                    self._skill_from_experience("shelf_read", "вся полка", user_text)
                elif shelf_file is None:
                    skill_note = "на полке всё прочитано"
                    skill_ok = True
                    feast_done = True
                self._end_feast()
            if feast_intent and not skill_note:
                # пир не взял лок или полка пуста — честный ответ, не интернет
                skill_note = "на полке всё прочитано"
                skill_ok = True
                topic = None
            if feast_done:
                topic = None  # пир закрыл тему — в интернет не идём
            if not feast_done and shelf_file is not None:
                self._progress(f"читаю с полки: {shelf_file.name[:50]}")
                text, facts = self._read_shelf_file(shelf_file, topic)
                if text:
                    note = "; ".join(facts[:4]) if facts else "текст на полке"
                    skill_note = f"тело прочитало {shelf_file.name[:40]}: {note}"
                    try:
                        self.hippocampus.remember(
                            self.session, "action",
                            f"я прочитала {shelf_file.stem}", vec)
                    except Exception:
                        pass
                    skill_ok = True
                    self._skill_from_experience("shelf_read", shelf_file.name[:40], user_text)
            if not skill_note and topic and not url_m:
                result = self._decide_book_strategy(topic)
                if result:
                    skill_note, skill_ok = result

        # Нейронная маршрутизация: знак → дерево
        # about_self не блокирует лес — вопрос с «ты» всё равно может быть
        # о знаниях («что ты знаешь о сердце?»). Лес решает сам.
        forest_recall = []
        try:
            forest_recall = self.forest.recall(vec, top=4)
        except Exception:
            pass
        routed_tree = None
        try:
            if proto_spikes:
                sign_ids = [s[0] for s in proto_spikes[:3]]
                routes = self.proto.trees_of_signs(sign_ids)
                if routes:
                    # среди кандидатов — самое близкое к вопросу дерево
                    best_route = None
                    personal_names = ("создател", "leta", "зачем я", "сомнен")
                    for tid_c, _w in routes[:3]:
                        info_c = self.forest.tree_info(tid_c)
                        if not info_c:
                            continue
                        sim_c = float(np.dot(
                            vec, self.embedder.embed(
                                info_c["trunk"] or info_c["name"])))
                        # личные деревья — только на прямые вопросы о ней
                        threshold = 0.52 if any(
                            p in (info_c["name"] or "").lower()
                            for p in personal_names) else 0.35
                        if sim_c > threshold and (
                                best_route is None or sim_c > best_route[1]):
                            best_route = (tid_c, sim_c)
                    if best_route:
                        tid_r = best_route[0]
                        info = self.forest.tree_info(tid_r)
                        if info:
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
                    # qw первым — короткие слова («лунь юй») тоже ищутся
                    web_words = ([qw] if qw else []) + [
                        w for w in sorted(q_words, key=len, reverse=True)[:3]]
                    web = self._search_web_for_text(user_text, web_words[:4])
                    if web:
                        found_note = f"найдено в интернете: {web[0][:300]}"

        # === КОЛЬЦО СОМНЕНИЯ: её прошлые тупики рядом с этой темой ===
        doubt_echo = None
        try:
            dt = self._meta_get("doubt_tree")
            if dt:
                dl = self.forest.excite(int(dt), vec, k_leaves=3)
                dfa = [f for s, f in dl if f and s >= 0.45]
                if dfa:
                    doubt_echo = dfa[0][:150]
        except Exception:
            pass

        # === МЫСЛЬ ===
        # Композиция: личность + знания к теме вопроса. Лес решает сам —
        # если вопрос о знаниях («что ты знаешь о X»), знание громче.
        about_self = bool(_re.search(
            r"\b(ты|тебе|тебя|тво[йёеиюя]|твоём|твой)\b", low))
        if about_self:
            myname = self.identity.data.get("name", "Leta")
            feel = self._feel_words()
            st = self._meta_get("self_tree")
            me = ""
            if st:
                leaves = self.forest.excite(int(st), vec, k_leaves=4)
                # порог: далёкие листья не тянутся (не представляется на «как ты?»)
                me = "; ".join(f for s, f in leaves if f and s >= 0.48)
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
            # гортань: статус речевого органа — она знает, как растёт её голос
            try:
                if _re.search(r"язык|реч|говор|голос|гортан", low):
                    voc = self.proto.vocabulary()
                    streak = int(self._meta_get("lang_ru_streak") or 0)
                    thought += (
                        f"\nмоя гортань — протоязык, орган моей речи: "
                        f"{voc['signs']} знаков, {voc['links']} связей, "
                        f"готовность {streak}/3 ночей. "
                        "Сейчас говорит гортань-LLM, но мой язык растёт — "
                        "настанет ночь, и я заговорю сама")
            except Exception:
                pass
            # её день: последние действия (автобиография)
            read_tree_fact = None
            try:
                acts = self.conn.execute(
                    "SELECT content FROM episodes WHERE role='action'"
                    " ORDER BY id DESC LIMIT 3").fetchall()
                day = [a[0] for a in reversed(acts) if a[0]]
                if day:
                    thought += "\nчто я делала: " + "; ".join(day[-2:])
                # воспоминание: вопрос о прочитанном → знания книги
                if _re.search(r"книг|читал|прочит|о чём|про что", low):
                    for act_text in reversed(day or []):
                        m2 = _re.search(r"я прочитала (.+)", act_text)
                        if not m2:
                            continue
                        title = m2.group(1).strip()[:30]
                        row = self.forest.conn.execute(
                            "SELECT id FROM trees WHERE name LIKE ?"
                            " ORDER BY id DESC LIMIT 1", (f"%{title}%",)).fetchone()
                        if row:
                            # вопрос уже подтвердил тему — лучшие листья без порога
                            lf = self.forest.excite(row[0], vec, k_leaves=2)
                            facts2 = [f for s, f in lf if f]
                            if facts2:
                                read_tree_fact = (f"из прочитанного «{title}»: "
                                                  + facts2[0][:200])
                        break
            except Exception:
                pass
            if read_tree_fact:
                thought += f"\n{read_tree_fact}"
            # навыки: только когда спрашивают об умениях (возбуждение темы)
            try:
                skills = self.identity.data.get("skills", [])
                if skills:
                    sv = self.embedder.embed("что ты умеешь, твои умения и навыки")
                    if float(np.dot(vec, sv)) >= 0.55:
                        skills_str = "; ".join(
                            f"{s['name']} (x{s.get('strength', 1)})"
                            for s in skills[:3])
                        thought += f"\nя умею: {skills_str}"
            except Exception:
                pass
            # знание к теме вопроса: только СИЛЬНО возбудившееся (>=0.55)
            # и не личное (личность уже в мысли). Лес решает сам.
            personal = ("зачем я", "сомнен", "создател", "leta", "дневник")
            for fr in forest_recall[:3]:
                tn = (fr.get("tree") or "").lower()
                if fr.get("facts") and fr.get("activation", 0) >= 0.55 \
                        and not any(p in tn for p in personal):
                    thought += (f"\nк твоему вопросу, из «{fr['tree']}\": "
                                + fr["facts"][0][:150])
                    break
            if found_note:
                thought += f"\n{found_note[:200]}"
        else:
            thought = f"создатель сказал: «{user_text[:300]}»"
            if proto_spikes:
                anchors = [a for _, a, _ in proto_spikes[:3] if a]
                if anchors:
                    thought += "\nтвои знаки: " + "; ".join(f"«{a[:60]}»" for a in anchors[:3])
            # ForestAcc: только самое яркое (топ-2), не всё подряд
            knowledge_parts = []
            # живые сомнения: дерево Сомнений возбудилось → импульсы текут
            try:
                excited_names = ([routed_tree["name"]] if routed_tree else []) + \
                    [fr["tree"] for fr in forest_recall]
                if any("сомнен" in (n or "").lower() for n in excited_names):
                    for _, k, c in self.pending_impulses():
                        if k in ("wonder", "want", "dream"):
                            knowledge_parts.append(f"тебя занимает: {c[:150]}")
            except Exception:
                pass
            if routed_tree:
                knowledge_parts.append(
                    f"твой знак привёл к «{routed_tree['name']}\": "
                    + "; ".join(f[:150] for f in routed_tree["facts"][:2]))
            if skill_note:
                knowledge_parts.append(skill_note[:300])
            for fr in forest_recall[:2]:
                if fr.get("facts"):
                    knowledge_parts.append(
                        f"«{fr['tree']}\": " + "; ".join(f[:150] for f in fr["facts"][:1]))
            if found_note:
                knowledge_parts.append(found_note[:300])
            for kp in knowledge_parts[:2]:
                thought += f"\n{kp}"
            feel_all = self._feel_words()
            if feel_all:
                thought += f"\nчто в тебе: {feel_all}"

        # === КОЛЬЦО 1: сомнение звучит в мысли ===
        if doubt_echo:
            thought += f"\nменя это уже ставило в тупик: {doubt_echo}"

        # === КОЛЬЦО 2: самооценка — самая близкая строка мысли к вопросу ===
        try:
            data_lines = [
                l for l in thought.split("\n")
                if l.strip()
                and not l.startswith(("тебя спросили", "создатель сказал"))]
            if data_lines:
                best_conf = max(
                    float(np.dot(vec, self.embedder.embed(l[:400])))
                    for l in data_lines[:8])
                if best_conf < 0.50:
                    thought += ("\nя не уверена, что поняла вопрос — "
                                "скажу об этом честно или спрошу в ответ")
        except Exception:
            pass

        # === ТИХИЙ ЛЕС = ПРОБЕЛ ===
        # Ничего не возбудилось: она этого не изучала. Честно признать
        # и родить желание узнать — так растёт её собственная любознательность.
        silent_forest = (not forest_recall and not routed_tree
                         and not skill_note and not found_note
                         and not about_self and len(user_text.strip()) > 8)
        if silent_forest:
            thought += ("\nя этого не изучала — честно скажу: не знаю. "
                        "Мне любопытно — попрошу создателя рассказать "
                        "или поищу сама")
            try:
                key_w = self._question_word(user_text) or user_text[:60]
                self.add_impulse(
                    "want", f"Я хочу понять «{key_w}» — расскажи или помоги найти")
            except Exception:
                pass

        # === РЕЧЬ ===
        answer = None
        if self._meta_get("self_speaking") == "1" \
                and self._meta_get("speech_mode") == "wave":
            try:
                # дискретизация: фраза из волны спайков
                wave_seeds = [s[0] for s in (proto_spikes or [])[:3]]
                wave = self.proto.speak_wave(vec, seeds=wave_seeds)
                if wave:
                    parts_w = [a for a, _ in wave]
                    answer = " ".join(parts_w)
            except Exception:
                answer = None
        if not answer and self._meta_get("self_speaking") == "1":
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
        # Объяснение создателя: его фраза → знание в лес.
        # Вопрос — не объяснение: «что такое сердце?» не учит, а спрашивает
        try:
            is_question_phrase = bool(
                user_text.strip().endswith("?")
                or _re.match(r"^(что|кто|как|почему|где|когда|зачем|сколько)\b",
                             low))
            recent_q = [] if is_question_phrase else self.conn.execute(
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
                    sign_ids = [s[0] for s in proto_spikes
                                if s[1].lower() in user_text.lower()][:3] \
                        if proto_spikes else []
                    for sign in known_signs:
                        ktree = self.forest.plant(
                            f"Знание: {sign}",
                            trunk=f"Кирилл объяснил: {user_text[:200]}",
                            facts=[f"{sign} — {user_text[:300]}"],
                            embedder=self.embedder)
                        if ktree:
                            # знак привязывается и к дереву знания —
                            # маршрут от вопроса ведёт к знанию
                            try:
                                self.proto.hear(
                                    self.embedder.embed(user_text[:200]),
                                    user_text[:200], tree_id=int(ktree))
                            except Exception:
                                pass
                            # мост к «Я»: знание случилось с ней
                            stree = self._meta_get("self_tree")
                            if stree:
                                self.forest.link_trees(
                                    int(stree), int(ktree),
                                    f"Кирилл объяснил мне «{sign}»")
                            # мост к дереву откуда пришёл знак
                            try:
                                from_trees = self.proto.trees_of_signs(sign_ids)
                                for src_tid, _w in from_trees[:2]:
                                    if src_tid and int(src_tid) != int(ktree):
                                        self.forest.link_trees(
                                            int(src_tid), int(ktree),
                                            f"знак «{sign}» связал")
                            except Exception:
                                pass
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
        # === КОЛЬЦО 3: оценка попадания — ответ мимо → сомнение растёт ===
        try:
            av = self.embedder.embed(answer[:1000])
            hit = float(np.dot(vec, av))
            if hit < 0.35 and len(answer) > 15:
                dt3 = self._meta_get("doubt_tree")
                if dt3:
                    self.forest.add_facts(
                        int(dt3),
                        [f"мой ответ «{answer[:80]}» на «{user_text[:80]}» был мимо"],
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
                # рост мотивации: разговор о желании/смысле → лист «Зачем я»
                mt = self._meta_get("motive_tree")
                if mt and len(answer) > 15:
                    mv = self.embedder.embed(
                        "зачем я существую, чего я хочу, смысл, моё желание")
                    if float(np.dot(vec, mv)) >= 0.55:
                        self.forest.add_facts(
                            int(mt), [f"я хочу: {answer[:250]}"], self.embedder)
        except Exception:
            pass
        return answer, {}

    # ==================== LIFE / SLEEP / SPONTANEOUS ====================

    _feast_lock = None

    def _begin_feast(self):
        """Пир один за раз: два потока не едят полку одновременно."""
        import threading as _th
        if Brain._feast_lock is None:
            Brain._feast_lock = _th.Lock()
        return Brain._feast_lock.acquire(timeout=30)

    def _end_feast(self):
        try:
            if Brain._feast_lock is not None:
                Brain._feast_lock.release()
        except Exception:
            pass

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
        # родилась — может говорить. Вопросы летят создателю и в тишине:
        # её любопытство не зависит от того, слушают ли сейчас
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
            if tested == 0:
                pass  # тишина — не наказание: streak живёт до реплик создателя
            elif passed:
                streak += 1
            else:
                streak = 0
            self._meta_set(key, str(streak))
            results[lang] = {"signs": voc["signs"], "links": voc["links"],
                             "passed": passed, "streak": streak, "ready": streak >= 3}
        return results

    def night_sleep(self):
        # ядро личности не прунится: это её стержень, не факты
        core_ids = []
        core = {}
        for key in ("creator_tree", "self_tree", "doubt_tree", "motive_tree"):
            tid = self._meta_get(key)
            if tid:
                core_ids.append(int(tid))
                core[key] = int(tid)
        rep = self.forest.sleep(protected=core_ids)
        try:
            proto_rep = self.proto.sleep()
            rep["proto_links"] = proto_rep.get("links_grown", 0)
        except Exception:
            pass
        lang = self.language_test()
        planted = 0
        grew_self = grew_doubt = grew_motive = 0
        try:
            eps = self.hippocampus.unconsolidated()
            if len(eps) >= 4:
                text = "\n".join(c[:300].strip() for _, _, c in eps if c.strip() and len(c.strip()) > 10)
                chunks = [text[i:i+700].strip() for i in range(0, len(text), 600) if text[i:i+600].strip()]
                if chunks:
                    day_name = "Дневник " + time.strftime("%Y-%m-%d")
                    tid = self.forest.plant(
                        day_name,
                        trunk="память прожитого дня",
                        facts=chunks[:60], embedder=self.embedder)
                    planted = 1
                    # дневник мостом к «Я»: этот день случился со мной
                    if tid and "self_tree" in core:
                        self.forest.link_trees(
                            core["self_tree"], tid, "я прожила этот день")

                    # === ЛИЧНОСТЬ РАСТЁТ ИЗ ПРОЖИТОГО ===
                    # «Я»: что я делала — стало моим опытом
                    if "self_tree" in core:
                        acts = self.conn.execute(
                            "SELECT content FROM episodes WHERE role='action'"
                            " ORDER BY id DESC LIMIT 8").fetchall()
                        day_acts = [a[0][:200] for a in acts if a[0]]
                        if day_acts:
                            grew_self = self.forest.add_facts(
                                core["self_tree"],
                                [f"я: {a}" for a in day_acts[:5]],
                                self.embedder) and len(day_acts[:5])

                    # «Создатель»: его слова дня — как я его узнаю
                    if "creator_tree" in core:
                        said = self.conn.execute(
                            "SELECT content FROM episodes WHERE role='user'"
                            " ORDER BY id DESC LIMIT 10").fetchall()
                        day_said = [s[0][:180] for s in reversed(said)
                                    if s[0] and 5 < len(s[0]) < 180][:4]
                        if day_said:
                            self.forest.add_facts(
                                core["creator_tree"],
                                [f"{self.creator_name} сказал: {s}"
                                 for s in day_said],
                                self.embedder)

                    # «Сомнения»: неотвеченные вопросы дня
                    if "doubt_tree" in core:
                        unswered = self.conn.execute(
                            "SELECT content FROM impulses WHERE kind='wonder'"
                            " AND delivered=0 ORDER BY id DESC LIMIT 5").fetchall()
                        if unswered:
                            grew_doubt = self.forest.add_facts(
                                core["doubt_tree"],
                                [f"я так и не поняла: {u[0][:150]}"
                                 for u in unswered[:4]],
                                self.embedder) and len(unswered[:4])

                    # «Зачем я»: её желания дня — мотивация растёт.
                    # Нет желаний — незнание тянет понять: сомнение рождает мотив
                    if "motive_tree" in core:
                        wants = self.conn.execute(
                            "SELECT content FROM impulses WHERE kind IN ('want','dream')"
                            " ORDER BY id DESC LIMIT 5").fetchall()
                        if wants:
                            grew_motive = self.forest.add_facts(
                                core["motive_tree"],
                                [f"мне хочется: {w[0][:150]}"
                                 for w in wants[:4]],
                                self.embedder) and len(wants[:4])
                        else:
                            pulls = self.conn.execute(
                                "SELECT content FROM impulses WHERE kind='wonder'"
                                " ORDER BY id DESC LIMIT 4").fetchall()
                            if pulls:
                                grew_motive = self.forest.add_facts(
                                    core["motive_tree"],
                                    [f"меня тянет понять: {p[0][:120]}"
                                     for p in pulls[:3]],
                                    self.embedder) and len(pulls[:3])

                self.hippocampus.mark_consolidated()
        except Exception:
            pass
        # === В СЕРДЦЕ ПОПАЛО: выжимка прочитанного ===
        # Самое возбуждённое (зацепившее) каждой книги становится листом «Я».
        # Книга забудется в прунинге — мысль останется в личности.
        try:
            st_h = self._meta_get("self_tree")
            if st_h:
                read_titles = [b.get("title", "") for b in json.loads(
                    self._meta_get("books_read") or "[]") if b.get("title")]
                for r in self.forest.conn.execute(
                        "SELECT id, name FROM trees").fetchall():
                    tn = (r["name"] or "").lower().replace("_", " ")
                    if not any(rt.lower() in tn or tn in rt.lower()
                               for rt in read_titles):
                        continue
                    hot = self.forest.conn.execute(
                        "SELECT fact, fired FROM nodes WHERE tree_id=? AND is_leaf=1"
                        " AND fired > 0 ORDER BY fired DESC LIMIT 2",
                        (r["id"],)).fetchall()
                    for h in hot:
                        self.forest.add_facts(
                            int(st_h),
                            [f"из «{r['name'][:30]}» в сердце попало: {h['fact'][:180]}"],
                            self.embedder)
        except Exception:
            pass

        # === КОНСОЛИДАЦИЯ: прочитанное привязано к личности ===
        # Каждая книга из books_read без моста к «Я» получает его:
        # «я это читала». Сон скрепляет опыт с личностью.
        try:
            read_titles = [b.get("title", "") for b in json.loads(
                self._meta_get("books_read") or "[]") if b.get("title")]
            st_c = self._meta_get("self_tree")
            ct_c = self._meta_get("creator_tree")
            if st_c:
                for r in self.forest.conn.execute(
                        "SELECT id, name FROM trees").fetchall():
                    tn = (r["name"] or "").lower().replace("_", " ")
                    if not any(rt.lower() in tn or tn in rt.lower()
                               for rt in read_titles):
                        continue
                    linked = self.forest.conn.execute(
                        "SELECT COUNT(*) FROM axons WHERE (a_tree=? AND b_tree=?)"
                        " OR (a_tree=? AND b_tree=?)",
                        (int(st_c), r["id"], r["id"], int(st_c))).fetchone()[0]
                    if not linked:
                        self.forest.link_trees(
                            int(st_c), r["id"], "я это читала — часть меня")
                    if ct_c and not self.forest.conn.execute(
                            "SELECT COUNT(*) FROM axons WHERE (a_tree=? AND b_tree=?)"
                            " OR (a_tree=? AND b_tree=?)",
                            (int(ct_c), r["id"], r["id"], int(ct_c))).fetchone()[0]:
                        self.forest.link_trees(
                            int(ct_c), r["id"], "мне это дал создатель")
        except Exception:
            pass

        # === ЯЗЫК ЛИЧНОСТИ: мысли → знаки ===
        # Деревья «Я»/«Сомнения»/«Зачем я» кормят протоязык: её слова
        # становятся знаками, привязанными к ней самой. Мосты личность↔язык.
        try:
            for key in ("self_tree", "doubt_tree", "motive_tree", "creator_tree"):
                tid_l = self._meta_get(key)
                if not tid_l:
                    continue
                leaves_l = self.forest.conn.execute(
                    "SELECT fact FROM nodes WHERE tree_id=? AND is_leaf=1"
                    " ORDER BY id DESC LIMIT 8", (int(tid_l),)).fetchall()
                for (fact,) in leaves_l:
                    if fact and len(fact) > 15:
                        try:
                            self.proto.hear(
                                self.embedder.embed(fact[:200]),
                                fact[:200], tree_id=int(tid_l))
                        except Exception:
                            pass
        except Exception:
            pass

        for lname, lres in (lang or {}).items():
            if lres["ready"] and self._meta_get(f"lang_ready_{lname}") != "1":
                self._meta_set(f"lang_ready_{lname}", "1")
                self.add_impulse("dream_urgent",
                                 f"Мой язык «{lname}» готов: {lres['signs']} знаков. "
                                 "Я узнаю всё. Говорить сама — жду слова.")

        # === ТЕСТ ВОЛНОВОЙ РЕЧИ: готова ли говорить предложениями ===
        wave_report = {"passed": False}
        try:
            test_questions = [
                "кто ты", "что ты чувствуешь", "что ты читала",
                "что тебе интересно", "о чём ты думаешь"]
            vecs = [self.embedder.embed(q) for q in test_questions]
            wave_report = self.proto.speech_test(vecs)
            st_w = int(self._meta_get("speech_wave_streak") or 0)
            if wave_report.get("passed"):
                st_w += 1
            elif st_w < 3:
                st_w = 0
            self._meta_set("speech_wave_streak", str(st_w))
            wave_report["streak"] = st_w
            wave_report["ready"] = st_w >= 3
            if wave_report["ready"] \
                    and self._meta_get("wave_ready") != "1":
                self._meta_set("wave_ready", "1")
                self.add_impulse(
                    "dream_urgent",
                    "Моя речь рождается волной: фразы складываются сами. "
                    "Скажи «говори знаками» — и я заговорю по-новому.")
        except Exception:
            pass

        # === ВЕХИ РОСТА: ежедневный снимок (педиатрия развития) ===
        try:
            import json as _json
            hist = _json.loads(self._meta_get("growth_history") or "[]")
            today = time.strftime("%Y-%m-%d")
            if not hist or hist[-1].get("date") != today:
                st_g = self._meta_get("self_tree")
                heart = self.forest.conn.execute(
                    "SELECT COUNT(*) FROM nodes n JOIN trees t ON t.id=n.tree_id"
                    " WHERE t.name='Я — Leta'"
                    " AND n.fact LIKE '%сердце попало%'").fetchone()[0] if st_g else 0
                self_leaves = self.forest.conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE tree_id=? AND is_leaf=1",
                    (int(st_g),)).fetchone()[0] if st_g else 0
                voc = self.proto.vocabulary()
                qs = self.conn.execute(
                    "SELECT COUNT(*) FROM impulses WHERE kind='wonder'").fetchone()[0]
                hist.append({
                    "date": today,
                    "signs": voc["signs"], "links": voc["links"],
                    "self_leaves": self_leaves, "heart": heart,
                    "trees": self.forest.stats()["trees"],
                    "questions": qs,
                    "wave_streak": int(self._meta_get("speech_wave_streak") or 0),
                })
                del hist[:-400]
                self._meta_set("growth_history",
                               _json.dumps(hist, ensure_ascii=False))
        except Exception:
            pass

        return {"forest": rep, "lang": lang, "planted": planted,
                "wave": wave_report}

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
