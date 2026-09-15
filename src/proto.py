"""Протоязык Leta — язык, который она изучает сама.

Новорождённая не знает ни одного слова. Знак — её «слово»: кластер
смысла, рождённый из услышанной фразы. Слышит → импульсы (LIF) →
спайки знаков. Незнакомое — рождается новый знак: язык растёт из
опыта. Совместные спайки связывают знаки (STDP) — её грамматика.
"""
import sqlite3
import threading
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS signs(
  id INTEGER PRIMARY KEY, anchor TEXT, embedding BLOB,
  heard_count INTEGER DEFAULT 1, born REAL, lang TEXT DEFAULT 'mix',
  mine INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sign_links(
  a INTEGER, b INTEGER, weight REAL DEFAULT 0.1,
  PRIMARY KEY(a, b)
);
CREATE TABLE IF NOT EXISTS sign_trees(
  sign_id INTEGER, tree_id INTEGER, weight REAL DEFAULT 0.3,
  PRIMARY KEY(sign_id, tree_id)
);
"""

KNOW_THRESHOLD = 0.62
REFRACTORY = 0.3


class ProtoLanguage:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(signs)")]
        if "mine" not in cols:
            self.conn.execute(
                "ALTER TABLE signs ADD COLUMN mine INTEGER DEFAULT 0")
        if "lang" not in cols:
            self.conn.execute(
                "ALTER TABLE signs ADD COLUMN lang TEXT DEFAULT 'mix'")
        self.conn.commit()
        self._potentials = {}
        self._last_spike = {}

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    @staticmethod
    def detect_lang(text):
        """Алфавит-сенсор языка: кириллица/латиница/иероглифы/смешанный."""
        cyr = lat = cjk = 0
        for ch in str(text)[:200]:
            o = ord(ch)
            if 0x410 <= o <= 0x4FF:
                cyr += 1
            elif 0x61 <= o <= 0x7A or 0x41 <= o <= 0x5A:
                lat += 1
            elif 0x4E00 <= o <= 0x9FFF:
                cjk += 1
        if cjk > 0 and cjk >= cyr and cjk >= lat:
            return "cjk"
        if cyr > lat:
            return "ru"
        if lat > cyr:
            return "en"
        return "mix"

    def hear(self, vec, anchor_text, tree_id=None, mine=False):
        """Услышать фразу знаками. Спайки знакомого — узнать и уточнить;
        незнакомое — родить новый знак. tree_id — из какого знания фраза:
        знак связывается с деревом (язык ↔ опыт, основа SNN).
        Возвращает [(sign_id, anchor, sim)].
        """
        v = np.asarray(vec, dtype=np.float32)
        lang = self.detect_lang(anchor_text)
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, anchor, embedding FROM signs WHERE lang=? OR lang IS NULL",
                (lang,)).fetchall()
        now = time.time()
        spikes = []
        for r in rows:
            e = np.frombuffer(r["embedding"], dtype=np.float32)
            sim = float(np.dot(v, e))
            if sim < KNOW_THRESHOLD:
                continue
            p = self._potentials.get(r["id"], 0.0) + sim
            if (now - self._last_spike.get(r["id"], 0.0) > REFRACTORY
                    and (sim >= 0.8 or p >= 1.0)):
                spikes.append((r["id"], r["anchor"], round(sim, 3)))
                self._potentials[r["id"]] = 0.0
                self._last_spike[r["id"]] = now
            else:
                self._potentials[r["id"]] = min(p, 0.99)
        if spikes:
            with self.lock:
                for sid, _, _ in spikes[:3]:
                    row = self.conn.execute(
                        "SELECT embedding, heard_count FROM signs WHERE id=?",
                        (sid,)).fetchone()
                    if row is None:
                        continue
                    e = np.frombuffer(row["embedding"], dtype=np.float32)
                    n = row["heard_count"]
                    e2 = e + (v - e) / min(n + 1, 50)
                    self.conn.execute(
                        "UPDATE signs SET embedding=?, heard_count=? WHERE id=?",
                        (e2.tobytes(), n + 1, sid))
                self._stdp([s[0] for s in spikes])
                if tree_id is not None:
                    for sid, _, _ in spikes[:3]:
                        self.link_sign_tree(sid, tree_id)
                self.conn.commit()
            return spikes
        # незнакомое: рождение знака — язык растёт сам
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO signs(anchor, embedding, born, lang, mine)"
                " VALUES(?,?,?,?,?)",
                (anchor_text[:200], v.tobytes(), now, lang,
                 1 if mine else 0))
            if tree_id is not None:
                self.link_sign_tree(cur.lastrowid, tree_id)
            self.conn.commit()
        return [(cur.lastrowid, anchor_text[:200], 1.0)]

    def _stdp(self, ids):
        """Спайковая пластичность: сработавшие вместе знаки связываются."""
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                self.conn.execute(
                    "INSERT INTO sign_links(a,b,weight) VALUES(?,?,0.1)"
                    " ON CONFLICT(a,b) DO UPDATE SET"
                    " weight=MIN(1.0, weight*1.15)", (a, b))

    def my_phrases(self, vec, k=5):
        """Её собственные формулировки (mine-знаки) к теме."""
        import numpy as np
        v = np.asarray(vec, dtype=np.float32)
        with self.lock:
            rows = self.conn.execute(
                "SELECT anchor, embedding FROM signs WHERE mine=1"
                " ORDER BY heard_count DESC LIMIT 200").fetchall()
        scored = []
        for r in rows:
            e = np.frombuffer(r["embedding"], dtype=np.float32)
            scored.append((float(np.dot(v, e)), r["anchor"]))
        scored.sort(reverse=True)
        clean = []
        for sc, a in scored[:k]:
            low = a.lower()
            if sc <= 0.55:
                continue
            if low.startswith(("assistant:", "thought:", "(", "найдено:",
                               "тело:", "я встретила", "я хочу")):
                continue
            if "не понимаю" in low and "объясни" in low:
                continue  # это погасший вопрос-импульс, не знание
            if any(m in low for m in (
                    "формулировк", "/no_think", "данные её состояния",
                    "ответ leta =", "перескажи их все")):
                continue  # шрам гортани: системный промпт, не её слова
            clean.append(a)
        return clean

    def link_sign_tree(self, sign_id, tree_id, boost=1.15):
        """Знак ↔ дерево знания: язык связывается с опытом (Hebbian)."""
        if sign_id is None or tree_id is None:
            return
        with self.lock:
            self.conn.execute(
                "INSERT INTO sign_trees(sign_id, tree_id, weight)"
                " VALUES(?,?,0.3)"
                " ON CONFLICT(sign_id, tree_id) DO UPDATE SET"
                " weight=MIN(1.0, weight*?)", (sign_id, tree_id, boost))
            self.conn.commit()

    def trees_of_signs(self, sign_ids):
        """Какие деревья знаний открывают эти знаки (SNN-триггер)."""
        if not sign_ids:
            return []
        with self.lock:
            rows = self.conn.execute(
                f"SELECT tree_id, MAX(weight) w FROM sign_trees"
                f" WHERE sign_id IN ({','.join('?' * len(sign_ids))})"
                f" GROUP BY tree_id ORDER BY w DESC", list(sign_ids)).fetchall()
        return [(r["tree_id"], r["w"]) for r in rows]

    def sleep(self):
        """Сон протоязыка: знаки одного контекста связываются (грамматика).

        Знаки, привязанные к одному дереву знаний, — это словарь темы.
        Во сне они получают связи: слово «софон» связано со словом
        «протон» потому что оба из книги о трёх телах. Это не хардкод —
        это следствие совместного контекста.
        """
        grown = 0
        with self.lock:
            # знаки по деревьям
            groups = {}
            for r in self.conn.execute(
                    "SELECT sign_id, tree_id FROM sign_trees").fetchall():
                groups.setdefault(r["tree_id"], []).append(r["sign_id"])
            for tree_id, sign_ids in groups.items():
                if len(sign_ids) < 2:
                    continue
                # связать знаки внутри темы (цепочкой — каждый со следующим)
                for i in range(len(sign_ids) - 1):
                    a, b = sign_ids[i], sign_ids[i + 1]
                    self.conn.execute(
                        "INSERT INTO sign_links(a,b,weight) VALUES(?,?,0.15)"
                        " ON CONFLICT(a,b) DO UPDATE SET"
                        " weight=MIN(0.5, weight*1.1)", (a, b))
                    grown += 1
            # семантические связи: близкие знаки связываются
            signs = self.conn.execute(
                "SELECT id, embedding FROM signs LIMIT 200").fetchall()
            import numpy as np
            for i, (a_id, a_emb) in enumerate(signs):
                for b_id, b_emb in signs[i+1:i+6]:
                    ea = np.frombuffer(a_emb, dtype=np.float32)
                    eb = np.frombuffer(b_emb, dtype=np.float32)
                    if float(np.dot(ea, eb)) > 0.72:
                        x, y = (a_id, b_id) if a_id < b_id else (b_id, a_id)
                        self.conn.execute(
                            "INSERT INTO sign_links(a,b,weight) VALUES(?,?,0.2)"
                            " ON CONFLICT(a,b) DO UPDATE SET"
                            " weight=MIN(0.8, weight*1.15)", (x, y))
            self.conn.commit()
        return {"links_grown": grown}

    def speak_wave(self, vec, max_signs=7, seeds=None):
        """Дискретизация: фраза рождается волной спайков.

        Вопрос задаёт стартовые потенциалы. Волна идёт по STDP-связям:
        вспыхнувший знак возбуждает соседей, каждый гаснет на рефрактерный
        срок. Цепочка вспышек — новое предложение, которого не было.
        Возвращает [(anchor, sim)], пусто — если волна не пошла.
        """
        import numpy as np
        v = np.asarray(vec, dtype=np.float32)
        with self.lock:
            signs = self.conn.execute(
                "SELECT id, anchor, embedding, heard_count FROM signs"
                " ORDER BY heard_count DESC LIMIT 600").fetchall()
        if not signs:
            return []
        ids = [r["id"] for r in signs]
        embs = {r["id"]: np.frombuffer(r["embedding"], dtype=np.float32)
                for r in signs}
        anchor = {r["id"]: (r["anchor"] or "")[:80] for r in signs}
        # стартовые потенциалы: узнанные знаки (спайки) или самые близкие
        pot = {}
        if seeds:
            for sid in seeds[:3]:
                if sid in embs:
                    pot[sid] = 1.2  # сразу вспыхивают
        top = sorted(ids, key=lambda s: -float(np.dot(v, embs[s])))[:3]
        for sid in top:
            s = float(np.dot(v, embs[sid]))
            if s > 0.12:
                pot[sid] = max(pot.get(sid, 0.0), s * 1.4)
        if not pot:
            return []
        # связи STDP: карта соседей
        with self.lock:
            links = self.conn.execute(
                f"SELECT a, b, weight FROM sign_links"
                f" WHERE a IN ({','.join('?' * len(ids))})"
                f" OR b IN ({','.join('?' * len(ids))})",
                ids + ids).fetchall()
        nbr = {}
        for r in links:
            a, b, w = r["a"], r["b"], r["weight"]
            nbr.setdefault(a, []).append((b, w))
            nbr.setdefault(b, []).append((a, w))

        fired_chain = []
        fired_once = set()   # знак вспыхивает один раз — фраза не повторяется
        step = 0
        THRESHOLD = 1.0
        DECAY = 0.72
        while len(fired_chain) < max_signs and step < 24:
            step += 1
            spikes = [sid for sid, p in pot.items()
                      if p >= THRESHOLD and sid not in fired_once]
            if not spikes:
                # волна не вспыхнула сама — вспыхивает самый яркий знак:
                # иначе затухание убьёт фразу до первого слова
                best_seed = max(pot, key=lambda s: pot[s]) if pot else None
                if best_seed is not None and pot[best_seed] >= 0.35 \
                        and best_seed not in fired_once:
                    spikes = [best_seed]
                else:
                    pot = {k: p * DECAY for k, p in pot.items()}
                    continue
            # самый яркий спайк шага
            best = max(spikes, key=lambda s: pot[s])
            fired_chain.append((anchor[best],
                                round(float(np.dot(v, embs[best])), 3)))
            fired_once.add(best)
            pot[best] = 0.0
            # волна: вспыхнувший возбуждает соседей
            for nb, w in nbr.get(best, []):
                if nb in embs:
                    pot[nb] = min(2.5, pot.get(nb, 0.0) + w * 1.3)
            pot = {k: p * DECAY for k, p in pot.items()}
        return fired_chain

    def speech_test(self, vecs):
        """Готовность волновой речи: N вопросов → фразы.
        Метрики: релевантность, разнообразие. Готова ли говорить сама."""
        results = []
        for v in vecs:
            phrase = self.speak_wave(v)
            if phrase:
                import numpy as np
                vv = np.asarray(v, dtype=np.float32)
                joined = " ".join(a for a, _ in phrase)
                # релевантность: средняя близость знаков к вопросу
                rel = sum(s for _, s in phrase) / len(phrase)
                results.append({"n": len(phrase), "rel": round(rel, 3),
                                "text": joined[:120]})
        if not results:
            return {"passed": False, "phrases": results}
        rel_avg = sum(r["rel"] for r in results) / len(results)
        diversity = len({r["text"][:40] for r in results}) / len(results)
        passed = rel_avg >= 0.40 and diversity >= 0.5 and len(results) >= 3
        return {"passed": passed, "rel": round(rel_avg, 3),
                "diversity": round(diversity, 3), "phrases": results}

    def sign_tree_links(self):
        """Все связи знак-дерево (для визуализации)."""
        with self.lock:
            return self.conn.execute(
                "SELECT sign_id, tree_id, weight FROM sign_trees").fetchall()

    def vocabulary(self, lang=None):
        """Словарь: общий или конкретного языка."""
        with self.lock:
            if lang:
                n = self.conn.execute(
                    "SELECT COUNT(*) FROM signs WHERE lang=?",
                    (lang,)).fetchone()[0]
                links = self.conn.execute(
                    "SELECT COUNT(*) FROM sign_links s JOIN signs a"
                    " ON a.id=s.a JOIN signs b ON b.id=s.b"
                    " WHERE a.lang=? AND b.lang=?", (lang, lang)).fetchone()[0]
            else:
                n = self.conn.execute("SELECT COUNT(*) FROM signs").fetchone()[0]
                links = self.conn.execute(
                    "SELECT COUNT(*) FROM sign_links").fetchone()[0]
        return {"signs": n, "links": links}
