"""Лес Leta — мозг на деревьях решений (архитектура: нейрон = дерево).

Сигнал — эмбеддинг (bge-m3, сетчатка). Нейрон-знание = дерево:
trunk (суть), Z-сплиты (смысловые развилки), листья-факты.
Дендриты = связи листьев между деревьями. Аксон = мост дерево→дерево
(Hebbian: совместная активация усиливает). ForestAcc = агрегация
возбуждений леса — что «загорелось».
"""
import heapq
import sqlite3
import threading
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS trees(
  id INTEGER PRIMARY KEY, name TEXT, trunk TEXT,
  created REAL, excitements INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nodes(
  id INTEGER PRIMARY KEY, tree_id INTEGER, parent_id INTEGER, side INTEGER,
  direction BLOB, threshold REAL, is_leaf INTEGER, fact TEXT, embedding BLOB,
  fired INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS axons(
  id INTEGER PRIMARY KEY, a_tree INTEGER, b_tree INTEGER,
  name TEXT, weight REAL DEFAULT 0.3, fired REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_nodes_tree ON nodes(tree_id);
CREATE INDEX IF NOT EXISTS idx_axons_pair ON axons(a_tree, b_tree);
"""

MAX_LEAF_FACTS = 3
MAX_DEPTH = 7
LEAVES_PER_EXCITE = 3
MIN_ACTIVATION = 0.45
DEFAULT_SPROUT_SIM = 0.62
SLEEP_BRIDGE_SIM = 0.72


class Forest:
    """Лес нейронов-деревьев. Потокобезопасен."""

    def __init__(self, path, sprout_sim=DEFAULT_SPROUT_SIM):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.sprout_sim = sprout_sim
        self.conn.executescript(SCHEMA)
        # миграция: колонка fired у старых баз
        cols = [r["name"] for r in self.conn.execute("PRAGMA table_info(nodes)")]
        if "fired" not in cols:
            self.conn.execute("ALTER TABLE nodes ADD COLUMN fired INTEGER DEFAULT 0")
        self.conn.commit()

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    # ============ РОЖДЕНИЕ ДЕРЕВА-НЕЙРОНА ============

    def plant(self, name, trunk, facts, embedder):
        """Вырастить нейрон из фактов. facts = [текст,...]. Возвращает id."""
        items = []
        for f in facts:
            f = str(f).strip()
            if len(f) > 3:
                items.append((f, np.asarray(embedder.embed(f), dtype=np.float32)))
        if not items:
            return None
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO trees(name, trunk, created) VALUES(?,?,?)",
                (name, trunk, time.time()))
            tree_id = cur.lastrowid
            self._grow(tree_id, None, None, items, 0)
            self.conn.commit()
        self._sprout_dendrites(tree_id, items)
        return tree_id

    def _grow(self, tree_id, parent_id, side, items, depth):
        """Рекурсивный рост: сплит Z = гиперплоскость между кластерами."""
        if len(items) <= MAX_LEAF_FACTS or depth >= MAX_DEPTH:
            for fact, emb in items:
                self.conn.execute(
                    "INSERT INTO nodes(tree_id, parent_id, side, is_leaf, fact, embedding)"
                    " VALUES(?,?,?,1,?,?)",
                    (tree_id, parent_id, side, fact, emb.tobytes()))
            return
        embs = np.stack([e for _, e in items])
        set_a = self._bisect(embs)
        a = [items[i] for i in range(len(items)) if i in set_a]
        b = [items[i] for i in range(len(items)) if i not in set_a]
        if not a or not b:
            for fact, emb in items:
                self.conn.execute(
                    "INSERT INTO nodes(tree_id, parent_id, side, is_leaf, fact, embedding)"
                    " VALUES(?,?,?,1,?,?)",
                    (tree_id, parent_id, side, fact, emb.tobytes()))
            return
        ca = np.mean([e for _, e in a], axis=0)
        cb = np.mean([e for _, e in b], axis=0)
        d = ca - cb
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            for fact, emb in items:
                self.conn.execute(
                    "INSERT INTO nodes(tree_id, parent_id, side, is_leaf, fact, embedding)"
                    " VALUES(?,?,?,1,?,?)",
                    (tree_id, parent_id, side, fact, emb.tobytes()))
            return
        direction = (d / norm).astype(np.float32)
        threshold = float(np.dot((ca + cb) / 2, direction))
        cur = self.conn.execute(
            "INSERT INTO nodes(tree_id, parent_id, side, direction, threshold, is_leaf)"
            " VALUES(?,?,?,?,?,0)",
            (tree_id, parent_id, side, direction.tobytes(), threshold))
        node_id = cur.lastrowid
        self._grow(tree_id, node_id, 0, a, depth + 1)
        self._grow(tree_id, node_id, 1, b, depth + 1)

    @staticmethod
    def _bisect(embs):
        """Разделить векторы на два кластера (Ллойд, k=2). Возвращает set индексов A."""
        mid = embs.mean(axis=0)
        order = np.argsort(embs @ mid)
        half = max(1, len(embs) // 2)
        set_a = set(order[:half].tolist())
        for _ in range(5):
            ca = embs[list(set_a)].mean(axis=0)
            rest = [i for i in range(len(embs)) if i not in set_a]
            if not rest:
                break
            cb = embs[rest].mean(axis=0)
            d = ca - cb
            n = np.linalg.norm(d)
            if n < 1e-6:
                break
            proj = embs @ (d / n)
            thr = float(np.dot((ca + cb) / 2, d / n))
            new_a = set(np.where(proj < thr)[0].tolist())
            if not new_a or len(new_a) == len(embs) or new_a == set_a:
                break
            set_a = new_a
        return set_a

    # ============ ДЕНДРИТЫ И АКСОНЫ ============

    def _sprout_dendrites(self, tree_id, items):
        """Листья нового дерева тянутся к чужим листьям — рождаются мосты."""
        sprouted = 0
        with self.lock:
            rows = self.conn.execute(
                "SELECT n.tree_id, n.fact, n.embedding, t.name FROM nodes n"
                " JOIN trees t ON t.id = n.tree_id"
                " WHERE n.is_leaf=1 AND n.tree_id != ? LIMIT 2000",
                (tree_id,)).fetchall()
            for fact, emb in items:
                for r in rows:
                    oe = np.frombuffer(r["embedding"], dtype=np.float32)
                    if float(np.dot(emb, oe)) > self.sprout_sim:
                        if self._add_axon(tree_id, r["tree_id"],
                                          f"{fact[:60]} ~ {r['fact'][:60]}"):
                            sprouted += 1
                        break
                if sprouted >= 5:
                    break
            self.conn.commit()

    def _add_axon(self, t1, t2, name):
        """Мост между деревьями (без дубликатов). True если родился новый."""
        a, b = (t1, t2) if t1 < t2 else (t2, t1)
        with self.lock:
            row = self.conn.execute(
                "SELECT id FROM axons WHERE a_tree=? AND b_tree=?", (a, b)).fetchone()
            if row:
                return False
            self.conn.execute(
                "INSERT INTO axons(a_tree, b_tree, name, weight, fired) VALUES(?,?,?,0.3,?)",
                (a, b, name, time.time()))
            self.conn.commit()
        return True

    def _strengthen_axon(self, t1, t2, factor=1.06):
        """Hebbian: деревья загорелись вместе — мост сильнее."""
        a, b = (t1, t2) if t1 < t2 else (t2, t1)
        with self.lock:
            self.conn.execute(
                "UPDATE axons SET weight = MIN(1.0, weight * ?), fired = ?"
                " WHERE a_tree=? AND b_tree=?", (factor, time.time(), a, b))

    # ============ ВОЗБУЖДЕНИЕ И FOREST ACC ============

    def excite(self, tree_id, signal, k_leaves=LEAVES_PER_EXCITE):
        """Best-first обход: сигнал идёт по сплитам, листья оцениваются
        сходством ещё в куче — первым достаётся самый близкий лист.

        Возвращает [(похожесть, факт), ...] отсортированно по убыванию.
        """
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM nodes WHERE tree_id=?", (tree_id,)).fetchall()
        if not rows:
            return []
        children = {}
        roots = []
        for r in rows:
            if r["parent_id"] is None:
                roots.append(r)
            else:
                children.setdefault(r["parent_id"], []).append(r)
        if not roots:
            return []

        def priority(penalty, node):
            if node["is_leaf"]:
                emb = np.frombuffer(node["embedding"], dtype=np.float32)
                return penalty - 2.0 * float(np.dot(signal, emb))
            # стимул углубляться: внутренний узел дешевле пустого листа
            return penalty - 0.5

        counter = 0
        heap = []
        for r in roots:
            counter += 1
            heapq.heappush(heap, (priority(0.0, r), counter, r))
        leaves = []
        while heap and len(leaves) < k_leaves:
            _, _, node = heapq.heappop(heap)
            if node["is_leaf"]:
                emb = np.frombuffer(node["embedding"], dtype=np.float32)
                sim = float(np.dot(signal, emb))
                leaves.append((sim, node["fact"]))
                with self.lock:
                    self.conn.execute(
                        "UPDATE nodes SET fired = fired + 1 WHERE id=?",
                        (node["id"],))
                continue
            direction = np.frombuffer(node["direction"], dtype=np.float32)
            proj = float(np.dot(signal, direction))
            margin = abs(proj - node["threshold"])
            for child in children.get(node["id"], []):
                right_side = (child["side"] == 1) == (proj >= node["threshold"])
                pen = 0.0 if right_side else margin
                counter += 1
                heapq.heappush(heap, (priority(pen, child), counter, child))
        return sorted(leaves, reverse=True)

    def recall(self, signal, top=5, min_act=MIN_ACTIVATION, spread=True,
               waves=3):
        """ForestAcc: что загорелось в лесу от сигнала.

        Столбцы-каскад: волна 1 — прямое возбуждение деревьев сигналом,
        волна 2..N — возбуждение через аксионы от загоревшихся (с затуханием).
        Возвращает [{tree, trunk, activation, facts, wave}].
        """
        signal = np.asarray(signal, dtype=np.float32)
        with self.lock:
            trees = self.conn.execute("SELECT * FROM trees").fetchall()
        acts = {}
        facts_by_tree = {}
        wave_of = {}
        tree_by_id = {t["id"]: t for t in trees}
        # волна 1: прямой сигнал
        for t in trees:
            leaves = self.excite(t["id"], signal)
            if not leaves:
                continue
            best = max(s for s, _ in leaves)
            if best >= min_act:
                acts[t["id"]] = best
                facts_by_tree[t["id"]] = sorted(leaves, reverse=True)[:3]
                wave_of[t["id"]] = 1
        # волны 2..N: каскад по аксонам-мостам
        if spread and waves > 1:
            with self.lock:
                axons = self.conn.execute("SELECT * FROM axons").fetchall()
            frontier = dict(acts)
            decay = 0.6
            for wave in range(2, waves + 1):
                nxt = {}
                for ax in axons:
                    for src, dst in ((ax["a_tree"], ax["b_tree"]),
                                     (ax["b_tree"], ax["a_tree"])):
                        if src in frontier and dst not in acts \
                                and dst in tree_by_id:
                            boost = frontier[src] * ax["weight"] * decay
                            if boost >= min_act:
                                nxt[dst] = boost
                                wave_of[dst] = wave
                if not nxt:
                    break
                acts.update(nxt)
                facts_by_tree.update({k: [] for k in nxt})
                frontier = nxt
                decay *= 0.6
        # Hebbian: загоревшиеся вместе — мост крепче
        hot = [t for t, a in acts.items() if a >= min_act + 0.1]
        with self.lock:
            for i in range(len(hot)):
                for j in range(i + 1, len(hot)):
                    self._strengthen_axon(hot[i], hot[j])
            if hot:
                self.conn.execute(
                    f"UPDATE trees SET excitements = excitements + 1"
                    f" WHERE id IN ({','.join('?' * len(hot))})", hot)
            self.conn.commit()
        result = []
        for tid, act in sorted(acts.items(), key=lambda kv: -kv[1])[:top]:
            t = tree_by_id[tid]
            result.append({
                "tree": t["name"],
                "trunk": t["trunk"],
                "activation": round(act, 3),
                "facts": [f for _, f in facts_by_tree.get(tid, [])],
                "wave": wave_of.get(tid, 1),
            })
        return result

    def add_facts(self, tree_id, facts, embedder):
        """Живой рост дерева: новые листья от корня."""
        try:
            with self.lock:
                row = self.conn.execute(
                    "SELECT id FROM nodes WHERE tree_id=? AND parent_id IS NULL",
                    (tree_id,)).fetchone()
                if not row:
                    return None
                root = row["id"]
                existing = [r["fact"] for r in self.conn.execute(
                    "SELECT fact FROM nodes WHERE tree_id=? AND is_leaf=1",
                    (tree_id,)).fetchall()]
                for f in facts[:5]:
                    f = str(f).strip()
                    if len(f) < 4 or f in existing:
                        continue
                    emb = np.asarray(embedder.embed(f), dtype=np.float32)
                    self.conn.execute(
                        "INSERT INTO nodes(tree_id, parent_id, side, is_leaf,"
                        " fact, embedding) VALUES(?,?,0,1,?,?)",
                        (tree_id, root, f[:400], emb.tobytes()))
                self.conn.commit()
            return True
        except Exception:
            return None

    def link_trees(self, t1, t2, name):
        """Явный мост между деревьями (контекстная связь, не поиск похожести)."""
        return self._add_axon(t1, t2, name)

    def sleep(self, days_for_prune=2.0, protected=None):
        """Сон-реструктуризация: прунинг, рост дендритов, затухание мостов.

        protected — id деревьев ядра личности (не прунятся никогда).
        Возвращает отчёт о том, что мозг реально сделал ночью.
        """
        import time as _time
        now = _time.time()
        report = {"pruned": 0, "bridges_grown": 0, "bridges_decayed": 0}
        protected = set(protected or [])
        with self.lock:
            # 1. прунинг: мёртвые листья старых деревьев
            # прунинг: не возбуждавшиеся + отрицательно оценённые
            rows = self.conn.execute(
                "SELECT n.id, n.tree_id FROM nodes n JOIN trees t"
                " ON t.id = n.tree_id WHERE n.is_leaf=1"
                " AND (n.fired <= 0 OR (n.fired < 2 AND t.created < ?))"
                " AND t.created < ?",
                (now - days_for_prune * 86400,
                 now - days_for_prune * 86400)).fetchall()
            for r in rows:
                if r["tree_id"] in protected:
                    continue
                self.conn.execute(
                    "DELETE FROM nodes WHERE id=?", (r["id"],))
                # осиротевших детей нет (лист), пустые деревья не трогаем
            report["pruned"] = len(rows)
            # 2. затухание мостов; слабые умирают.
            # МОСТЫ ЯДРА НЕ ЗАТУХАЮТ: стержень личности — орган, не память
            axon_pairs = {}
            for r2 in self.conn.execute(
                    "SELECT id, a_tree, b_tree FROM axons").fetchall():
                axon_pairs[r2["id"]] = (r2["a_tree"], r2["b_tree"])
            axons = self.conn.execute(
                "SELECT id, weight FROM axons").fetchall()
            for ax in axons:
                pair = axon_pairs.get(ax["id"], (None, None))
                if pair[0] in protected and pair[1] in protected:
                    continue  # мост ядра личности — вечен
                w = ax["weight"] * 0.9
                if w < 0.05:
                    self.conn.execute("DELETE FROM axons WHERE id=?", (ax["id"],))
                    report["bridges_decayed"] += 1
                else:
                    self.conn.execute(
                        "UPDATE axons SET weight=? WHERE id=?", (w, ax["id"]))
            # 3. рост дендритов: похожие листья разных деревьев → мосты
            leaves = self.conn.execute(
                "SELECT n.id, n.tree_id, n.embedding FROM nodes n"
                " WHERE n.is_leaf=1 LIMIT 2000").fetchall()
            embs = {}
            for r in leaves:
                embs[r["id"]] = (r["tree_id"],
                                 np.frombuffer(r["embedding"], dtype=np.float32))
            ids = list(embs)
            grown = 0
            for i, a in enumerate(ids[:600]):
                ta, ea = embs[a]
                for b in ids[i + 1:]:
                    tb, eb = embs[b]
                    if tb == ta:
                        continue
                    if float(np.dot(ea, eb)) > SLEEP_BRIDGE_SIM:
                        if self._add_axon(ta, tb,
                                          "вырос во сне: смыслы соприкоснулись"):
                            grown += 1
                        break
                if grown >= 5:
                    break
            report["bridges_grown"] = grown
            self.conn.commit()
        return report

    def tree_info(self, tree_id):
        """Информация о дереве по id."""
        with self.lock:
            row = self.conn.execute(
                "SELECT id, name, trunk FROM trees WHERE id=?",
                (tree_id,)).fetchone()
        return dict(row) if row else None

    def stats(self):
        with self.lock:
            trees = self.conn.execute("SELECT COUNT(*) FROM trees").fetchone()[0]
            leaves = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE is_leaf=1").fetchone()[0]
            splits = self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE is_leaf=0").fetchone()[0]
            axons = self.conn.execute("SELECT COUNT(*) FROM axons").fetchone()[0]
        return {"trees": trees, "leaves": leaves, "splits": splits, "axons": axons}
