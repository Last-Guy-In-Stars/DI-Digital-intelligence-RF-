import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS neurons(
  id INTEGER PRIMARY KEY,
  concept TEXT,
  domain TEXT,
  embedding BLOB,
  activations INTEGER DEFAULT 1,
  last_activated REAL,
  created REAL
);
CREATE TABLE IF NOT EXISTS synapses(
  pre INTEGER,
  post INTEGER,
  strength REAL,
  co_activations INTEGER DEFAULT 1,
  updated REAL,
  PRIMARY KEY(pre, post)
);
"""


class Cortex:
    def __init__(self, conn, hebbian_cfg):
        self.conn = conn
        self.heb = hebbian_cfg
        self.conn.executescript(SCHEMA)

    def all_neurons(self):
        return self.conn.execute(
            "SELECT id, concept, domain, embedding FROM neurons"
        ).fetchall()

    def upsert_neuron(self, concept, domain, vec):
        for nid, blob in self.conn.execute(
            "SELECT id, embedding FROM neurons"
        ).fetchall():
            v = np.frombuffer(blob, dtype=np.float32)
            if float(np.dot(v, vec)) > self.heb["merge_similarity"]:
                self._activate(nid)
                self.conn.commit()
                return nid
        cur = self.conn.execute(
            "INSERT INTO neurons(concept, domain, embedding, last_activated, created)"
            " VALUES (?,?,?,?,?)",
            (concept, domain, vec.tobytes(), time.time(), time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def _activate(self, nid):
        self.conn.execute(
            "UPDATE neurons SET activations=activations+1, last_activated=? WHERE id=?",
            (time.time(), nid),
        )

    def reinforce(self, ids):
        now = time.time()
        factor = self.heb["reinforce_factor"]
        initial = self.heb["initial_strength"]
        ceiling = self.heb["max_strength"]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                self.conn.execute(
                    "INSERT INTO synapses(pre, post, strength, co_activations, updated)"
                    " VALUES(?,?,?,1,?)"
                    " ON CONFLICT(pre,post) DO UPDATE SET"
                    " strength=MIN(strength*?+0.05, ?),"
                    " co_activations=co_activations+1, updated=?",
                    (a, b, initial, now, factor, ceiling, now),
                )
        self.conn.commit()

    def recall(self, vec, top_k=5):
        rows = self.all_neurons()
        if not rows:
            return [], []
        sims = []
        for nid, concept, domain, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            sims.append((float(np.dot(v, vec)), nid, concept, domain))
        sims.sort(key=lambda x: -x[0])
        primary = sims[:top_k]
        ids = [s[1] for s in primary]
        for nid in ids:
            self._activate(nid)
        self.reinforce(ids)
        spread = self._spreading(ids, ids)
        return primary, spread

    def _spreading(self, ids, exclude):
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT pre, post, strength FROM synapses WHERE pre IN ({marks}) OR post IN ({marks})",
            ids + ids,
        ).fetchall()
        scores = {}
        for pre, post, strength in rows:
            other = post if pre in ids else pre
            scores[other] = max(scores.get(other, 0.0), strength)
        result = []
        for nid, strength in sorted(scores.items(), key=lambda x: -x[1]):
            if nid in exclude:
                continue
            row = self.conn.execute(
                "SELECT concept, domain FROM neurons WHERE id=?", (nid,)
            ).fetchone()
            if row:
                result.append((strength, nid, row[0], row[1]))
            if len(result) >= 4:
                break
        return result

    def random_knowledge(self, n=2):
        rows = self.conn.execute(
            "SELECT concept FROM neurons WHERE domain NOT IN ('general','брат')"
            " ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        return [r[0] for r in rows]

    def dream_seed(self):
        row = self.conn.execute(
            "SELECT id FROM neurons ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if not row:
            return []
        seed = row[0]
        ids = [seed]
        rows = self.conn.execute(
            "SELECT post AS other, strength FROM synapses WHERE pre=?"
            " UNION ALL SELECT pre, strength FROM synapses WHERE post=?"
            " ORDER BY strength DESC LIMIT 2",
            (seed, seed),
        ).fetchall()
        ids.extend(other for other, _ in rows)
        marks = ",".join("?" * len(ids))
        concepts = self.conn.execute(
            f"SELECT concept FROM neurons WHERE id IN ({marks})", ids
        ).fetchall()
        return [c for (c,) in concepts]

    def weak_domains(self, max_count=3):
        rows = self.conn.execute(
            "SELECT domain, COUNT(*) n FROM neurons GROUP BY domain"
            " HAVING domain NOT IN ('general', 'unknown') ORDER BY n ASC LIMIT ?",
            (max_count,),
        ).fetchall()
        return [d for d, _ in rows]

    def max_similarity(self, vec):
        best = 0.0
        for _, _, _, blob in self.all_neurons():
            best = max(best, float(np.dot(np.frombuffer(blob, dtype=np.float32), vec)))
        return best

    def decay(self):
        now = time.time()
        tau = self.heb["decay_tau_days"] * 86400.0
        rows = self.conn.execute(
            "SELECT pre, post, strength, updated FROM synapses"
        ).fetchall()
        for pre, post, strength, updated in rows:
            elapsed = max(now - updated, 0.0)
            new_strength = strength * float(np.exp(-elapsed / tau))
            if new_strength < self.heb["prune_below"]:
                self.conn.execute(
                    "DELETE FROM synapses WHERE pre=? AND post=?", (pre, post)
                )
            else:
                self.conn.execute(
                    "UPDATE synapses SET strength=? WHERE pre=? AND post=?",
                    (new_strength, pre, post),
                )
        self.conn.commit()
