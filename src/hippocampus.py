import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY,
  session TEXT,
  ts REAL,
  role TEXT,
  content TEXT,
  embedding BLOB,
  consolidated INTEGER DEFAULT 0
);
"""


class Hippocampus:
    def __init__(self, conn):
        self.conn = conn
        self.conn.executescript(SCHEMA)

    def remember(self, session, role, content, vec):
        self.conn.execute(
            "INSERT INTO episodes(session, ts, role, content, embedding) VALUES (?,?,?,?,?)",
            (session, time.time(), role, content, vec.tobytes()),
        )
        self.conn.commit()

    def recent(self, n=4):
        rows = self.conn.execute(
            "SELECT role, content FROM episodes WHERE role IN ('user','assistant')"
            " ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [(role, content[:280]) for role, content in reversed(rows)]

    def last_thoughts(self, n=4):
        rows = self.conn.execute(
            "SELECT content FROM episodes WHERE role='thought'"
            " ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def last_user_episode(self):
        row = self.conn.execute(
            "SELECT embedding FROM episodes WHERE role='user' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def last_assistant_content(self):
        row = self.conn.execute(
            "SELECT content FROM episodes WHERE role='assistant'"
            " AND content NOT LIKE '[меня перебили]%'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def last_assistant_embedding(self):
        row = self.conn.execute(
            "SELECT embedding FROM episodes WHERE role='assistant'"
            " AND content NOT LIKE '[меня перебили]%'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def search(self, vec, k=4):
        rows = self.conn.execute(
            "SELECT id, role, content, embedding FROM episodes ORDER BY id DESC LIMIT 2000"
        ).fetchall()
        scored = []
        for eid, role, content, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            scored.append((float(np.dot(v, vec)), eid, role, content))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def unconsolidated(self, limit=200):
        return self.conn.execute(
            "SELECT session, role, content FROM episodes WHERE consolidated=0 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()

    def mark_consolidated(self):
        self.conn.execute("UPDATE episodes SET consolidated=1 WHERE consolidated=0")
        self.conn.commit()
