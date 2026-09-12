from pathlib import Path

import numpy as np


class Thalamus:
    def __init__(self, conn, cfg, root):
        self.conn = conn
        self.novelty_threshold = cfg["retrieval"]["novelty_threshold"]
        self.rf_path = root / "brain" / "thalamus_rf.joblib"
        self.rf = None
        self._load_rf()

    def _load_rf(self):
        if self.rf_path.exists():
            import joblib

            self.rf = joblib.load(self.rf_path)

    def assess(self, vec, cortex, hippocampus):
        sim_neurons = cortex.max_similarity(vec)
        episodes = hippocampus.search(vec, k=1)
        sim_episodes = episodes[0][0] if episodes else 0.0
        familiarity = max(sim_neurons, sim_episodes)
        return {
            "familiarity": round(familiarity, 3),
            "novel": familiarity < self.novelty_threshold,
            "domain": self.predict_domain(vec, cortex),
        }

    def predict_domain(self, vec, cortex):
        if self.rf is not None:
            try:
                return str(self.rf.predict([vec])[0])
            except Exception:
                pass
        return self._nearest_domain(vec, cortex)

    def _nearest_domain(self, vec, cortex):
        best_sim, best_domain = 0.5, "general"
        for _, _, domain, blob in cortex.all_neurons():
            sim = float(np.dot(np.frombuffer(blob, dtype=np.float32), vec))
            if sim > best_sim:
                best_sim, best_domain = sim, domain
        return best_domain

    def maybe_train_rf(self, cortex):
        rows = cortex.all_neurons()
        if len(rows) < 20:
            return False
        X = [np.frombuffer(blob, dtype=np.float32) for _, _, _, blob in rows]
        y = [domain for _, _, domain, _ in rows]
        if len(set(y)) < 2:
            return False
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=60, max_depth=12, n_jobs=-1)
        rf.fit(X, y)
        import joblib

        joblib.dump(rf, self.rf_path)
        self.rf = rf
        return True
