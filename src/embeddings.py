import threading

import numpy as np
from llama_cpp import Llama


class Embedder:
    def __init__(self, cfg):
        self._lock = threading.Lock()
        self.model = Llama(
            model_path=str(cfg.embedder_path),
            embedding=True,
            n_ctx=512,
            n_gpu_layers=cfg["n_gpu_layers"],
            verbose=False,
        )

    def embed(self, text: str) -> np.ndarray:
        text = text.replace("\n", " ")[:1500]
        with self._lock:
            vec = np.array(
                self.model.create_embedding([text])["data"][0]["embedding"],
                dtype=np.float32,
            )
        return vec / (np.linalg.norm(vec) + 1e-9)
