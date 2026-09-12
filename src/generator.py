import re
import threading

from llama_cpp import Llama


class Generator:
    def __init__(self, cfg):
        self._lock = threading.Lock()
        self.llm = Llama(
            model_path=str(cfg.llm_path),
            n_ctx=cfg["context_size"],
            n_gpu_layers=cfg["n_gpu_layers"],
            verbose=False,
        )
        arch = self.llm.metadata.get("general.architecture", "")
        self.is_llama = "llama" in arch
        self.stop_tokens = (
            ["<|eot_id|>", "<|start_header_id|>"]
            if self.is_llama
            else ["<|im_end|>", "<|im_start|>"]
        )

    def chat(self, messages, temperature=0.7, max_tokens=400):
        prompt = self._format(messages)
        with self._lock:
            out = self.llm(
                prompt,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stop=self.stop_tokens,
                echo=False,
            )
        text = self._clean(out["choices"][0]["text"])
        if out["choices"][0].get("finish_reason") == "length":
            text = self._trim_to_sentence(text)
        return text

    @staticmethod
    def _trim_to_sentence(text):
        for i in range(len(text) - 1, max(len(text) - 300, 0), -1):
            if text[i] in ".!?…":
                return text[: i + 1].strip()
        return text

    @staticmethod
    def _clean(text):
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
        text = text.replace("</think>", " ").replace("<think>", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _format(self, messages):
        if self.is_llama:
            parts = ["<|begin_of_text|>"]
            for m in messages:
                parts.append(
                    f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
                    f"{m['content']}<|eot_id|>"
                )
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
