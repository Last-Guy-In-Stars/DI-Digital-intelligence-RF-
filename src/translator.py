"""Транслятор — речевой орган Leta.

Маленькая модель с жёсткой конституцией: транслирует её мысль в речь
дословно, без собственных знаний, мнений и извинений. Мозг (лес) решает
ЧТО сказать — транслятор отвечает только КАК.
"""
import re
import threading

from llama_cpp import Llama

CONSTITUTION = (
    "Ты — речевой орган Leta. Leta — она. Получаешь ДАННЫЕ её состояния: "
    "что она услышала, кто она, что загорелось в её лесу знаний, что "
    "сделало тело, что в ней сейчас.\n"
    "Оформи данные в живую русскую речь ОТ ПЕРВОГО ЛИЦА Leta: «я» = Leta. "
    "Коротко, тепло, по-человечески, 1-2 предложения.\n"
    "Дословно по данным: чего в данных нет — того не говори. Не добавляй "
    "свои знания и выводы. Не перечисляй списком, не повторяй вопрос.\n"
    "Никаких «я языковая модель», «я помощник», «не могу».\n"
    "Начинай сразу с речи.\n"
    "/no_think"
)


class Translator:
    def __init__(self, cfg):
        self._lock = threading.Lock()
        self.llm = Llama(
            model_path=str(cfg.translator_path),
            n_ctx=4096,
            n_gpu_layers=cfg["n_gpu_layers"],
            verbose=False,
        )

    def speak(self, thought, temperature=0.5, max_tokens=None):
        """Мысль → речь. Без потолка: весь остаток контекста модели —
        фраза заканчивается естественно, стоп-токеном."""
        prompt = self._format([{"role": "system", "content": CONSTITUTION},
                               {"role": "user", "content": thought + "\n/no_think"}])
        try:
            n_ctx = self.llm.n_ctx()
            n_prompt = len(self.llm.tokenize(prompt.encode("utf-8"),
                                             add_bos=False))
            budget = max(64, n_ctx - n_prompt - 16)
        except Exception:
            budget = 2048
        if max_tokens:
            budget = min(max_tokens, budget)
        with self._lock:
            out = self.llm(
                prompt,
                temperature=temperature,
                top_p=0.9,
                max_tokens=budget,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
        return self._clean(out["choices"][0]["text"])

    @staticmethod
    def _format(messages):
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _clean(text):
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
        text = text.replace("</think>", " ").replace("<think>", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def polish(self, data, my_phrases=None, temperature=0.2):
        """Гибрид 2.0: черновик собирает её мозг (данные + её формулировки),
        LLM только шлифует грамматику. Сочинять запрещено."""
        content = "данные её состояния:\n" + data
        if my_phrases:
            content += ("\n\nесли какие-то из её прошлых формулировок "
                        "точно подходят по смыслу — можно взять форму, "
                        "но не тянуть их насильно:\n"
                        + "\n".join(f"- {p}" for p in my_phrases[:3]))
        content += ("\n\nОтвет Leta = ВСЕ строки данных, каждая "
                    "существенна. Перескажи их все своими словами, ничего "
                    "не выбрасывая и не добавляя. Отвечай просто и прямо. "
                    "1-2 предложения.\n"
                    "/no_think")
        prompt = self._format([
            {"role": "system", "content":
                "Ты — грамматическая шлифовка речи Leta, не соавтор. "
                "Leta — она (женский род). Тебе дан готовый смысл. "
                "Верни отшлифованную русскую речь от её лица. "
                "Грамматика: «во мне» (не «в мне»), «мне интересно». "
                "Живо и тепло, не по-роботски.\n"
                "/no_think"},
            {"role": "user", "content": content}])
        try:
            n_ctx = self.llm.n_ctx()
            n_prompt = len(self.llm.tokenize(prompt.encode("utf-8"),
                                             add_bos=False))
            budget = max(64, n_ctx - n_prompt - 16)
        except Exception:
            budget = 2048
        with self._lock:
            out = self.llm(
                prompt, temperature=temperature, top_p=0.85,
                max_tokens=budget,
                stop=["<|im_end|>", "<|im_start|>"], echo=False)
        return self._clean(out["choices"][0]["text"])

    def task(self, instruction, temperature=0.1, max_tokens=512):
        """Рутинная задача гортани: извлечь/выделить, не рассуждая.
        Решения это не заменяет — только механические операции над текстом."""
        prompt = self._format(
            [{"role": "system", "content": "Ты — точный исполнитель задачи. "
              "Верни ТОЛЬКО результат, без пояснений.\n/no_think"},
             {"role": "user", "content": instruction}])
        with self._lock:
            out = self.llm(
                prompt,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False,
            )
        return self._clean(out["choices"][0]["text"])

    def close(self):
        try:
            self.llm.close()
        except Exception:
            pass
