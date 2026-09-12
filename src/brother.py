import json
import time

DEFAULT_BROTHER = {
    "name": "",
    "facts": [],
    "promises": [],
    "updated": None,
}


class Brother:
    def __init__(self, root):
        self.path = root / "brain" / "brother.json"
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write(DEFAULT_BROTHER)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self._write(self.data)

    def _write(self, data):
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def profile_prompt(self):
        d = self.data
        if not (d.get("name") or d.get("facts") or d.get("promises")):
            return None
        lines = []
        if d.get("name"):
            lines.append(f"создательа зовут {d['name']}.")
        for f in d.get("facts", [])[-8:]:
            lines.append(f"- {f}")
        for p in d.get("promises", [])[-4:]:
            lines.append(f"- Обещание: {p}")
        return "\n".join(lines)

    def consider_update(self, brain, dialogue_text):
        her_name = brain.identity.data.get("name", "Leta").lower()
        prompt = (
            "Ниже реплики КИРИЛЛА (человека) из диалога. Запиши ТОЛЬКО то, что "
            "он СКАЗАЛ ПРЯМО о себе — его точные слова о своём имени, дате "
            "рождения, вкусах, работе, планах, обещаниях. ЗАПРЕЩЕНО "
            "додумывать, делать выводы, заполнять пробелы: если он не сказал "
            "прямо — не записывай. Одна выдумка хуже десяти пропусков. "
            "НЕ записывай ничего о Leta. "
            'Верни строго JSON: {"name": "", "facts": [], "promises": []}. '
            "Если реплик создательа нет или он ничего не сказал о себе — верни "
            "пустые списки.\n\n"
            + dialogue_text[:5000]
            + "\n\n/no_think"
        )
        raw = brain.generator.chat(
            [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=400
        )
        parsed = brain._parse_json_obj(raw)
        if not parsed:
            return None
        changed = False
        name = str(parsed.get("name", "")).strip()
        stop_names = (
            "создатель",
            "создательа",
            "создательу",
            "создательом",
            "создательь",
            "сестра",
            "сестры",
            "сестре",
            "учитель",
            her_name.lower(),
        )
        if (
            name
            and len(name) < 40
            and name[0].isupper()
            and name.lower() not in stop_names
            and name.lower() != self.data.get("name", "").lower()
        ):
            self.data["name"] = name
            changed = True
        for f in parsed.get("facts", [])[:5]:
            f = self._norm_fact(f)
            if (
                not f
                or len(f) > 160
                or her_name in f.lower()
                or self._similar(brain, f)
                or self._is_her_wish(brain, f)
            ):
                continue
            self.data.setdefault("facts", []).append(f)
            try:
                brain.cortex.upsert_neuron(f, "создатель", brain.embedder.embed(f))
            except Exception:
                pass
            changed = True
        for p in parsed.get("promises", [])[:3]:
            p = str(p).strip()
            if not p or len(p) > 160 or her_name in p.lower() or p in self.data.get("promises", []):
                continue
            self.data.setdefault("promises", []).append(p)
            changed = True
        facts = self.data.setdefault("facts", [])
        del facts[:-15]
        promises = self.data.setdefault("promises", [])
        del promises[:-6]
        if changed:
            self.data["updated"] = time.time()
            self.save()
        return parsed

    @staticmethod
    def _norm_fact(f):
        if isinstance(f, dict):
            for k in ("факт", "fact", "text", "интерес", "интересы"):
                v = f.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        return str(f).strip()

    @staticmethod
    def _is_her_wish(brain, fact):
        try:
            import numpy as np

            vec = brain.embedder.embed(fact)
            for w in brain.identity.data.get("wishes", []):
                if float(np.dot(brain.embedder.embed(w), vec)) > 0.7:
                    return True
        except Exception:
            pass
        return False

    def _similar(self, brain, fact):
        try:
            import numpy as np

            vec = brain.embedder.embed(fact)
            for existing in self.data.get("facts", []):
                if (
                    float(np.dot(brain.embedder.embed(existing), vec)) > 0.88
                ):
                    return True
        except Exception:
            pass
        return False
