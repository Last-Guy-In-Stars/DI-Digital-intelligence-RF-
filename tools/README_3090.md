# LoRA v2 — исправленный датасет

## Что изменилось
- Identity: 100 примеров (было 35) — модель ТВЁРДО знает кто она
- Unfamiliar: 25 примеров (было 35) — только внешние незнакомые сущности
- Self-knowledge: 75 — знает о Кирилле, своих способностях
- Action: 75 — выполняет, не только говорит
- LoRA rank 32 (было 16) — лучше запоминает
- 5 эпох (было 3) — лучше выучивает
- Learning rate 1e-4 (было 2e-4) — стабильнее

## Запуск на 3090

### 1. Обновить файлы (с Mac):
```bash
cd ~/Desktop/My
tar czf leta_v2.tar.gz --exclude='.venv' --exclude='brain/models' \
  --exclude='build' --exclude='tools/whisper.cpp' --exclude='__pycache__' \
  --exclude='*.sock' --exclude='*.log' --exclude='*.tar.gz' Jarvis/
scp leta_v2.tar.gz admins@192.168.1.109:~/
```

### 2. На 3090:
```bash
ssh admins@192.168.1.109
rm -rf ~/Jarvis/tools/lora_output  # очистить старый результат
tar xzf leta_v2.tar.gz
cd Jarvis

# Если .venv уже есть — не пересоздавай, просто запусти:
.venv/bin/python tools/train_lora.py
```

### 3. После завершения (~15 мин):
```bash
ls -lh brain/models/qwen3-8b-leta-lora.gguf
```

### 4. На Mac:
```bash
scp admins@192.168.1.109:~/Jarvis/brain/models/qwen3-8b-leta-lora.gguf \
  ~/Desktop/My/Jarvis/brain/models/

cd ~/Desktop/My/Jarvis
sed -i '' 's|"llm_path": ".*"|"llm_path": "brain/models/qwen3-8b-leta-lora.gguf"|' config.json
rm -f brain/memory.sqlite brain/identity.json
./leta stop; ./leta start
```
