import json
import sqlite3
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "brain" / "memory.sqlite"
LOG = ROOT / "brain" / "life.log"

DIM, RESET = "\033[2m", "\033[0m"


def alive():
    r = subprocess.run(["pgrep", "-f", "src[.]main"], capture_output=True)
    pids = r.stdout.decode().split()
    return pids


def main():
    pids = alive()
    print(f"{DIM}=== Leta ==={RESET}")
    if pids:
        print(f"живёт (pid {' '.join(pids)})")
    else:
        print("не запущена — последняя жизнь в журнале ниже")

    try:
        conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    except Exception:
        print("память не найдена")
        return

    def q(sql):
        try:
            return conn.execute(sql).fetchall()
        except Exception:
            return []

    rows = q(
        "SELECT ts, content FROM episodes WHERE role='thought'"
        " ORDER BY id DESC LIMIT 4"
    )
    if rows:
        print(f"\n{DIM}--- её последние мысли ---{RESET}")
        for ts, c in reversed(rows):
            when = time.strftime("%H:%M", time.localtime(ts))
            print(f"[{when}] {c[:180]}")

    rows = q(
        "SELECT ts, content FROM episodes WHERE role='assistant'"
        " AND content NOT LIKE '[меня перебили]%'"
        " ORDER BY id DESC LIMIT 4"
    )
    if rows:
        print(f"\n{DIM}--- её последние слова ---{RESET}")
        for ts, c in reversed(rows):
            when = time.strftime("%H:%M", time.localtime(ts))
            print(f"[{when}] {c[:160]}")

    rows = q(
        "SELECT kind, content FROM impulses WHERE delivered=0 ORDER BY id LIMIT 5"
    )
    if rows:
        print(f"\n{DIM}--- несказанное (ждёт встречи) ---{RESET}")
        for kind, c in rows:
            print(f"({kind}) {c[:140]}")

    rows = q(
        "SELECT domain, COUNT(*) n FROM neurons GROUP BY domain"
        " ORDER BY n DESC LIMIT 6"
    )
    total = q("SELECT COUNT(*) FROM neurons")
    if rows:
        top = ", ".join(f"{d} ({n})" for d, n in rows)
        print(f"\n{DIM}--- знания ---{RESET}")
        print(f"всего {total[0][0] if total else 0}: {top}")

    rows = q("SELECT ts, mood FROM dreams ORDER BY id DESC LIMIT 3")
    if rows:
        print(f"{DIM}--- сны ---{RESET} " + ", ".join(
            f"{time.strftime('%d.%m', time.localtime(ts))} ({m})" for ts, m in rows
        ))

    try:
        ident = json.loads((ROOT / "brain" / "identity.json").read_text(encoding="utf-8"))
        wishes = ident.get("wishes", [])
        if wishes:
            print(f"\n{DIM}--- желания ---{RESET}")
            for w in wishes[-4:]:
                print(f"• {w[:150]}")
        skills = ident.get("skills", [])
        if skills:
            print(f"{DIM}--- её навыки ---{RESET}")
            for s in skills:
                cnt = conn.execute(
                    "SELECT value FROM meta WHERE key=?",
                    (f"skill_used:{s['name']}",),
                ).fetchone()
                used = cnt[0] if cnt else "0"
                print(f"  • {s['name']} — сработал {used} раз")
        else:
            print(f"{DIM}--- навыки ---{RESET} пока не создала ни одного")
        self_text = ident.get("self", "")
        if self_text:
            lines = [l for l in self_text.split("\n") if l.strip()]
            print(f"{DIM}--- о себе (последняя запись) ---{RESET}")
            print(lines[-1][:200])
    except Exception:
        pass

    books_dir = ROOT / "brain" / "books"
    if books_dir.exists():
        files = sorted(books_dir.glob("*.txt"))
        if files:
            print(f"{DIM}--- её полка книг ---{RESET}")
            for f in files[-4:]:
                print(f"  • {f.stem[:50]} ({f.stat().st_size // 1000} КБ)")

    try:
        import psutil

        bat = psutil.sensors_battery()
        if bat:
            charge = "заряжается" if bat.power_plugged else "на своей энергии"
            print(f"\n{DIM}--- тело ---{RESET} заряд {bat.percent:.0f}% ({charge}), "
                  f"CPU {psutil.cpu_percent(0.2):.0f}%")
    except Exception:
        pass

    if LOG.exists():
        tail = (
            subprocess.run(
                ["tail", "-6", str(LOG)], capture_output=True
            )
            .stdout.decode(errors="ignore")
            .replace("\x1b[0m", "")
        )
        import re

        tail = re.sub(r"\x1b\[[0-9;]*m", "", tail)
        lines = [l for l in tail.split("\n") if l.strip()]
        if lines:
            print(f"\n{DIM}--- журнал жизни (последнее) ---{RESET}")
            for l in lines:
                print(l[:170])
    conn.close()


if __name__ == "__main__":
    main()
