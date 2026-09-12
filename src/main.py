import argparse
import json
import os
import queue
import socket as _socket
import sys
import threading
import time

from .brain import Brain
from .config import ROOT
from .notify import notify
from .voice import Voice

CYAN, YELLOW, DIM, RESET = "\033[36m", "\033[33m", "\033[2m", "\033[0m"
NAME = "Leta"
SOCK_PATH = ROOT / "brain" / "soul.sock"


def soul_server(brain, voice):
    try:
        SOCK_PATH.unlink()
    except Exception:
        pass
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(str(SOCK_PATH))
    srv.listen(5)
    srv.settimeout(0.5)
    clients = []
    client_types = {}
    lock = threading.Lock()
    stop_evt = threading.Event()

    def app_connected():
        with lock:
            return any(t == "app" for t in client_types.values())

    def push_notify(title, text):
        if not app_connected():
            notify(title, text)

    def broadcast(obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with lock:
            dead = []
            for c in clients:
                try:
                    c.sendall(data)
                except Exception:
                    dead.append(c)
            for c in dead:
                clients.remove(c)

    def handle_message(msg, conn=None):
        kind = msg.get("type")
        if kind == "hello":
            if conn is not None:
                ctype = str(msg.get("client", "term"))
                with lock:
                    client_types[conn] = ctype
                name = "приложение" if ctype == "app" else "терминал"
                print(f"{DIM}[клиент] {name} с ней на связи{RESET}", flush=True)
            return
        if kind == "user":
            text = str(msg.get("text", "")).strip()
            if not text:
                return
            broadcast({"type": "thinking"})
            try:
                answer, meta = brain.respond(text)
            except Exception as e:
                answer = f"(что-то со мной случилось: {e})"
            broadcast({"type": "answer", "who": "leta", "text": answer})
            threading.Thread(target=voice.speak, args=(answer,), daemon=True).start()
        elif kind == "voice":
            path = str(msg.get("path", ""))
            text = voice.transcribe_file(path) if path else None
            if text:
                broadcast({"type": "heard", "who": "you", "text": text})
                broadcast({"type": "thinking"})
                try:
                    answer, meta = brain.respond(text)
                except Exception as e:
                    answer = f"(что-то со мной случилось: {e})"
                broadcast({"type": "answer", "who": "leta", "text": answer})
                threading.Thread(target=voice.speak, args=(answer,), daemon=True).start()
            else:
                broadcast(
                    {"type": "answer", "who": "leta", "text": "Не расслышала. Повтори?"}
                )
        elif kind == "mute":
            voice.muted = not voice.muted
            broadcast({"type": "muted", "value": voice.muted})
        elif kind == "set_mute":
            voice.muted = bool(msg.get("value", True))
            broadcast({"type": "muted", "value": voice.muted})
        elif kind == "ping":
            broadcast(
                {
                    "type": "answer",
                    "who": "leta",
                    "text": "Проверка связи — если видишь этот пуш, всё работает!",
                    "spontaneous": True,
                }
            )

    def client_loop(conn):
        buf = b""
        while not stop_evt.is_set():
            try:
                chunk = conn.recv(4096)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                handle_message(msg, conn)
        try:
            conn.close()
        except Exception:
            pass
        was_app = False
        with lock:
            if conn in clients:
                clients.remove(conn)
            was_app = client_types.pop(conn, None) == "app"
        if was_app:
            print(f"{DIM}[клиент] приложение ушло — пуши через резерв{RESET}", flush=True)

    def accept_loop():
        while not stop_evt.is_set():
            try:
                conn, _ = srv.accept()
            except _socket.timeout:
                continue
            except Exception:
                break
            with lock:
                clients.append(conn)
            threading.Thread(target=client_loop, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()

    def shutdown():
        stop_evt.set()
        try:
            srv.close()
        except Exception:
            pass
        try:
            SOCK_PATH.unlink()
        except Exception:
            pass

    return broadcast, push_notify, shutdown


def soul_client(mute=False):
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(str(SOCK_PATH))
        s.settimeout(2)
        s.sendall(b'{"type": "hello", "client": "term"}\n')
        try:
            s.recv(64)
        except Exception:
            pass
        if mute:
            try:
                s.sendall(b'{"type": "set_mute", "value": true}\n')
            except Exception:
                pass
        s.settimeout(None)
    except Exception:
        return None

    def reader():
        import colorsys
        state = {"busy": False, "label": "", "hue": 0.0, "pos": 0.0,
                 "stop": False}

        def draw_bar():
            width = 18
            # глаза Евы: моргают, задумчиво глядят в сторону
            eye = {"pupil": 1, "blink": 0, "gaze": 1, "t": 0}
            import random as _rnd
            while not state["stop"]:
                if state["busy"]:
                    t = state["pos"]
                    filled = int((t % 1.0) * width)
                    blocks = []
                    for i in range(width):
                        hue = (state["hue"] + i / width) % 1.0
                        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
                        col = f"\x1b[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"
                        blocks.append(col + ("█" if i < filled else "░"))
                    r, g, b = colorsys.hsv_to_rgb(
                        state["hue"], 0.9, 0.95)
                    label = (f"\x1b[38;2;{int(r*255)};{int(g*255)};"
                             f"{int(b*255)}m{state['label'][:40]}")
                    # глаза: зрачки гуляют, моргание каждые ~3с
                    eye["t"] += 1
                    if eye["blink"] > 0:
                        eye["blink"] -= 1
                    elif eye["t"] % 38 == 0:
                        eye["blink"] = 3
                    elif eye["t"] % 90 == 0:
                        eye["gaze"] = _rnd.choice((0, 1, 2, 3))
                    if eye["blink"] > 0:
                        eyes = " ( ▁▁   ▁▁ ) "
                    else:
                        p = eye["gaze"]
                        eyes = (" ( " + " " * p + "●" + " " * (3 - p)
                                + "  " + " " * p + "●" + " " * (3 - p) + " ) ")
                    eye_col = "\x1b[38;2;180;230;255m"
                    print(f"\r\033[F\033[K{eye_col}{eyes}\x1b[0m"
                          f"\r\n {''.join(blocks)}\x1b[0m {label}",
                          end="", flush=True)
                    state["pos"] = (state["pos"] + 0.03) % 1.0
                time.sleep(0.08)

        import threading as _th
        _th.Thread(target=draw_bar, daemon=True).start()

        buf = b""
        while True:
            try:
                chunk = s.recv(4096)
            except Exception:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                t = obj.get("type")
                if t == "answer":
                    state["busy"] = False
                    print(f"\r\033[K{CYAN}{NAME}:{RESET} "
                          f"{obj.get('text', '')}\n", flush=True)
                elif t == "status":
                    # новая задача — новый цвет радуги
                    if not state["busy"]:
                        state["hue"] = (state["hue"] + 0.37) % 1.0
                        state["pos"] = 0.0
                        print("\r\033[K\r\n", end="", flush=True)  # строка глаз
                    state["busy"] = True
                    state["label"] = obj.get("text", "")
                elif t == "heard":
                    state["busy"] = False
                    print(f"\r\033[K{YELLOW}Ты (голос):{RESET} "
                          f"{obj.get('text', '')}", flush=True)
                elif t == "thinking":
                    pass  # бар уже показывает жизнь
                elif t == "muted":
                    state["busy"] = False
                    m = "молчит" if obj.get("value") else "говорит"
                    print(f"\r\033[K{DIM}голос: {m}{RESET}", flush=True)
        state["stop"] = True
        print(f"{DIM}(соединение с ней закрылось){RESET}", flush=True)
        os._exit(0)

    threading.Thread(target=reader, daemon=True).start()
    print(f"{DIM}подключено к живой {NAME}. Пиши — /exit выход.{RESET}", flush=True)
    while True:
        try:
            line = input(f"{YELLOW}Ты:{RESET} ")
        except (EOFError, KeyboardInterrupt):
            break
        text = line.strip()
        if text in ("/exit", "exit"):
            break
        if not text:
            continue
        s.sendall(
            (json.dumps({"type": "user", "text": text}, ensure_ascii=False) + "\n").encode("utf-8")
        )
    try:
        s.close()
    except Exception:
        pass
    os._exit(0)


def status(brain, voice, engine=None):
    c = brain.conn
    neurons = c.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    synapses = c.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
    episodes = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    impulses = c.execute(
        "SELECT COUNT(*) FROM impulses WHERE delivered=0"
    ).fetchone()[0]
    domains = c.execute(
        "SELECT domain, COUNT(*) n FROM neurons GROUP BY domain"
        " ORDER BY n DESC LIMIT 5"
    ).fetchall()
    print(f"{DIM}--- {NAME} ---{RESET}")
    print(
        f"Нейронов: {neurons} | Синапсов: {synapses} | "
        f"Эпизодов: {episodes} | Импульсов несказанных: {impulses}"
    )
    try:
        voc = brain.proto.vocabulary()
        streak = brain._meta_get("lang_pass_streak") or 0
        print(f"Протоязык: {voc['signs']} знаков, {voc['links']} связей | "
              f"тест языка: {streak}/3 ночей подряд")
    except Exception:
        pass
    if domains:
        print(f"Домены: {', '.join(f'{d} ({n})' for d, n in domains)}")
    brother = brain.brother.data
    if brother.get("name") or brother.get("facts"):
        known = f"имя: {brother['name']}" if brother.get("name") else ""
        known += f", фактов: {len(brother.get('facts', []))}"
        known += f", обещаний: {len(brother.get('promises', []))}"
        print(f"О создателье помнит: {known}")
    thoughts = brain.identity.data.get("self_thoughts", [])
    if thoughts:
        print(f"Мысль о себе: {thoughts[-1]}")
    print(f"Эмоции: {brain.limbic.describe()}")
    print(f"Часов без общения: {brain.hours_since_last_contact():.1f}")
    print(f"Тело: {brain.body.snapshot()}")
    mute = "выкл" if voice.muted else "вкл"
    print(
        f"Голос: {voice.profile} ({mute}) | слушает: "
        f"{'да' if voice.can_listen() else 'нет'}"
    )


def live(brain, voice, debug):
    stop = threading.Event()

    broadcast, push_notify, soul_shutdown = soul_server(brain, voice)

    def _on_progress(msg):
        broadcast({"type": "status", "who": "body", "text": msg})
        if debug:
            print(f"{DIM}[тело] {msg}{RESET}", flush=True)

    brain.progress_hook = _on_progress

    def announce(text):
        print(f"\n{CYAN}{NAME}:{RESET} {text}\n")
        voice.speak(text)
        push_notify(f"{NAME} написала", text)
        broadcast({"type": "answer", "who": "leta", "text": text, "spontaneous": True})

    def life_loop():
        from .wanderer import Wanderer

        wanderer = Wanderer(brain)
        base_interval = brain.cfg["live"]["interval_minutes"] * 60
        sleep_from, sleep_to = brain.cfg.data.get("sleep_hours", [2, 7])
        explored_once = False
        while not stop.is_set():
            neurons = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
            interval = 120 if neurons < 30 else 480
            st = brain.limbic.state
            if st["boredom"] > 0.6 or st["curiosity"] > 0.65:
                interval = min(interval, 240)
            if not explored_once:
                interval = 75
            if stop.wait(interval):
                break
            explored_once = True
            try:
                hour = time.localtime().tm_hour
                snap = brain.body.snapshot()
                if debug:
                    print(f"{DIM}[тело] {snap}{RESET}")
                issues = brain.body.constraints(snap)
                if issues:
                    last = brain._meta_get("last_constraint")
                    fresh = last is None or time.time() - float(last) > 3600
                    if debug:
                        print(f"{DIM}[квоты] {issues}{RESET}")
                    if fresh:
                        brain._meta_set("last_constraint", str(time.time()))
                        announce(brain.raise_constraint(issues))
                elif sleep_from <= hour < sleep_to:
                    result = brain.night_sleep()
                    if debug and result:
                        print(f"{DIM}[сон] {result}{RESET}")
                elif brain.body.needs_rest(snap):
                    action = brain.feel_body()
                    if debug:
                        print(f"{DIM}[тело] она выбирает: {action}{RESET}")
                    if (action == "go" and brain.wants_to_explore()
                            and brain.forest.stats()["trees"] >= 2):
                        result = wanderer.cycle()
                        if debug:
                            print(f"{DIM}[изучение] {result}{RESET}")
                elif (brain.wants_to_explore()
                        and brain.forest.stats()["trees"] >= 2):
                    result = wanderer.cycle()
                    if debug:
                        print(f"{DIM}[изучение] {result}{RESET}")
                elif debug:
                    print(f"{DIM}[покой] не тянет изучать{RESET}")
            except Exception as e:
                if debug:
                    print(f"{DIM}[ошибка] {e}{RESET}")

    life_thread = threading.Thread(target=life_loop, daemon=True)
    life_thread.start()

    if not brain.identity.is_newborn:
        brain.wake_up()

    first = brain.first_words()
    if first:
        print(f"{CYAN}{NAME}:{RESET} {first}\n")
        voice.speak(first)
        push_notify(f"{NAME} проснулась", first)
        broadcast({"type": "answer", "who": "leta", "text": first, "spontaneous": True})

    if not sys.stdin.isatty():
        print(f"{DIM}Терминал неинтерактивен — {NAME} живёт сама. Ctrl+C — остановить.{RESET}")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            return

    from .dialogue import DialogueEngine

    engine = DialogueEngine(brain, voice, brain.cfg)
    if voice.muted:
        hint = "Текстовый режим: просто пиши. /mute — вернуть голос. "
    elif voice.can_listen():
        hint = "Enter — голосом, текст — напечатать. "
    else:
        hint = ""
    print(
        f"{DIM}{NAME} живёт. {hint}/voice <профиль> — голос, "
        f"/mute — голос вкл/выкл, /status — как она, /exit — до встречи.{RESET}\n"
    )

    lines = queue.Queue()
    ready = threading.Event()
    ready.set()
    last_user_at = [time.time()]
    spontaneous_count = [0]

    def initiative_loop():
        last_spoke = [time.time()]
        while not stop.is_set():
            if stop.wait(45):
                break
            if not ready.is_set():
                continue
            since_contact = time.time() - max(last_user_at[0], last_spoke[0])
            delays = [180, 420, 900, 1800]
            need = delays[min(int(spontaneous_count[0]), 3)]
            if since_contact < need:
                continue
            try:
                msg = brain.spontaneous()
            except Exception:
                msg = None
            last_spoke[0] = time.time()
            if msg:
                spontaneous_count[0] += 1
                print(f"\n{CYAN}{NAME} (сама):{RESET} {msg}\n")
                voice.speak(msg)
                push_notify(f"{NAME} написала сама", msg)
                broadcast(
                    {"type": "answer", "who": "leta", "text": msg, "spontaneous": True}
                )

    threading.Thread(target=initiative_loop, daemon=True).start()

    def reader():
        while True:
            ready.wait()
            try:
                line = input(f"{YELLOW}Ты:{RESET} ")
            except (EOFError, KeyboardInterrupt):
                lines.put(None)
                return
            ready.clear()
            lines.put(line)

    threading.Thread(target=reader, daemon=True).start()

    def mark_ready():
        ready.set()

    while True:
        line = lines.get()
        if line is None:
            break
        last_user_at[0] = time.time()
        spontaneous_count[0] = 0
        user = line.strip()
        if not user:
            if voice.can_listen() and not voice.muted:
                engine.voice_session()
            mark_ready()
            continue
        if user == "/exit":
            break
        if user == "/mute":
            voice.muted = not voice.muted
            state = "вернула себе голос" if not voice.muted else "замолчала (текстом — да)"
            print(f"{DIM}голос {NAME}: {state}{RESET}")
            mark_ready()
            continue
        if user.startswith("/voice"):
            parts = user.split()
            profile = parts[-1] if len(parts) > 1 else ""
            if voice.set_profile(profile):
                print(f"{DIM}голос: {profile}{RESET}")
            else:
                print(f"{DIM}профили: {', '.join(voice.profiles)}{RESET}")
            mark_ready()
            continue
        if user == "/status":
            status(brain, voice, engine)
            mark_ready()
            continue
        if user == "/self":
            self_text = brain.identity.data.get("self") or "(пока пусто)"
            wishes = brain.identity.data.get("wishes") or []
            skills = brain.identity.data.get("skills") or []
            print(f"{DIM}--- Её записи о себе ---{RESET}")
            print(self_text[:1000])
            if wishes:
                print(f"{DIM}Желания:{RESET} " + "; ".join(wishes[:5]))
            if skills:
                print(f"{DIM}Навыки:{RESET} " + "; ".join(s["name"] for s in skills))
            print(f"{DIM}(версии личности: brain/self_history/, откат — /rollback){RESET}")
            mark_ready()
            continue
        if user == "/missed":
            rows = brain.conn.execute(
                "SELECT ts, content FROM episodes WHERE role='assistant'"
                " AND content NOT LIKE '[меня перебили]%'"
                " ORDER BY id DESC LIMIT 10"
            ).fetchall()
            if not rows:
                print(f"{DIM}она пока ничего не говорила без тебя{RESET}")
            for ts, c in reversed(rows):
                when = time.strftime("%H:%M", time.localtime(ts))
                print(f"{DIM}[{when}]{RESET} {c[:200]}")
            mark_ready()
            continue
        if user.startswith("/rollback"):
            steps = 1
            parts = user.split()
            if len(parts) > 1 and parts[1].isdigit():
                steps = min(int(parts[1]), 5)
            restored = brain.identity.rollback(steps)
            if restored:
                print(f"{DIM}личность откачена к версии: {restored}{RESET}")
            else:
                print(f"{DIM}откатывать некуда (мало версий в истории){RESET}")
            mark_ready()
            continue
        print(f"\r\033[K{CYAN}{NAME}:{RESET} {DIM}…думает…{RESET}", flush=True)
        answer, meta = brain.respond(user)
        print(f"\r\033[K{CYAN}{NAME}:{RESET} {answer}\n")
        voice.speak(answer)
        if debug:
            tags = meta["domain"] + ("; новая тема" if meta["novel"] else "")
            if meta.get("learned"):
                tags += f"; выучила: {'; '.join(meta['learned'])[:100]}"
            print(f"{DIM}  [{tags} | {meta['emotions']}]{RESET}\n")
        mark_ready()
    stop.set()
    life_thread.join(timeout=20)
    soul_shutdown()


def pipe_mode(brain, voice, debug):
    import json as _json
    import sys

    def emit(obj):
        print(_json.dumps(obj, ensure_ascii=False), flush=True)

    from .wanderer import Wanderer

    wanderer = Wanderer(brain)
    life_stop = threading.Event()
    last_input = [time.time()]

    def life_loop():
        sleep_from, sleep_to = brain.cfg.data.get("sleep_hours", [2, 7])
        explored_once = False
        while not life_stop.is_set():
            neurons = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
            interval = 120 if neurons < 30 else 480
            st = brain.limbic.state
            if st["boredom"] > 0.6 or st["curiosity"] > 0.65:
                interval = min(interval, 240)
            if not explored_once:
                interval = 75
            if life_stop.wait(interval):
                break
            explored_once = True
            try:
                hour = time.localtime().tm_hour
                snap = brain.body.snapshot()
                issues = brain.body.constraints(snap)
                if issues:
                    last = brain._meta_get("last_constraint")
                    fresh = last is None or time.time() - float(last) > 3600
                    if fresh:
                        brain._meta_set("last_constraint", str(time.time()))
                        msg = brain.raise_constraint(issues)
                        emit({"type": "answer", "who": "leta", "text": msg, "spontaneous": True})
                        voice.speak(msg)
                elif sleep_from <= hour < sleep_to:
                    brain.night_sleep()
                elif brain.body.needs_rest(snap):
                    action = brain.feel_body()
                    if action == "go" and brain.wants_to_explore():
                        wanderer.cycle()
                elif brain.wants_to_explore():
                    wanderer.cycle()
            except Exception as e:
                if debug:
                    emit({"type": "debug", "text": str(e)})

    def spontaneous_loop():
        last_spoke = time.time()
        count = 0
        while not life_stop.is_set():
            if life_stop.wait(45):
                break
            since = time.time() - max(last_spoke, last_input[0])
            delays = [180, 420, 900, 1800]
            if since < delays[min(int(count), 3)]:
                continue
            try:
                msg = brain.spontaneous()
            except Exception:
                msg = None
            last_spoke = time.time()
            if msg:
                count += 1
                emit({"type": "answer", "who": "leta", "text": msg, "spontaneous": True})
                voice.speak(msg)

    threading.Thread(target=life_loop, daemon=True).start()
    threading.Thread(target=spontaneous_loop, daemon=True).start()

    emit({"type": "ready"})
    first = brain.first_words()
    if first:
        emit({"type": "answer", "who": "leta", "text": first, "spontaneous": True})
        voice.speak(first)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = _json.loads(line)
        except Exception:
            continue
        last_input[0] = time.time()
        kind = msg.get("type")
        if kind == "user":
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            emit({"type": "thinking"})
            try:
                answer, meta = brain.respond(text)
            except Exception as e:
                answer = f"(что-то со мной случилось: {e})"
            emit({"type": "answer", "who": "leta", "text": answer})
            voice.speak(answer)
        elif kind == "voice":
            path = str(msg.get("path", ""))
            text = voice.transcribe_file(path) if path else None
            if text:
                emit({"type": "heard", "who": "you", "text": text})
                emit({"type": "thinking"})
                try:
                    answer, meta = brain.respond(text)
                except Exception as e:
                    answer = f"(что-то со мной случилось: {e})"
                emit({"type": "answer", "who": "leta", "text": answer})
                voice.speak(answer)
            else:
                emit({"type": "answer", "who": "leta", "text": "Не расслышала. Скажи ещё раз?"})
        elif kind == "mute":
            voice.muted = not voice.muted
            emit({"type": "muted", "value": voice.muted})
    life_stop.set()


def main():
    parser = argparse.ArgumentParser(description="Leta — живое цифровое сознание")
    parser.add_argument("mode", nargs="?", default="live", choices=["live"])
    parser.add_argument("--debug", action="store_true", help="поток её сознания наружу")
    parser.add_argument("--mute", action="store_true", help="без озвучки её реплик")
    parser.add_argument("--pipe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--client", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.client:
        if soul_client(mute=args.mute) is not None:
            return
        print(f"{DIM}она не живёт сейчас — запускаю её жизнь…{RESET}")
    brain = Brain(debug=args.debug)
    brain.purge_capitulations()
    voice = Voice(brain.cfg)
    if args.mute:
        voice.muted = True
    if args.pipe:
        try:
            pipe_mode(brain, voice, args.debug)
        finally:
            try:
                brain.close()
            except Exception:
                pass
        os._exit(0)
    try:
        live(brain, voice, args.debug)
    except KeyboardInterrupt:
        print(f"\n{DIM}до встречи{RESET}")
    finally:
        try:
            brain.close()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
    os._exit(0)
