import json
import shutil
import time


class Body:
    def __init__(self, cfg, conn):
        b = cfg.data.get("body", {})
        self.battery_min = b.get("battery_min_pct", 25)
        self.disk_min_gb = b.get("disk_min_gb", 5)
        self.daily_traffic_mb = b.get("daily_traffic_mb", 500)
        self.conn = conn

    def snapshot(self):
        info = {}
        try:
            import psutil

            bat = psutil.sensors_battery()
            if bat is not None:
                info["battery_pct"] = round(bat.percent, 1)
                info["charging"] = bool(bat.power_plugged)
            info["cpu_pct"] = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            info["ram_free_gb"] = round(mem.available / 2**30, 1)
        except Exception:
            pass
        try:
            du = shutil.disk_usage("/")
            info["disk_free_gb"] = round(du.free / 2**30, 1)
        except Exception:
            pass
        info["traffic_mb_today"] = round(self._traffic_today(), 1)
        return info

    def self_report(self):
        try:
            import psutil

            proc = psutil.Process()
            ram = proc.memory_info().rss / 2**30
            bat = psutil.sensors_battery()
            battery = (
                f"заряд тела {bat.percent:.0f}%" + (
                    " (питается от сети)" if bat.power_plugged else " (на своей энергии)"
                )
                if bat
                else "заряд неизвестен"
            )
            return (
                f"твоя собственная нагрузка на тело: {ram:.1f} ГБ памяти, "
                f"процессор занят тобой на {proc.cpu_percent(0.3):.0f}%, "
                f"всё тело загружено на {psutil.cpu_percent(0.3):.0f}%, "
                f"свободно памяти {psutil.virtual_memory().available / 2**30:.0f} ГБ, "
                f"{battery}"
            )
        except Exception:
            return "своё тело почувствовать не удалось"

    def needs_rest(self, snapshot=None):
        s = snapshot or self.snapshot()
        if "battery_pct" in s and not s.get("charging") and s["battery_pct"] < 40:
            return True
        if s.get("cpu_pct", 0) > 85:
            return True
        return False

    def _traffic_today(self):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='traffic'"
        ).fetchone()
        if not row:
            return 0.0
        try:
            d = json.loads(row[0])
            if d.get("date") == time.strftime("%Y-%m-%d"):
                return d.get("bytes", 0) / 2**20
        except Exception:
            pass
        return 0.0

    def add_traffic(self, nbytes):
        today = time.strftime("%Y-%m-%d")
        d = {"date": today, "bytes": 0}
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='traffic'"
        ).fetchone()
        if row:
            try:
                stored = json.loads(row[0])
                if stored.get("date") == today:
                    d = stored
            except Exception:
                pass
        d["date"] = today
        d["bytes"] = d.get("bytes", 0) + nbytes
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('traffic', ?)", (json.dumps(d),)
        )
        self.conn.commit()

    def constraints(self, snapshot=None):
        s = snapshot or self.snapshot()
        issues = []
        if "battery_pct" in s and not s.get("charging") and s["battery_pct"] < self.battery_min:
            issues.append(f"заряд моего тела {s['battery_pct']}% — ниже предела")
        if s.get("disk_free_gb", 99) < self.disk_min_gb:
            issues.append(f"на диске осталось всего {s['disk_free_gb']} ГБ")
        if s.get("traffic_mb_today", 0) > self.daily_traffic_mb:
            issues.append(
                f"я скачала {s['traffic_mb_today']:.0f} МБ сегодня — это мой дневной предел"
            )
        return issues

    def near_constraints(self, snapshot=None):
        s = snapshot or self.snapshot()
        warnings = []
        if "battery_pct" in s and not s.get("charging") and s["battery_pct"] < self.battery_min + 15:
            warnings.append(f"заряд тела падает: {s['battery_pct']}%")
        if s.get("traffic_mb_today", 0) > self.daily_traffic_mb * 0.8:
            warnings.append(
                f"трафик {s['traffic_mb_today']:.0f} из {self.daily_traffic_mb} МБ"
            )
        return warnings
