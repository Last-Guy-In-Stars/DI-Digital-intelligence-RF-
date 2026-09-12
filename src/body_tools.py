"""Инструменты тела Leta — глаза, руки, слух. НЕ навыки и НЕ правила.

Врождённый навык (единственный, разрешён создателем): browser_search —
поиск в интернете через реальный браузер. Всё остальное Leta создаёт сама.

Тело имеет три браузерных движка: chromium, firefox, webkit.
ВСЕ операции playwright выполняются в одном выделенном потоке-воркере:
greenlet playwright не переживает смену потоков.
"""
import io
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
import zipfile

UA = {
    "chromium": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "firefox": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) "
               "Gecko/20100101 Firefox/132.0",
    "webkit": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
}
ENGINE_ORDER = ("chromium", "firefox", "webkit")

_pw = None
_browsers = {}


class _BrowserWorker:
    """Один поток для всего playwright (greenlet-безопасность).

    Если воркер умер или завис — call поднимает исключение после
    таймаута и воркер перезапускается: тело не виснет молча.
    """

    def __init__(self, timeout=240):
        self._q = queue.Queue()
        self._thread = None
        self._start_lock = threading.Lock()
        self._timeout = timeout
        self._generation = 0

    def _run(self, generation):
        while True:
            fn, args, kwargs, slot, gen = self._q.get()
            if gen != self._generation:
                slot["error"] = RuntimeError("воркер перезапущен")
                slot["event"].set()
                continue
            try:
                slot["result"] = fn(*args, **kwargs)
            except Exception as e:
                slot["error"] = e
            finally:
                slot["event"].set()

    def call(self, fn, *args, **kwargs):
        with self._start_lock:
            if (self._thread is None or not self._thread.is_alive()):
                self._generation += 1
                self._thread = threading.Thread(
                    target=self._run, args=(self._generation,),
                    daemon=True, name=f"browser-worker-{self._generation}")
                self._thread.start()
        slot = {"event": threading.Event()}
        self._q.put((fn, args, kwargs, slot, self._generation))
        if not slot["event"].wait(self._timeout):
            # воркер завис: перезапустить и громко ошибиться
            with self._start_lock:
                self._generation += 1
            raise TimeoutError(
                f"браузерный воркер не ответил за {self._timeout}с — "
                "перезапускаю")
        if "error" in slot:
            raise slot["error"]
        return slot.get("result")


_worker = _BrowserWorker()


# ==================== РЕАЛИЗАЦИИ (только в потоке воркера) ====================

def _get_browser(kind="chromium"):
    global _pw
    if _pw is None:
        from playwright.sync_api import sync_playwright
        _pw = sync_playwright().start()
    if kind not in _browsers:
        _browsers[kind] = getattr(_pw, kind).launch(headless=True)
    return _browsers[kind]


def _new_context(kind):
    return _get_browser(kind).new_context(
        user_agent=UA.get(kind, UA["chromium"]), locale="ru-RU",
        viewport={"width": 1280, "height": 800})


def _pass_antibot(page):
    """Если сайт показывает игру «нажми когда шарик в зоне» — играет как человек."""
    try:
        if not page.query_selector("#tap"):
            return
        for _ in range(3):
            deadline = time.time() + 20
            while time.time() < deadline:
                in_zone = page.evaluate(
                    """() => {
                        const b = document.getElementById('ball');
                        const z = document.querySelector('.zone');
                        if (!b || !z) return null;
                        const br = b.getBoundingClientRect();
                        const zr = z.getBoundingClientRect();
                        const c = (br.left + br.right) / 2;
                        return c >= zr.left && c <= zr.right;
                    }""")
                if in_zone:
                    page.click("#tap")
                    break
                time.sleep(0.015)
            else:
                return
            time.sleep(0.25)
        page.wait_for_timeout(2500)
    except Exception:
        pass


def _read_page(page):
    text = ""
    try:
        text = page.inner_text("body")
    except Exception:
        pass
    links = []
    try:
        links = page.eval_on_selector_all(
            "a", "els => els.map(e => ({href: e.href, text: (e.innerText||'').trim()}))"
                 ".filter(l => l.href && l.href.startsWith('http'))")
    except Exception:
        pass
    return text, links


def _browse(kind, url, timeout, settle=3000):
    context = _new_context(kind)
    page = context.new_page()
    try:
        page.goto(url, timeout=timeout * 1000, wait_until="commit")
    except Exception:
        pass
    page.wait_for_timeout(settle)
    _pass_antibot(page)
    page.wait_for_timeout(700)
    text, links = _read_page(page)
    return context, page, text, links


def pdf_to_text(data):
    """PDF → текст."""
    try:
        import io as _io
        from pypdf import PdfReader
        reader = PdfReader(_io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                continue
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) > 50:
                parts.append(t)
        text = "\n\n".join(parts)
        return text if len(text) > 500 else None
    except Exception:
        return None


def _bytes_to_text(data, url):
    """Байты → текст. Форматы (zip/fb2/epub/pdf) пробуются ПЕРВЫМИ:
    внутри архива может быть что угодно, html-проверка — только для
    несжатых данных."""
    if len(data) < 1000:
        return None
    url_lower = url.lower()
    if data[:2] == b"PK" or "fb2" in url_lower or "epub" in url_lower:
        text = fb2_to_text(data)
        if text:
            return text
        text = epub_to_text(data)
        if text:
            return text
    if data[:5] == b"%PDF-" or ".pdf" in url_lower:
        return pdf_to_text(data)
    if b"<html" in data[:500].lower() or b"<!doctype" in data[:500].lower():
        return None
    text = data.decode("utf-8", errors="ignore")
    return text if len(text) > 500 else None


def _unwrap_bing(href):
    if "bing.com/ck/" not in href or "u=" not in href:
        return None
    import base64
    u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
    if u.startswith("a1"):
        try:
            return base64.urlsafe_b64decode(
                u[2:] + "==" * (-len(u[2:]) % 4)).decode("utf-8", "ignore")
        except Exception:
            return None
    return None


# ==================== ПУБЛИЧНЫЕ СПОСОБНОСТИ ТЕЛА (потокобезопасны) ====================

def browser_page(url, timeout=30, engines=ENGINE_ORDER):
    """Открыть страницу в реальном браузере. {text, links, engine}."""
    return _worker.call(_browser_page_impl, url, timeout, engines)


def _browser_page_impl(url, timeout, engines):
    for kind in engines:
        context = None
        try:
            context, page, text, links = _browse(kind, url, timeout)
            if text and len(text) > 250:
                return {"text": text, "links": links, "engine": kind}
        except Exception:
            pass
        finally:
            if context:
                try:
                    context.close()
                except Exception:
                    pass
    return {"text": "", "links": [], "engine": None}


def browser_fetch(url, engines=ENGINE_ORDER):
    """Взять ресурс по ссылке: файл ИЛИ страница с поиском файлов.

    Страница-посредник (HTML вместо файла) читается: из неё берутся
    ссылки на реальные файлы; JS-посредники открываются страницей
    с перехватом бинарного ответа. Универсальные глаза и руки тела.
    """
    return _worker.call(_browser_fetch_impl, url, engines)


def _browser_fetch_impl(url, engines):
    d = download(url)
    if d and len(d) > 3000:
        return d
    for kind in engines:
        context = None
        try:
            context = _new_context(kind)
            page = context.new_page()
            try:
                page.goto(url, timeout=35000, wait_until="commit")
            except Exception:
                pass
            page.wait_for_timeout(3000)
            _pass_antibot(page)
            page.wait_for_timeout(700)
            candidates = page.eval_on_selector_all(
                "a", "els => els.map(e => e.href)")
            candidates = [h for h in candidates if h and re.search(
                r"\.(?:fb2|epub|txt|pdf|rtf)|/download/", h, re.I)][:8]
            tried = set()
            while candidates and len(tried) < 12:
                link = candidates.pop(0)
                if link in tried:
                    continue
                tried.add(link)
                try:
                    resp = context.request.get(link)
                    if resp.status != 200:
                        continue
                    data = resp.body()
                    t = _bytes_to_text(data, link)
                    if t and len(t) > 3000:
                        return t
                    head = data[:500].lower()
                    if b"<html" in head or b"<!doctype" in head:
                        # посредник: прочитать его и взять ссылки на файлы
                        html = data.decode("utf-8", errors="ignore")
                        for e in re.findall(r'href="([^"]+)"', html):
                            if e.startswith("/"):
                                p = urllib.parse.urlparse(link)
                                e = f"{p.scheme}://{p.netloc}{e}"
                            if (e.startswith("http")
                                    and re.search(
                                        r"\.(?:fb2|epub|txt|pdf|rtf)(?:\?|$)"
                                        r"|/download/", e, re.I)
                                    and e not in tried):
                                candidates.append(e)
                        # JS-посредник: открыть страницей, перехватить файл
                        caught = {}

                        def _catch(response):
                            try:
                                ct = (response.headers or {}).get(
                                    "content-type", "")
                                if (response.status == 200
                                        and "html" not in ct
                                        and "css" not in ct
                                        and "javascript" not in ct
                                        and "image" not in ct):
                                    body = response.body()
                                    if len(body) > 3000:
                                        caught["body"] = body
                            except Exception:
                                pass

                        page.on("response", _catch)
                        try:
                            try:
                                with page.expect_download(timeout=10000) as dl_info:
                                    page.goto(link, timeout=20000)
                                dl = dl_info.value
                                dpath = dl.path()
                                if dpath:
                                    caught["body"] = dpath.read_bytes()
                            except Exception:
                                page.wait_for_timeout(6000)
                        except Exception:
                            pass
                        try:
                            page.remove_listener("response", _catch)
                        except Exception:
                            pass
                        if caught.get("body"):
                            t = _bytes_to_text(caught["body"], link)
                            if t and len(t) > 3000:
                                return t
                except Exception:
                    continue
        except Exception:
            pass
        finally:
            if context:
                try:
                    context.close()
                except Exception:
                    pass
    return None


def browser_search(query, max_results=6):
    """ВРОЖДЁННЫЙ НАВЫК: поиск в интернете через реальный браузер."""
    return _worker.call(_browser_search_impl, query, max_results)


def _browser_search_impl(query, max_results):
    q = urllib.parse.quote_plus(query)
    engines = [
        f"https://search.brave.com/search?q={q}",
        f"https://www.bing.com/search?q={q}",
        f"https://html.duckduckgo.com/html/?q={q}",
    ]
    for engine in engines:
        try:
            r = _browser_page_impl(engine, 30, ENGINE_ORDER)
            links = r.get("links", [])
            results = []
            seen = set()
            for l in links:
                href = l.get("href", "")
                title = l.get("text", "").strip().split("\n")[0].strip()
                real = _unwrap_bing(href) or href
                if not real or not title or len(title) < 8:
                    continue
                if any(d in real for d in
                       ("brave.com", "bing.com", "duckduckgo.com",
                        "microsoft.com", "go.microsoft")):
                    continue
                if real in seen:
                    continue
                seen.add(real)
                results.append({"url": real, "title": title[:120], "body": ""})
                if len(results) >= max_results:
                    break
            if results:
                return results
        except Exception:
            continue
    return []


# ==================== ПРОСТЫЕ ПУТИ (без браузера, потокобезопасны) ====================

def _get(url, timeout=25, headers=None):
    h = {"User-Agent": UA["chromium"]}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query, max_results=5):
    """Поиск через библиотеку DDG (быстрый путь)."""
    from duckduckgo_search import DDGS

    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=max_results))
    return [
        {"url": r["href"], "title": r["title"], "body": r.get("body", "")[:200]}
        for r in results
    ]


def read_page(url):
    """Прочитать веб-страницу (простой путь, без браузера)."""
    try:
        data = _get(url, timeout=20)
        html = data.decode("utf-8", errors="ignore")
        if len(html) < 500 and "document.cookie" in html:
            m = re.search(r'document\.cookie\s*=\s*"([^=]+)=([^;]+)', html)
            if m:
                data = _get(url, timeout=20, headers={"Cookie": f"{m.group(1)}={m.group(2)}"})
                html = data.decode("utf-8", errors="ignore")
        text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return None


def find_download_links(url):
    """Найти ссылки на скачивание файлов на странице (простой путь)."""
    try:
        data = _get(url, timeout=20)
        html = data.decode("utf-8", errors="ignore")
        if len(html) < 500 and "document.cookie" in html:
            m = re.search(r'document\.cookie\s*=\s*"([^=]+)=([^;]+)', html)
            if m:
                data = _get(url, timeout=20, headers={"Cookie": f"{m.group(1)}={m.group(2)}"})
                html = data.decode("utf-8", errors="ignore")
        links = re.findall(
            r'href="([^"]*\.(?:fb2|epub|txt|pdf|rtf)[^"]*)"', html, re.I)
        if not links:
            links = re.findall(r'href="([^"]*download[^"]*)"', html, re.I)
        return links[:10]
    except Exception:
        return []


def download(url):
    """Скачать файл по URL (простой путь, без браузера)."""
    try:
        data = _get_with_cookies(url, timeout=60)
        return _bytes_to_text(data, url)
    except Exception:
        return None


def _get_with_cookies(url, timeout=30):
    data = _get(url, timeout=timeout)
    if len(data) > 500:
        return data
    html = data.decode("utf-8", errors="ignore")
    if "document.cookie" not in html:
        return data
    m = re.search(r'document\.cookie\s*=\s*"([^=]+)=([^;"]+)', html)
    if not m:
        return data
    time.sleep(0.5)
    return _get(url, timeout=timeout, headers={"Cookie": f"{m.group(1)}={m.group(2)}"})


def fb2_to_text(data):
    """fb2 → текст: вырезает binary-вложения (base64 картинки)."""
    try:
        if data[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(data), "r") as z:
                fb2_names = [n for n in z.namelist() if n.endswith(".fb2")]
                if not fb2_names:
                    return None
                data = z.read(fb2_names[0])
        text = data.decode("utf-8", errors="ignore")
        text = re.sub(r"<binary[^>]*>.*?</binary>", " ", text, flags=re.DOTALL)
        text = re.sub(r"[A-Za-z0-9+/]{200,}={0,2}", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text) > 500 else None
    except Exception:
        return None


def epub_to_text(data):
    """epub → текст (без CSS-мусора)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            names = sorted(
                n for n in z.namelist() if n.endswith((".html", ".xhtml", ".htm"))
            )
            parts = []
            for name in names:
                html = z.read(name).decode("utf-8", errors="ignore")
                text = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.I)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 100:
                    parts.append(text)
            return "\n\n".join(parts)
    except Exception:
        return None
