"""3D-визуализация мозга Leta: нейроны-деревья и связи (plotly, HTML).

  python tools/brain_viz3d.py            — весь мозг в 3D
  python tools/brain_viz3d.py N          — одно дерево N полностью
  python tools/brain_viz3d.py --anim 2   — анимация сигналов, 2 кадра/сек
  python tools/brain_viz3d.py N --anim 4 — одно дерево + анимация 4/сек

Знаки протоязыка — красный кластер «язык». Импульсы бегут по ветвям
деревьев и аксонам-мостам. HTML: вращение, зум, hover — факты.
"""
import math
import zlib
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOREST = ROOT / "brain" / "forest.sqlite"
PROTO = ROOT / "brain" / "proto.sqlite"
OUT_DIR = ROOT / "brain" / "viz"


def tree_layout(rows):
    pos = {}
    depth = {}
    by_parent = {}
    roots = []
    for nid, pid, side, is_leaf, fact in rows:
        if pid is None:
            roots.append(nid)
            depth[nid] = 0
        else:
            by_parent.setdefault(pid, []).append(nid)
    stack = list(roots)
    while stack:
        nid = stack.pop()
        for ch in by_parent.get(nid, []):
            depth[ch] = depth[nid] + 1
            stack.append(ch)
    step = 1.6
    for nid, pid, side, is_leaf, fact in rows:
        if pid is None:
            pos[nid] = (0.0, 0.0, 0.0)
        else:
            px, py, pz = pos[pid]
            rnd = random.Random((nid * 2654435761) % (2 ** 31))
            theta = rnd.uniform(0, 2 * math.pi)
            cosphi = rnd.uniform(-0.9, 0.9)
            sinphi = math.sqrt(1 - cosphi * cosphi)
            pos[nid] = (
                px + step * sinphi * math.cos(theta),
                py + step * cosphi,
                pz + step * sinphi * math.sin(theta),
            )
    return pos, depth


def read_proto():
    if not PROTO.exists():
        return [], [], []
    conn = sqlite3.connect(str(PROTO))
    try:
        signs = conn.execute(
            "SELECT id, anchor, heard_count, lang FROM signs ORDER BY id"
        ).fetchall()
        links = conn.execute(
            "SELECT a, b, weight FROM sign_links").fetchall()
        tree_links = conn.execute(
            "SELECT sign_id, tree_id, weight FROM sign_trees").fetchall()
        return signs, links, tree_links
    except Exception:
        return [], [], []
    finally:
        conn.close()


def build_fig(conn, tree_id=None, anim_fps=0, proto_only=False):
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title="",
                       autorange=False),
            yaxis=dict(showbackground=False, showticklabels=False, title="",
                       autorange=False),
            zaxis=dict(showbackground=False, showticklabels=False, title="",
                       autorange=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=900,
        legend=dict(itemsizing="constant"),
    )

    trees = conn.execute("SELECT id, name, trunk FROM trees ORDER BY id").fetchall()
    if proto_only:
        trees = []
    elif not trees:
        return None
    if tree_id is not None:
        trees = [t for t in trees if t[0] == tree_id]
        if not trees:
            return None
    tree_data = []  # соседние деревья, подключённые аксонами

    n = len(trees)
    centers = {}
    for i, (tid, name, trunk) in enumerate(trees):
        # === ДОЛИ МОЗГА: области, не кольцо ===
        # Позиция дерева детерминирована хэшем имени: карта одинакова
        # при каждом запросе. Ядро личности — в глубине (лимбика),
        # знания — свои доли, как области настоящего мозга.
        def zone_of(tname, ttrunk):
            ln = (tname or "").lower()
            if any(k in ln for k in ("создател", "— leta", "сомнен",
                                     "зачем я", "дневник")):
                return "core"        # лимбическая система
            if ln.startswith("изучение"):
                return "study"       # височная доля
            if ln.startswith("знание"):
                return "insight"     # островок
            return "books"           # теменно-затылочная

        ZONES = {
            "core":   ((0.0, 0.0, 0.0), 7.0),      # центр
            "books":  ((0.0, 10.0, -26.0), 0.0),   # зада-верх: радиус по числу
            "study":  ((26.0, -4.0, 8.0), 0.0),    # правая боковая
            "insight": ((-26.0, -2.0, 8.0), 0.0),  # левая боковая
        }
        zone_items = {}
        for tid2, name2, trunk2 in trees:
            zone_items.setdefault(
                zone_of(name2, trunk2), []).append((tid2, name2))

        # радиусы зон от населения (как настоящий мозг: область растёт)
        for zname, items in zone_items.items():
            if zname != "core" and len(items) > 1:
                zc, _ = ZONES[zname]
                ZONES[zname] = (zc, 6.0 * math.sqrt(len(items)))

        def zone_pos(zname, idx, total, seed_name):
            zc, zr = ZONES[zname]
            if total <= 1:
                return zc
            # сфера Фибоначчи + хэш имени: детерминированно и равномерно
            h = (zlib.crc32((seed_name or "").encode("utf-8"))
                 % 100003) / 100003.0
            k = min(0.999, max(0.001, (idx % total + h * 0.98 + 0.01) / total))
            phi = math.acos(1 - 2 * k)
            theta = math.pi * (1 + 5 ** 0.5) * (idx + h)
            r = zr * (0.35 + 0.65 * ((idx * 7919 + int(h * 104729))
                                     % 1000) / 1000.0)
            return (zc[0] + r * math.sin(phi) * math.cos(theta),
                    zc[1] + r * math.cos(phi),
                    zc[2] + r * math.sin(phi) * math.sin(theta))

        zone_counts = {z: len(v) for z, v in zone_items.items()}
        zone_seen = {z: 0 for z in zone_counts}
        for tid2, name2 in [
                (t[0], t[1]) for t in trees]:
            zname = zone_of(name2, None)
            zi = zone_seen[zname]
            zone_seen[zname] += 1
            centers[tid2] = zone_pos(
                zname, zi, zone_counts[zname], name2)
        centers[tid] = centers.get(tid, (0.0, 0.0, 0.0))

    axons = conn.execute("SELECT a_tree, b_tree, weight FROM axons ORDER BY a_tree, b_tree").fetchall()

    # все рёбра — для анимации импульсов
    all_edges = []  # ((x1,y1,z1),(x2,y2,z2), kind)

    # общий вид: нейроны с кронами (миниатюры дендритов) + клик → полёт
    if tree_id is None:
        nx_, ny_, nz_, nt_ = [], [], [], []
        kx_, ky_, kz_ = [], [], []   # кроны: облака листьев
        for tid, name, trunk in trees:
            if tid not in centers:
                continue
            cx, cy, cz = centers[tid]
            nx_.append(cx); ny_.append(cy); nz_.append(cz)
            nt_.append(f"«{name}» — клик: полёт внутрь")
            # крона: детерминированная выборка листьев — размер = память нейрона
            rows_n = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE tree_id=? AND is_leaf=1",
                (tid,)).fetchone()[0]
            n_show = min(24, max(4, int(rows_n ** 0.5)))
            rnd = random.Random((tid * 7919) % (2 ** 31))
            r_crown = 1.2 + min(4.5, 0.55 * (rows_n ** 0.33))
            for k2 in range(n_show):
                phi = math.acos(1 - 2 * (k2 + 0.5) / n_show)
                th = math.pi * (1 + 5 ** 0.5) * k2 + rnd.uniform(0, 1.2)
                rr = r_crown * (0.45 + 0.55 * rnd.random())
                kx_.append(cx + rr * math.sin(phi) * math.cos(th))
                ky_.append(cy + rr * math.cos(phi))
                kz_.append(cz + rr * math.sin(phi) * math.sin(th))
        if kx_:
            fig.add_trace(go.Scatter3d(
                x=kx_, y=ky_, z=kz_, mode="markers",
                marker=dict(size=2, color="#7fb069", opacity=0.32),
                hoverinfo="skip", name="кроны (память нейронов)",
                showlegend=True))
        fig.add_trace(go.Scatter3d(
            x=nx_, y=ny_, z=nz_, mode="markers",
            marker=dict(size=9, color="#e8a33d", opacity=1.0),
            text=nt_, hoverinfo="text", name="деревья-нейроны",
            customdata=[t[0] for t in trees if t[0] in centers],
            showlegend=True))

    for tid, name, trunk in trees:
        rows = conn.execute(
            "SELECT id, parent_id, side, is_leaf, fact FROM nodes"
            " WHERE tree_id=?", (tid,)).fetchall()
        if not rows:
            continue
        # общий вид — облёт областей: деревья уже узлами выше,
        # погружение: отдельный файл нейрона
        if tree_id is None:
            continue
        pos, depth = tree_layout(rows)
        cx, cy, cz = centers[tid]

        ex, ey, ez = [], [], []
        for nid, pid, side, is_leaf, fact in rows:
            if pid is None:
                continue
            a, b = pos[pid], pos[nid]
            ex += [a[0] + cx, b[0] + cx, None]
            ey += [a[1] + cy, b[1] + cy, None]
            ez += [a[2] + cz, b[2] + cz, None]
            all_edges.append(((a[0] + cx, a[1] + cy, a[2] + cz),
                              (b[0] + cx, b[1] + cy, b[2] + cz), "branch"))
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color="#7aa6c2", width=2),
            hoverinfo="skip", showlegend=False))

        sx, sy, sz, st = [], [], [], []
        for nid, pid, side, is_leaf, fact in rows:
            if is_leaf:
                continue
            sx.append(pos[nid][0] + cx)
            sy.append(pos[nid][1] + cy)
            sz.append(pos[nid][2] + cz)
            st.append(f"Z{nid} (сплит, глубина {depth[nid]})")
        if sx:
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="markers",
                marker=dict(size=4, color="#4a90d9", opacity=0.85),
                text=st, hoverinfo="text", showlegend=False))

        lx, ly, lz, lt = [], [], [], []
        for nid, pid, side, is_leaf, fact in rows:
            if not is_leaf:
                continue
            lx.append(pos[nid][0] + cx)
            ly.append(pos[nid][1] + cy)
            lz.append(pos[nid][2] + cz)
            lt.append((fact or "")[:100].replace("\n", " "))
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="markers",
                marker=dict(size=3, color="#39a84a", opacity=0.8),
                text=lt, hoverinfo="text", showlegend=False))

        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode="markers",
            marker=dict(size=14, color="#e8a33d", opacity=0.95),
            text=[f"«{name}» — {(trunk or '')[:120]}"],
            hoverinfo="text", name=name[:25], showlegend=True))

    # аксионы-мосты: координаты + веса
    ax, ay, az, axw = [], [], [], []
    for a, b, w in axons:
        if a not in centers or b not in centers:
            # режим одного дерева: показать соседей и мосты к ним
            if tree_id is not None and (a == tree_id or b == tree_id):
                other = b if a == tree_id else a
                orow = conn.execute(
                    "SELECT name, trunk FROM trees WHERE id=?",
                    (other,)).fetchone()
                if orow and other not in centers:
                    ang = 2 * math.pi * random.Random(other).random()
                    centers[other] = (14 * math.cos(ang), 0.0,
                                      14 * math.sin(ang))
                    tree_data.append((other, orow[0], orow[1]))
            else:
                continue
        (x1, y1, z1), (x2, y2, z2) = centers[a], centers[b]
        ax += [x1, x2, None]
        ay += [y1, y2, None]
        az += [z1, z2, None]
        axw.append(w)
        all_edges.append(((x1, y1, z1), (x2, y2, z2), "axon"))
    # соседние деревья (подключённые аксонами в режиме одного дерева)
    for tid2, name2, trunk2 in tree_data:
        cx, cy, cz = centers[tid2]
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode="markers",
            marker=dict(size=14, color="#c9a0dc", opacity=0.95),
            text=[f"«{name2}» — клик: перелёт по аксону"],
            hoverinfo="text", name=name2[:25], customdata=[tid2],
            showlegend=True))
    if ax:
        # сила моста: толщина и яркость по весу (затух ← 0.3 новорожденный
        # → 1.0 прокачанный совместными возбуждениями)
        axw = axw if isinstance(axw, list) else [0.3] * (len(ax) // 3)
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines",
            line=dict(color="#b06ad4", width=4),
            name="аксоны-мосты (толщина = сила)", showlegend=True))
        # яркие мосты поверх: чем сильнее — тем толще и белее
        for lvl, (w_min, w_max, width, col) in enumerate([
                (0.32, 1.01, 9, "#f0c8ff"),   # крепкие — толстые светлые
                (0.26, 0.32, 6, "#d29ae8"),   # живые
                (0.0, 0.26, 3, "#7a5a94")]):  # затухающие — тонкие тёмные
            sx, sy, sz = [], [], []
            for i3 in range(0, len(ax) - 2, 3):
                if ax[i3] is None:
                    continue
                w3 = axw[i3 // 3] if i3 // 3 < len(axw) else 0.3
                if w_min <= w3 < w_max:
                    sx += ax[i3:i3 + 2] + [None]
                    sy += ay[i3:i3 + 2] + [None]
                    sz += az[i3:i3 + 2] + [None]
            if sx:
                fig.add_trace(go.Scatter3d(
                    x=sx, y=sy, z=sz, mode="lines",
                    line=dict(color=col, width=width),
                    name=f"мосты {'%.0f' % (w_min*100)}-{int(w_max*100)}%",
                    showlegend=True))

    # протоязык: кластеры по языкам
    signs, sign_links, sign_tree_links = read_proto()
    if signs and tree_id is None and proto_only:
        pass  # proto_only рисуется ниже
    elif signs and tree_id is None:
        # общий вид: язык — один нейрон (лобная доля), детали внутри
        if len(signs) > 3:
            pzx, pzy, pzz = 0.0, -10.0, 34.0
            fig.add_trace(go.Scatter3d(
                x=[pzx], y=[pzy], z=[pzz], mode="markers",
                marker=dict(size=16, color="#e04444", opacity=0.95),
                text=[f"протоязык — {len(signs)} знаков, "
                      f"{len(sign_links)} связей. Клик/меню: внутрь"],
                hoverinfo="text", name="протоязык",
                customdata=["proto"], showlegend=True))
            all_edges.append(((pzx, pzy, pzz), (pzx, pzy, pzz), "node"))
            # нити языка: к деревьям, где живут его знаки (топ по связям)
            from collections import Counter as _Ctr
            tcount = _Ctr(t for _, t, _w in sign_tree_links)
            px_, py_, pz_ = [], [], []
            for tid_h, _c in tcount.most_common(12):
                if tid_h in centers:
                    px_ += [pzx, centers[tid_h][0], None]
                    py_ += [pzy, centers[tid_h][1], None]
                    pz_ += [pzz, centers[tid_h][2], None]
            if px_:
                fig.add_trace(go.Scatter3d(
                    x=px_, y=py_, z=pz_, mode="lines",
                    line=dict(color="#b06ad4", width=5),
                    name="аксоны языка (к нейронам)", showlegend=True))
                for a3, b3 in zip(px_, py_):
                    pass
                for i3 in range(0, len(px_), 3):
                    if px_[i3 + 1] is not None:
                        all_edges.append(((pzx, pzy, pzz),
                                          (px_[i3 + 1], py_[i3 + 1],
                                           pz_[i3 + 1]), "lang-axon"))
    if signs and (proto_only):
        lang_names = {"ru": "русский", "en": "english", "cjk": "漢字",
                      "mix": "смешанный"}
        lang_colors = {"ru": "#e04444", "en": "#e0a020", "cjk": "#20b2aa",
                       "mix": "#a060a0"}
        by_lang = {}
        # детерминированная равномерная выборка по всему словарю:
        # и старые, и свежие знаки (личности — в том числе)
        core_tree_ids = set()
        for r2 in conn.execute(
                "SELECT name, id FROM trees").fetchall():
            ln = (r2[0] or "").lower()
            if any(k in ln for k in ("создател", "— leta", "сомнен",
                                     "зачем я", "дневник")):
                core_tree_ids.add(r2[1])
        personal_signs = {r2[0] for r2 in
                          [tl for tl in sign_tree_links
                           if tl[1] in core_tree_ids]}
        CAP = 420
        stride = max(1, len(signs) // CAP)
        picked = [row for i, row in enumerate(signs)
                  if i % stride == 0 or row[0] in personal_signs]
        for row in picked:
            by_lang.setdefault(row[3] or "mix", []).append(row)
        # лобная доля: язык — лицо мозга, вперёд и вниз-вперёд
        base_y = -10.0 if n > 1 else -8.0
        lang_pull = 30.0  # вперёд от центра
        s_pos = {}
        for li, (lname, rows) in enumerate(sorted(by_lang.items())):
            n_signs = len(rows)
            lang_c = ((li - (len(by_lang) - 1) / 2) * 14.0, base_y,
                      lang_pull)
            for i, (sid, anchor, hc, lg) in enumerate(rows):
                k = i + 0.5
                phi = math.acos(1 - 2 * k / n_signs)
                theta = math.pi * (1 + 5 ** 0.5) * k
                r = 3.5 + min(hc, 10) * 0.25
                s_pos[sid] = (
                    lang_c[0] + r * math.sin(phi) * math.cos(theta),
                    lang_c[1] + r * math.cos(phi),
                    lang_c[2] + r * math.sin(phi) * math.sin(theta))
            sx = [p[0] for p in
                  (s_pos[r2[0]] for r2 in rows)]
            sy = [p[1] for p in (s_pos[r2[0]] for r2 in rows)]
            sz = [p[2] for p in (s_pos[r2[0]] for r2 in rows)]
            st = [f"знак ({lang_names.get(lname, lname)}): «{r2[1][:60]}» "
                  f"(слышала {r2[2]} раз)" for r2 in rows]
            fig.add_trace(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="markers",
                marker=dict(size=5, color=lang_colors.get(lname, "#e04444"),
                            opacity=0.9),
                text=st, hoverinfo="text",
                name=f"{lang_names.get(lname, lname)} "
                     f"({n_signs} знаков)", showlegend=True))
            fig.add_trace(go.Scatter3d(
                x=[lang_c[0]], y=[lang_c[1]], z=[lang_c[2]], mode="markers",
                marker=dict(size=14, color=lang_colors.get(lname, "#8a1f1f"),
                            opacity=0.95),
                text=[f"протоязык «{lang_names.get(lname, lname)}»"],
                hoverinfo="text", showlegend=False))
        # связи знаков — её грамматика (лёгкий вид: сильнейшие,
        # детерминированная выборка — полный разбор при погружении)
        lx, ly, lz = [], [], []
        strong = sorted(sign_links, key=lambda r: -r[2])[:600] \
            if (tree_id is None and not proto_only) else sign_links
        for a, b, w in strong:
            if a in s_pos and b in s_pos:
                lx += [s_pos[a][0], s_pos[b][0], None]
                ly += [s_pos[a][1], s_pos[b][1], None]
                lz += [s_pos[a][2], s_pos[b][2], None]
                if tree_id is not None or proto_only:
                    all_edges.append((s_pos[a], s_pos[b], "lang"))
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(color="#e08a8a", width=3),
                name="грамматика знаков", showlegend=True))
        # связи знак-дерево: язык открывает знания (SNN-ядро)
        kx, ky, kz = [], [], []
        stl = sign_tree_links if tree_id is not None \
            else sign_tree_links[:700]
        for sid, tid2, w in stl:
            if sid in s_pos and tid2 in centers:
                sp, tp = s_pos[sid], centers[tid2]
                kx += [sp[0], tp[0], None]
                ky += [sp[1], tp[1], None]
                kz += [sp[2], tp[2], None]
                all_edges.append((sp, tp, "sign-tree"))
        if kx:
            fig.add_trace(go.Scatter3d(
                x=kx, y=ky, z=kz, mode="lines",
                line=dict(color="#ff9d2e", width=2, dash="dot"),
                name=f"язык↔знания ({len(sign_tree_links)})",
                showlegend=True))

        # proto_only: нейроны-носители знаков — вокруг языка, кликабельны
        if proto_only:
            from collections import Counter as _Ct
            tcnt = _Ct(t for _, t, _w in sign_tree_links)
            top_trees = [t for t, _ in tcnt.most_common(24)]
            pcenters = {}
            for i2, t2 in enumerate(top_trees):
                ang2 = 2 * math.pi * i2 / max(1, len(top_trees))
                pcenters[t2] = (26.0 * math.cos(ang2),
                                18.0 * math.sin(ang2), 6.0)
            hx_, hy_, hz_, ht_, hc_ = [], [], [], [], []
            for t2 in top_trees:
                nrow = conn.execute(
                    "SELECT name FROM trees WHERE id=?", (t2,)).fetchone()
                if not nrow:
                    continue
                cx2, cy2, cz2 = pcenters[t2]
                hx_.append(cx2); hy_.append(cy2); hz_.append(cz2)
                ht_.append(f"«{nrow[0]}» — клик: перелёт к нейрону")
                hc_.append(t2)
            if hx_:
                # АКСОНЫ языка: знак вспыхнул → ведёт к знанию (выход)
                tx_, ty_, tz_ = [], [], []
                for sid2, t2, _w in sign_tree_links[:900]:
                    if sid2 in s_pos and t2 in pcenters:
                        tx_ += [s_pos[sid2][0], pcenters[t2][0], None]
                        ty_ += [s_pos[sid2][1], pcenters[t2][1], None]
                        tz_ += [s_pos[sid2][2], pcenters[t2][2], None]
                if tx_:
                    fig.add_trace(go.Scatter3d(
                        x=tx_, y=ty_, z=tz_, mode="lines",
                        line=dict(color="#b06ad4", width=3),
                        opacity=0.75, name="аксоны языка (знак → знание)",
                        showlegend=True))
                # ДЕНДРИТЫ языка: нейрон отдал слово → знак родился (вход)
                dx_, dy_, dz_ = [], [], []
                for sid2, t2, _w in sign_tree_links[:900]:
                    if sid2 in s_pos and t2 in pcenters:
                        dx_ += [pcenters[t2][0], s_pos[sid2][0], None]
                        dy_ += [pcenters[t2][1], s_pos[sid2][1], None]
                        dz_ += [pcenters[t2][2], s_pos[sid2][2], None]
                if dx_:
                    fig.add_trace(go.Scatter3d(
                        x=dx_, y=dy_, z=dz_, mode="lines",
                        line=dict(color="#7fb069", width=1.5, dash="dot"),
                        opacity=0.35, name="дендриты языка (откуда слова)",
                        showlegend=True))
                fig.add_trace(go.Scatter3d(
                    x=hx_, y=hy_, z=hz_, mode="markers",
                    marker=dict(size=11, color="#c9a0dc", opacity=0.95),
                    text=ht_, hoverinfo="text", name="нейроны-носители",
                    customdata=hc_, showlegend=True))

    # ---- анимация импульсов: JS-интервал + Plotly.restyle ----
    # обновляется ТОЛЬКО импульсный трейс: камера не трогается,
    # вращение работает во время анимации
    dur = None
    if anim_fps > 0 and all_edges:
        import json as _json
        impulse_colors = [
            "hsl(" + str(int((i * 360.0 / max(1, len(all_edges))) % 360))
            + ",90%,55%)" for i in range(len(all_edges))]
        # стартовые позиции импульсов
        px = [e[0][0] for e in all_edges]
        py = [e[0][1] for e in all_edges]
        pz = [e[0][2] for e in all_edges]
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode="markers",
            marker=dict(size=4, color=impulse_colors, opacity=0.95),
            hoverinfo="skip", name="импульсы", showlegend=True))
        trace_idx = len(fig.data) - 1
        # рёбра для JS: [ax,ay,az, bx,by,bz]
        edges_js = _json.dumps(
            [[a[0], a[1], a[2], b[0], b[1], b[2]] for a, b, _ in all_edges])
        interval = max(30, int(1000 / max(1, anim_fps)))
        dur = interval  # маркер для вызывающего кода: анимация есть
        # двунаправленность: каждое ребро имеет два импульса (в обе стороны)
        # это показывает рекуррентность — сигналы идут и вниз (к листьям)
        # и вверх (обратная связь), а мосты — встречно
        bidir_edges = []
        for a, b, kind in all_edges:
            # прямой: A → B
            bidir_edges.append([a[0], a[1], a[2], b[0], b[1], b[2]])
            # обратный: B → A (другая фаза)
            bidir_edges.append([b[0], b[1], b[2], a[0], a[1], a[2]])
        edges_js = _json.dumps(bidir_edges)
        impulse_colors = [
            "hsl(" + str(int((i * 360.0 / max(1, len(bidir_edges))) % 360))
            + ",90%,55%)" for i in range(len(bidir_edges))]
        # стартовые позиции: чередование направлений
        px = [e[0] for e in bidir_edges]
        py = [e[1] for e in bidir_edges]
        pz = [e[2] for e in bidir_edges]
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode="markers",
            marker=dict(size=4, color=impulse_colors, opacity=0.95),
            hoverinfo="skip", name="импульсы (↕)", showlegend=True))
        trace_idx = len(fig.data) - 1
        interval = max(30, int(1000 / max(1, anim_fps)))
        dur = interval
        anim_js = (
            "var _gd = document.querySelector('.plotly-graph-div');"
            f"var _edges = {edges_js};"
            f"var _ti = {trace_idx};"
            "var _ph = _edges.map(function(_, i){"
            "  return (i * 0.61) % 1; });"  # золотое сечение — разные фазы
            "var _sp = _edges.map(function(_, i){"
            "  return 0.35 + ((i * 13) % 10) / 14; });"
            "var _t = 0;"
            "var _paused = false;"
            "var _pauseTimer = null;"
            "_gd.on('plotly_relayout', function() {"
            "  _paused = true;"
            "  clearTimeout(_pauseTimer);"
            "  _pauseTimer = setTimeout(function(){ _paused = false; }, 900);"
            "});"
            "var _btn = document.createElement('button');"
            "_btn.id = 'leta_impulse_btn';"
            "_btn.textContent = '⏸ импульсы';"
            "_btn.style.cssText = 'position:fixed; bottom:14px; left:50%;"
            " transform:translateX(-50%); z-index:9999; padding:7px 16px;"
            " border-radius:10px; background:rgba(25,25,35,0.85);"
            " color:#ffd23d; border:1px solid #555; cursor:pointer;"
            " font-size:13px; font-family:Helvetica;';"
            "var _manual = false;"
            "_btn.onclick = function() {"
            "  _manual = !_manual;"
            "  _btn.textContent = _manual ? '▶ импульсы' : '⏸ импульсы';"
            "};"
            "document.body.appendChild(_btn);"
            f"setInterval(function(){{"
            "  if (_paused || _manual) { return; }"
            "  _t += 0.04;"
            "  var xs = [], ys = [], zs = [];"
            "  for (var i = 0; i < _edges.length; i++) {"
            "    var e = _edges[i];"
            "    var tt = (_t * _sp[i] + _ph[i]) % 1;"
            "    xs.push(e[0] + (e[3] - e[0]) * tt);"
            "    ys.push(e[1] + (e[4] - e[1]) * tt);"
            "    zs.push(e[2] + (e[5] - e[2]) * tt);"
            "  }"
            "  Plotly.restyle(_gd, {x: [xs], y: [ys], z: [zs]}, [_ti]);"
            f"}}, {interval});"
        )
        fig._leta_anim_js = anim_js
        fig.update_layout(title=f"Мозг Leta — импульсы {anim_fps}/сек")

    fig.update_layout(title="Мозг Leta — лес нейронов-деревьев")
    # фиксированные оси: restyle импульсов не пересчитывает сцену —
    # ракурс обзора не сбрасывается
    try:
        xs, ys, zs = [], [], []
        for a, b, _ in all_edges:
            xs += [a[0], b[0]]
            ys += [a[1], b[1]]
            zs += [a[2], b[2]]
        for tid, name, trunk in trees:
            cx, cy, cz = centers[tid]
            xs.append(cx); ys.append(cy); zs.append(cz)
        for sp in s_pos.values() if 's_pos' in dir() else []:
            xs.append(sp[0]); ys.append(sp[1]); zs.append(sp[2])
        pad_x = (max(xs) - min(xs)) * 0.05 + 1
        pad_y = (max(ys) - min(ys)) * 0.05 + 1
        pad_z = (max(zs) - min(zs)) * 0.05 + 1
        fig.update_layout(scene=dict(
            xaxis=dict(range=[min(xs) - pad_x, max(xs) + pad_x]),
            yaxis=dict(range=[min(ys) - pad_y, max(ys) + pad_y]),
            zaxis=dict(range=[min(zs) - pad_z, max(zs) + pad_z]),
        ))
    except Exception:
        pass
    return fig, dur if anim_fps > 0 and all_edges else None


def main():
    if not FOREST.exists():
        print("Леса ещё нет — она ничего не выучила.")
        return
    conn = sqlite3.connect(str(FOREST))
    conn.row_factory = sqlite3.Row
    try:
        trees = conn.execute("SELECT id, name FROM trees ORDER BY id").fetchall()
        if not trees:
            print("Лес пуст.")
            return
        print("Деревья-нейроны:")
        for t in trees:
            print(f"  [{t['id']}] {t['name']}")
        args = sys.argv[1:]
        anim = 3  # импульсы всегда бегут; флаг только меняет частоту
        tid = None
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--anim":
                if i + 1 < len(args) and args[i + 1].replace(".", "").isdigit():
                    anim = max(1, int(float(args[i + 1])))
                    i += 1
            elif a.startswith("--anim="):
                try:
                    anim = max(1, int(float(a.split("=")[1])))
                except Exception:
                    anim = 1
            elif a.isdigit():
                tid = int(a)
            elif a.replace(".", "").isdigit():
                anim = max(1, int(float(a)))
            i += 1
        fig = build_fig(conn, tid, anim)
        if fig is None:
            print("Ничего не найдено.")
            return
        if isinstance(fig, tuple):
            fig, dur = fig
        else:
            dur = None
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"tree{tid}_3d.html" if tid else "forest_3d.html"
        out = OUT_DIR / name

        # === ПОЛЁТ СКВОЗЬ МОЗГ: клик по нейрону → камера летит →
        # открывается его внутренний мир (файл уже рядом) ===
        if tid is None:
            flight_js = """
;(function(){
    var gd = document.querySelector('.plotly-graph-div');
    if(!gd){ return; }
    gd.on('plotly_click', function(ev){
      if(!ev.points || !ev.points.length) return;
      var pt = ev.points[0];
      if(!pt.customdata) return;
      var tid = pt.customdata;
      var x = pt.x, y = pt.y, z = pt.z;
      var cam = gd._fullLayout.scene.camera || {eye:{x:1.25,y:1.25,z:1.25}};
      var e0 = cam.eye;
      var t0 = null, dur = 1400;
      function fly(ts){
        if(!t0) t0 = ts;
        var k = Math.min(1, (ts - t0)/dur);
        k = k*k*(3-2*k);
        gd._fullLayout.scene.camera.eye = {
          x: e0.x + (x*0.65 - e0.x)*k,
          y: e0.y + (y*0.65 - e0.y)*k,
          z: e0.z + (z*0.65 + 6 - e0.z)*k
        };
        Plotly.relayout(gd, {'scene.camera': gd._fullLayout.scene.camera});
        if(k < 1){ requestAnimationFrame(fly); }
        else { window.location.href = 'neurons/neuron_' + tid + '.html'; }
      }
      requestAnimationFrame(fly);
    });
})();
"""
            # полёт живёт в post_script: gd готов, клики ловятся
            post = (getattr(fig, "_leta_anim_js", "") or "") + flight_js
            fig.write_html(str(out), include_plotlyjs=True,
                           post_script=post)

            # === папка нейронов: каждый — отдельный мир ===
            ndir = OUT_DIR / "neurons"
            ndir.mkdir(parents=True, exist_ok=True)
            conn2 = sqlite3.connect(str(FOREST))
            conn2.row_factory = sqlite3.Row
            made = 0
            for t in trees:
                npath = ndir / f"neuron_{t[0]}.html"
                if npath.exists():
                    made += 1
                    continue
                fig_n = build_fig(conn2, t[0], anim)
                if fig_n is None:
                    continue
                if isinstance(fig_n, tuple):
                    fig_n = fig_n[0]
                post_n = getattr(fig_n, "_leta_anim_js", "") or ""
                fig_n.write_html(str(npath), include_plotlyjs=True,
                                 post_script=post_n)
                made += 1
            # протоязык — тоже нейрон (детали знаков внутри)
            ppath = ndir / "neuron_proto.html"
            if not ppath.exists():
                fig_p = build_fig(conn2, None, anim, proto_only=True)
                if fig_p is not None:
                    if isinstance(fig_p, tuple):
                        fig_p = fig_p[0]
                    hop_p = """
;(function(){
    var gd = document.querySelector('.plotly-graph-div');
    if(!gd){ return; }
    gd.on('plotly_click', function(ev){
      if(!ev.points || !ev.points.length) return;
      var pt = ev.points[0];
      if(!pt.customdata) return;
      window.location.href = 'neuron_' + pt.customdata + '.html';
    });
})();
"""
                    post_p = (getattr(fig_p, "_leta_anim_js", "")
                              or "") + hop_p
                    fig_p.write_html(str(ppath), include_plotlyjs=True,
                                     post_script=post_p)
                    made += 1
            conn2.close()

            # === оверлей-меню: выбор нейрона без капризов 3D-клика ===
            opts = ['<option value="neurons/neuron_proto.html">'
                    'протоязык (знаки)</option>']
            for t in trees:
                opts.append(
                    f'<option value="neurons/neuron_{t[0]}.html">'
                    f'{t[1][:40]}</option>')
            menu_js = """
<div style="position:fixed;top:10px;left:10px;z-index:999;
     background:rgba(20,20,28,0.88);padding:8px 10px;border-radius:10px;
     font:13px system-ui;color:#ddd">
  <b>мозг Leta</b> ·
  <select id="leta_nav" style="max-width:260px;background:#1c1c26;
     color:#eee;border:1px solid #444;border-radius:6px;padding:3px 6px">
    <option value="">— выбрать нейрон —</option>
  """ + "\n".join(opts) + """
  </select>
  <span style="opacity:.6">клик по нейрону — тоже полетит</span>
</div>
<script>
(function(){
  var sel = document.getElementById('leta_nav');
  if(!sel){ return; }
  sel.addEventListener('change', function(){
    if(sel.value){ window.location.href = sel.value; }
  });
})();
</script>
"""
            html = out.read_text(encoding="utf-8")
            html = html.replace("</body>", menu_js + "</body>")
            out.write_text(html, encoding="utf-8")

            print(f"нарисовано: {out} + {made} нейронов в viz/neurons/")
            print("нейрон: клик по нему или меню слева сверху")
            subprocess.run(["open", str(out)])
            return

        # одиночный файл дерева: перелёты по аксонам к соседям
        hop_js = """
;(function(){
    var gd = document.querySelector('.plotly-graph-div');
    if(!gd){ return; }
    gd.on('plotly_click', function(ev){
      if(!ev.points || !ev.points.length) return;
      var pt = ev.points[0];
      if(!pt.customdata) return;
      var tid = pt.customdata;
      var x = pt.x, y = pt.y, z = pt.z;
      var cam = gd._fullLayout.scene.camera || {eye:{x:1.25,y:1.25,z:1.25}};
      var e0 = cam.eye;
      var t0 = null, dur = 1200;
      function fly(ts){
        if(!t0) t0 = ts;
        var k = Math.min(1, (ts - t0)/dur);
        k = k*k*(3-2*k);
        gd._fullLayout.scene.camera.eye = {
          x: e0.x + (x*0.7 - e0.x)*k,
          y: e0.y + (y*0.7 - e0.y)*k,
          z: e0.z + (z*0.7 - e0.z)*k
        };
        Plotly.relayout(gd, {'scene.camera': gd._fullLayout.scene.camera});
        if(k < 1){ requestAnimationFrame(fly); }
        else { window.location.href = 'neuron_' + tid + '.html'; }
      }
      requestAnimationFrame(fly);
    });
})();
"""
        post = (getattr(fig, "_leta_anim_js", "") or "") + hop_js
        fig.write_html(str(out), include_plotlyjs=True,
                       post_script=post)
        # соседям тоже — перелёт возможен от каждого
        print(f"нарисовано: {out} (импульсы {anim}/сек, радуга, вечно, "
              "вращение свободно — plotly вшит)")
        subprocess.run(["open", str(out)])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
