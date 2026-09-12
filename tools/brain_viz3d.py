"""3D-визуализация мозга Leta: нейроны-деревья и связи (plotly, HTML).

  python tools/brain_viz3d.py            — весь мозг в 3D
  python tools/brain_viz3d.py N          — одно дерево N полностью
  python tools/brain_viz3d.py --anim 2   — анимация сигналов, 2 кадра/сек
  python tools/brain_viz3d.py N --anim 4 — одно дерево + анимация 4/сек

Знаки протоязыка — красный кластер «язык». Импульсы бегут по ветвям
деревьев и аксонам-мостам. HTML: вращение, зум, hover — факты.
"""
import math
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


def build_fig(conn, tree_id=None, anim_fps=0):
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
    if not trees:
        return None
    if tree_id is not None:
        trees = [t for t in trees if t[0] == tree_id]
        if not trees:
            return None
    tree_data = []  # соседние деревья, подключённые аксонами

    n = len(trees)
    centers = {}
    for i, (tid, name, trunk) in enumerate(trees):
        if n == 1:
            centers[tid] = (0.0, 0.0, 0.0)
        else:
            ang = 2 * math.pi * i / n
            R = 12.0 * math.sqrt(n)
            centers[tid] = (R * math.cos(ang), 0.0, R * math.sin(ang))

    axons = conn.execute("SELECT a_tree, b_tree, weight FROM axons").fetchall()

    # все рёбра — для анимации импульсов
    all_edges = []  # ((x1,y1,z1),(x2,y2,z2), kind)

    for tid, name, trunk in trees:
        rows = conn.execute(
            "SELECT id, parent_id, side, is_leaf, fact FROM nodes"
            " WHERE tree_id=?", (tid,)).fetchall()
        if not rows:
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

    # аксионы-мосты
    ax, ay, az = [], [], []
    for a, b, w in axons:
        if a not in centers or b not in centers:
            # режим одного дерева: показать соседей и мосты к ним
            if tree_id is not None and (a == tree_id or b == tree_id):
                other = b if a == tree_id else a
                orow = conn.execute(
                    "SELECT name, trunk FROM trees WHERE id=?").fetchone()
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
        all_edges.append(((x1, y1, z1), (x2, y2, z2), "axon"))
    # соседние деревья (подключённые аксонами в режиме одного дерева)
    for tid2, name2, trunk2 in tree_data:
        cx, cy, cz = centers[tid2]
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz], mode="markers",
            marker=dict(size=14, color="#c9a0dc", opacity=0.95),
            text=[f"«{name2}» (сосед по аксонам) — {(trunk2 or '')[:80]}"],
            hoverinfo="text", name=name2[:25], showlegend=True))
    if ax:
        fig.add_trace(go.Scatter3d(
            x=ax, y=ay, z=az, mode="lines",
            line=dict(color="#b06ad4", width=5),
            name="аксоны-мосты", showlegend=True))

    # протоязык: кластеры по языкам
    signs, sign_links, sign_tree_links = read_proto()
    if signs and tree_id is None:
        lang_names = {"ru": "русский", "en": "english", "cjk": "漢字",
                      "mix": "смешанный"}
        lang_colors = {"ru": "#e04444", "en": "#e0a020", "cjk": "#20b2aa",
                       "mix": "#a060a0"}
        by_lang = {}
        for row in signs[:400]:
            by_lang.setdefault(row[3] or "mix", []).append(row)
        base_y = -20.0 if n > 1 else -14.0
        s_pos = {}
        for li, (lname, rows) in enumerate(sorted(by_lang.items())):
            n_signs = len(rows)
            lang_c = ((li - (len(by_lang) - 1) / 2) * 14.0, base_y, 0.0)
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
        # связи знаков — её грамматика
        lx, ly, lz = [], [], []
        for a, b, w in sign_links:
            if a in s_pos and b in s_pos:
                lx += [s_pos[a][0], s_pos[b][0], None]
                ly += [s_pos[a][1], s_pos[b][1], None]
                lz += [s_pos[a][2], s_pos[b][2], None]
                all_edges.append((s_pos[a], s_pos[b], "lang"))
        if lx:
            fig.add_trace(go.Scatter3d(
                x=lx, y=ly, z=lz, mode="lines",
                line=dict(color="#e08a8a", width=3),
                name="грамматика знаков", showlegend=True))
        # связи знак-дерево: язык открывает знания (SNN-ядро)
        kx, ky, kz = [], [], []
        for sid, tid2, w in sign_tree_links:
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
        # анимация: resty-интервал (вращение работает параллельно)
        post = getattr(fig, "_leta_anim_js", "") or ""
        fig.write_html(str(out), include_plotlyjs=True,
                       post_script=post)
        print(f"нарисовано: {out} (импульсы {anim}/сек, радуга, вечно, "
              "вращение свободно — plotly вшит)")
        subprocess.run(["open", str(out)])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
