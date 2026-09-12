"""Визуализация мозга Leta: нейроны-деревья и связи между ними (graphviz).

  python tools/brain_viz.py        — все деревья без глубины + связи-аксоны
  python tools/brain_viz.py N      — полное дерево N (сплиты, листья, мосты)
"""
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOREST = ROOT / "brain" / "forest.sqlite"
OUT_DIR = ROOT / "brain" / "viz"


def esc(s):
    return (str(s).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", " ").replace("\r", " "))


def viz_all(conn):
    trees = conn.execute("SELECT id, name, trunk FROM trees").fetchall()
    if not trees:
        return None
    leaf_counts = dict(conn.execute(
        "SELECT tree_id, COUNT(*) FROM nodes WHERE is_leaf=1 GROUP BY tree_id"
    ).fetchall())
    axons = conn.execute(
        "SELECT a_tree, b_tree, weight FROM axons").fetchall()
    lines = [
        "digraph forest {",
        "  rankdir=LR;",
        "  bgcolor=\"white\";",
        "  node [shape=box, style=\"rounded,filled\", fillcolor=\"#cfe8ff\", "
        "fontname=\"Helvetica\"];",
        "  edge [fontname=\"Helvetica\", fontsize=10];",
    ]
    for tid, name, trunk in trees:
        n = leaf_counts.get(tid, 0)
        lines.append(
            f'  t{tid} [label="{esc(name)}\\n{n} листьев\\n'
            f'{esc((trunk or "")[:60])}"];')
    for a, b, w in axons:
        lines.append(
            f'  t{a} -> t{b} [label="{w:.2f}", penwidth={1 + w * 3:.1f}, '
            f'color="#888888"];')
    lines.append("}")
    return "\n".join(lines)


def viz_tree(conn, tid):
    t = conn.execute(
        "SELECT name, trunk FROM trees WHERE id=?", (tid,)).fetchone()
    if not t:
        return None
    name, trunk = t
    rows = conn.execute(
        "SELECT id, parent_id, side, is_leaf, fact FROM nodes WHERE tree_id=?",
        (tid,)).fetchall()
    axons = conn.execute(
        "SELECT a_tree, b_tree, weight FROM axons WHERE a_tree=? OR b_tree=?",
        (tid, tid)).fetchall()
    names = dict(conn.execute("SELECT id, name FROM trees").fetchall())
    lines = [
        f'digraph tree{tid} {{',
        "  bgcolor=\"white\";",
        "  node [fontname=\"Helvetica\", fontsize=10];",
        "  edge [fontname=\"Helvetica\", fontsize=9];",
        f'  label="{esc(name)} — {esc((trunk or "")[:70])}";',
        "  labelloc=t; fontsize=14;",
    ]
    for nid, pid, side, is_leaf, fact in rows:
        if is_leaf:
            lines.append(
                f'  n{nid} [shape=box, style="filled", fillcolor="#d5f5d5", '
                f'label="{esc((fact or "")[:60])}"];')
        else:
            lines.append(
                f'  n{nid} [shape=circle, style="filled", '
                f'fillcolor="#cfe8ff", label="Z{nid}"];')
    for nid, pid, side, is_leaf, fact in rows:
        if pid is not None:
            lbl = "R" if side == 1 else "L"
            lines.append(f'  n{pid} -> n{nid} [label="{lbl}"];')
    for a, b, w in axons:
        other = b if a == tid else a
        oname = names.get(other, "?")
        lines.append(
            f'  t{other} [shape=box, style="rounded,filled", '
            f'fillcolor="#f0e0f0", label="{esc(oname[:30])}"];')
        lines.append(
            f'  t{other} -> n{rows[0][0]} [style=dashed, color="#aa66aa", '
            f'label="аксон {w:.2f}", constraint=false];')
    lines.append("}")
    return "\n".join(lines)


def render(dot_text, out_path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_file = out_path.with_suffix(".dot")
    dot_file.write_text(dot_text, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", str(dot_file), "-o", str(out_path)],
                   check=True, timeout=60)
    return out_path


def main():
    if not FOREST.exists():
        print("Леса ещё нет — она ничего не выучила.")
        return
    conn = sqlite3.connect(str(FOREST))
    conn.row_factory = sqlite3.Row
    try:
        trees = conn.execute(
            "SELECT id, name FROM trees ORDER BY id").fetchall()
        if not trees:
            print("Лес пуст — деревьев нет.")
            return
        print("Деревья-нейроны:")
        for t in trees:
            print(f"  [{t['id']}] {t['name']}")
        arg = sys.argv[1] if len(sys.argv) > 1 else None
        if arg and arg.isdigit():
            tid = int(arg)
            dot_text = viz_tree(conn, tid)
            if not dot_text:
                print(f"Дерева {tid} нет.")
                return
            out = render(dot_text, OUT_DIR / f"tree_{tid}.png")
        else:
            dot_text = viz_all(conn)
            out = render(dot_text, OUT_DIR / "forest.png")
        print(f"нарисовано: {out}")
        subprocess.run(["open", str(out)])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
