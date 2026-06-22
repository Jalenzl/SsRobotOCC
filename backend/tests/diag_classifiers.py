"""
诊断脚本：遍历所有 fixture STEP，输出每个 contour 的关键指标和分类结果。

对落入 `irregular` 的 contour，打印"为什么没被任何一个分类器认领"的详细原因。

用法（conda occ 环境）:
    cd backend
    python -c "import sys; sys.argv = ['diag']; exec(open('tests/diag_classifiers.py').read())"
"""
import sys
from pathlib import Path

import math

# ── Setup ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.occ.contour import analyze_part, _extract_face_wires
from app.occ.classifiers.base import compute_metrics
from app.occ.classifiers.registry import ClassifierRegistry

FIXTURES = [
    "tests/fixtures/cad/plate_with_hole_100.step",   # 圆形孔
    "tests/fixtures/cad/plate_with_slot_100.step",   # 槽形
    "tests/fixtures/cad/plate_with_rect_100.step",   # 矩形
    "tests/fixtures/cad/plate_with_blind_hole.step", # 盲孔（顶面圆形 + 侧面可能非圆）
    "tests/fixtures/cad/plate_with_boss.step",       # 凸台（外圆轮廓）
    "tests/fixtures/cad/box_100x60x20.step",         # 纯板（只有外边缘线）
    "tests/fixtures/cad/Guideclamp.STEP",
    "tests/fixtures/cad/user_sample.STEP",
]

registry = ClassifierRegistry()


def _why_irregular(pts2d):
    """逐个打印每个分类器的拒绝原因（供 irregular 兜底时诊断）。"""
    m = compute_metrics(pts2d)
    reasons = []
    from app.occ.classifiers import circle, slot, rectangle, hexagon

    # Circle
    c = circle.CircleClassifier()
    reasons.append(f"  circle: n={m.n} area={m.area_2d:.2f} aspect={m.aspect:.4f} circ={m.circularity:.4f}")
    if m.n < c.min_points or m.area_2d <= 0:
        reasons.append(f"    -> FAIL: n={m.n} < {c.min_points} or area={m.area_2d}")
    elif m.aspect > 1.2 or m.aspect < 0.83:
        reasons.append(f"    -> FAIL: aspect={m.aspect:.4f} outside [0.83, 1.2]")
    else:
        fit = circle._fit_circle_2d(m.pts2d)
        if fit:
            cx, cy, radius = fit
            radii = [math.hypot(p[0]-cx, p[1]-cy) for p in m.pts2d]
            mean_r = sum(radii) / len(radii)
            rel = sum(abs(r - mean_r) for r in radii) / (len(radii) * mean_r)
            arc_ok = rel < circle._ARC_FIT_REL_TOL
            reasons.append(f"    -> arc_fit rel={rel:.4f} (tol={circle._ARC_FIT_REL_TOL}) {'PASS' if arc_ok else 'FAIL'}, circ={m.circularity} (need {circle._CIRCULARITY_CIRCLE})")
        else:
            reasons.append(f"    -> FAIL: arc_fit returned None, circ={m.circularity} (need {circle._CIRCULARITY_CIRCLE})")
            if m.circularity < circle._CIRCULARITY_CIRCLE:
                reasons.append(f"    -> FAIL: circularity {m.circularity:.4f} < {circle._CIRCULARITY_CIRCLE}")

    # Slot
    s = slot.SlotClassifier()
    s_match = s.matches(m)
    reasons.append(f"  slot: matches={s_match}")
    if s_match:
        params = s.classify(m, face_normal=None)
        reasons.append(f"    -> L={params.get('length')} W={params.get('width')}")

    # Rectangle
    r = rectangle.RectangleClassifier()
    r_match = r.matches(m)
    reasons.append(f"  rectangle: matches={r_match}")
    if r_match:
        params = r.classify(m, face_normal=None)
        reasons.append(f"    -> L={params.get('length')} W={params.get('width')}")

    # Hexagon
    h = hexagon.HexagonClassifier()
    h_match = h.matches(m)
    reasons.append(f"  hexagon: matches={h_match}")

    return reasons


def diagnose_file(step_path: str) -> None:
    path = Path(step_path)
    if not path.exists():
        print(f"[SKIP] {path} (not found)")
        return

    print(f"\n{'='*80}")
    print(f"FILE: {path}")
    print('='*80)

    try:
        result = analyze_part(str(path), linear_deflection=0.1, per_face=True)
    except Exception as e:
        print(f"[ERROR] analyze_part failed: {e}")
        return

    contours = result.get("contours", [])
    if not contours:
        print("[INFO] No contours found")
        return

    irregular_count = 0
    for c in contours:
        ct = c.get("contour_type", "?")
        m_raw = c.get("metrics", {})
        pts2d = c.get("pts2d", [])

        # Re-derive metrics
        if pts2d:
            m = compute_metrics(pts2d)
        else:
            m = None

        tag = "<<< IRREGULAR" if ct == "irregular" else ""
        print(f"\n  [{ct}] {tag}")
        print(f"    is_outer={c.get('is_outer')} is_hole={c.get('is_hole')} face_idx={c.get('face_idx')}")
        if m:
            print(f"    n={m.n}  area={m.area_2d:.4f}  perimeter={m.perimeter:.4f}")
            print(f"    aspect={m.aspect:.4f}  bbox={m.bbox}")
            print(f"    circularity={m.circularity:.4f}  solidity={m.solidity:.4f}")
            print(f"    center={c.get('center')}  normal={c.get('normal')}")
        else:
            print(f"    (no pts2d available)")
        print(f"    D={c.get('parameters',{}).get('diameter')}  L={c.get('parameters',{}).get('length')}  W={c.get('parameters',{}).get('width')}")
        print(f"    _confidence={c.get('_confidence')}")

        if ct == "irregular" and pts2d:
            irregular_count += 1
            print("    --- why irregular ---")
            for line in _why_irregular(pts2d):
                print(line)

    print(f"\n  SUMMARY: {len(contours)} contours, {irregular_count} irregular")


def main() -> None:
    for fp in FIXTURES:
        diagnose_file(fp)


if __name__ == "__main__":
    main()
