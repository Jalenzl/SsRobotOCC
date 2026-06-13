"""3D 特征识别测试：通孔 / 盲孔 / 凸台 + 内外表面判定 + 装配体扩散。"""

from __future__ import annotations

import pytest

from app.models.cad import CadAnalyzeOptions, CadAnalyzeResult, CadFaceSpreadResult
from app.occ import occ_available

pytestmark = pytest.mark.skipif(
    not occ_available(),
    reason="pythonOCC not installed (use conda env occ)",
)

from app.occ.features.extractor import (  # noqa: E402
    extract_all_features,
    extract_face_spread_features,
)
from app.occ.features.face_side import FaceSideClassifier  # noqa: E402
from tests.fixtures.cad.generate_fixtures import (  # noqa: E402
    make_assembly_two_solids_shape,
    make_box_shape,
    make_plate_with_blind_hole_shape,
    make_plate_with_boss_shape,
    make_plate_with_hole_shape,
)


def _analyze(shape, **opts) -> dict:
    return extract_all_features(shape, CadAnalyzeOptions(**opts))


# --------------------------------------------------------------------------- #
# 通孔 / 盲孔 / 凸台 的方向、类型、深度
# --------------------------------------------------------------------------- #


class TestDepthRecognition:
    def test_through_hole(self):
        res = _analyze(make_plate_with_hole_shape(100, 10, 15))
        through = [h for h in res["holes"] if h["kind"] == "through"]
        assert through, "应识别出至少一个通孔"
        h = max(through, key=lambda x: x.get("depth") or 0.0)
        assert h["direction"] == "recess"
        assert h["through"] is True
        assert h["depth"] == pytest.approx(10.0, abs=1.0)

    def test_blind_hole(self):
        res = _analyze(make_plate_with_blind_hole_shape(100, 20, 12, 8))
        blind = [h for h in res["holes"] if h["kind"] == "blind"]
        assert blind, "应识别出盲孔"
        h = blind[0]
        assert h["direction"] == "recess"
        assert h["through"] is False
        assert h["depth"] == pytest.approx(8.0, abs=1.0)

    def test_boss_protrusion(self):
        res = _analyze(make_plate_with_boss_shape(100, 10, 15, 12))
        boss = [h for h in res["holes"] if h["kind"] == "boss"]
        assert boss, "应识别出凸台"
        h = boss[0]
        assert h["direction"] == "protrusion"
        assert h["depth"] == pytest.approx(12.0, abs=1.0)

    def test_depth_disabled_falls_back(self):
        res = _analyze(make_plate_with_blind_hole_shape(100, 20, 12, 8), enable_depth=False)
        # 关闭 3D 后 depth 应为空、kind 回退轮廓类型
        for h in res["holes"]:
            assert h["depth"] is None
            assert h["direction"] is None


# --------------------------------------------------------------------------- #
# 内 / 外表面判定
# --------------------------------------------------------------------------- #


class TestFaceSide:
    def test_box_all_outer(self):
        res = _analyze(make_box_shape(100, 60, 20))
        sides = {f["side"] for f in res["faces"]}
        assert sides == {"outer"}

    def test_hole_wall_is_inner(self):
        res = _analyze(make_plate_with_hole_shape(100, 10, 15))
        inner = [f for f in res["faces"] if f["side"] == "inner"]
        outer = [f for f in res["faces"] if f["side"] == "outer"]
        assert inner, "孔壁应判定为内表面"
        assert len(outer) >= 6

    def test_classifier_score_sign(self):
        shape = make_plate_with_hole_shape(100, 10, 15)
        clf = FaceSideClassifier(shape)
        for face in clf.faces:
            info = clf.classify(face)
            if info["side"] == "outer":
                assert info["score"] >= 0.0
            elif info["side"] == "inner":
                assert info["score"] < 0.0


# --------------------------------------------------------------------------- #
# 装配体多实体 + 内/外表面扩散
# --------------------------------------------------------------------------- #


class TestSpreadAssembly:
    def test_assembly_two_solids(self):
        res = _analyze(make_assembly_two_solids_shape(60, 10, 40, 10))
        assert res["summary"]["solid_count"] == 2

    def test_spread_outer_covers_assembly(self):
        shape = make_assembly_two_solids_shape(60, 10, 40, 10)
        clf = FaceSideClassifier(shape)
        outer_idx = next(i for i, f in enumerate(clf.faces) if clf.classify(f)["side"] == "outer")
        raw = extract_face_spread_features(shape, CadAnalyzeOptions(), face_id=f"face_{outer_idx}")
        model = CadFaceSpreadResult(**raw)
        assert model.side == "outer"
        assert model.solid_count == 2
        # 外表面扩散应覆盖两块板的多数外蒙皮面
        assert len(model.face_ids) >= 10

    def test_spread_inner_only_inner_faces(self):
        shape = make_plate_with_hole_shape(100, 10, 15)
        clf = FaceSideClassifier(shape)
        inner_idx = next(i for i, f in enumerate(clf.faces) if clf.classify(f)["side"] == "inner")
        raw = extract_face_spread_features(shape, CadAnalyzeOptions(), face_id=f"face_{inner_idx}")
        model = CadFaceSpreadResult(**raw)
        assert model.side == "inner"
        for f in model.faces:
            assert f.side == "inner"

    def test_spread_schema_roundtrip(self):
        shape = make_plate_with_boss_shape(100, 10, 15, 12)
        raw = extract_face_spread_features(shape, CadAnalyzeOptions(), face_id="face_0")
        model = CadFaceSpreadResult(**raw)
        assert model.target_face_id == "face_0"
        assert model.work_plane_normal is not None
