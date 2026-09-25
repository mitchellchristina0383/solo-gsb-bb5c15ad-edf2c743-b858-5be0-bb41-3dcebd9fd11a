"""sectionprops：净填充截面惯性矩与主轴角单元测试。"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.sectionprops import _ring_moments, compute_section_properties  # noqa: E402
from app.svggeo import analyze_contours  # noqa: E402


def _props(d, fill_rule="nonzero"):
    contours, _warnings = analyze_contours(d, fill_rule)
    return compute_section_properties(contours)


def test_ring_moments_unit_square():
    # 单位正方形（y 向下顺时针）：A=1, Sx=Sy=1/2, Ixx=Iyy=1/3, Ixy=1/4
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    a, sx, sy, ixx, iyy, ixy = _ring_moments(pts)
    assert a == 1.0
    assert sx == 0.5 and sy == 0.5
    assert math.isclose(ixx, 1.0 / 3.0, rel_tol=1e-12)
    assert math.isclose(iyy, 1.0 / 3.0, rel_tol=1e-12)
    assert ixy == 0.25
    # 反向后全部变号
    a2, sx2, sy2, ixx2, iyy2, ixy2 = _ring_moments(list(reversed(pts)))
    assert a2 == -a and sx2 == -sx and sy2 == -sy
    assert ixx2 == -ixx and iyy2 == -iyy and ixy2 == -ixy


def test_single_outer_square():
    # 10x10 正方形：Ix=Iy=10*10^3/12，主值相等 -> 轴角 0
    p = _props("M0,0h10v10h-10Z")
    assert p["area"] == 100.0
    assert p["centroid"] == {"x": 5.0, "y": 5.0}
    assert math.isclose(p["ix"], 10000.0 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["iy"], 10000.0 / 12.0, abs_tol=1e-6)
    assert p["ixy"] == 0.0
    assert math.isclose(p["i_max"], 10000.0 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["i_min"], 10000.0 / 12.0, abs_tol=1e-6)
    assert p["axis_angle_deg"] == 0.0
    assert p["contributing_contour_count"] == 1
    assert p["skipped_contours"] == []
    item = p["contours"][0]
    assert item["role"] == "outer" and item["depth"] == 0
    assert item["signed_contribution"] == 100.0
    assert item["centroid"] == {"x": 5.0, "y": 5.0}


def test_outer_minus_hole():
    # 外方 0..10 扣内孔 2..8：净面积 64，Ix=Iy=(10000-1296)/12
    p = _props("M0,0h10v10h-10Z M2,2h6v6h-6Z")
    assert p["area"] == 64.0
    assert p["centroid"] == {"x": 5.0, "y": 5.0}
    assert math.isclose(p["ix"], 8704.0 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["iy"], 8704.0 / 12.0, abs_tol=1e-6)
    assert p["ixy"] == 0.0
    assert math.isclose(p["i_max"], 8704.0 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["i_min"], 8704.0 / 12.0, abs_tol=1e-6)
    assert p["axis_angle_deg"] == 0.0
    assert p["contributing_contour_count"] == 2
    assert p["contours"][1]["signed_contribution"] == -36.0


def test_wide_rectangle_axis_90():
    # 20x10 横放矩形：较大主值 Iy 对应竖直主轴 -> 90°
    p = _props("M0,0h20v10h-20Z")
    assert p["area"] == 200.0
    assert p["centroid"] == {"x": 10.0, "y": 5.0}
    assert math.isclose(p["ix"], 20.0 * 10.0 ** 3 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["iy"], 10.0 * 20.0 ** 3 / 12.0, abs_tol=1e-6)
    assert p["ixy"] == 0.0
    assert math.isclose(p["i_max"], 10.0 * 20.0 ** 3 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["i_min"], 20.0 * 10.0 ** 3 / 12.0, abs_tol=1e-6)
    assert p["axis_angle_deg"] == 90.0


def test_tall_rectangle_axis_0():
    # 10x20 竖放矩形：较大主值 Ix 对应水平主轴 -> 0°
    p = _props("M0,0h10v20h-10Z")
    assert math.isclose(p["i_max"], 10.0 * 20.0 ** 3 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["i_min"], 20.0 * 10.0 ** 3 / 12.0, abs_tol=1e-6)
    assert p["axis_angle_deg"] == 0.0


def test_hole_with_island_offcentroid():
    # 外 100 - 孔 64 + 岛 4 = 40，净质心 (5.2, 5.2)；岛在 y=x 方向偏移
    d = "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z"
    p = _props(d)
    assert p["area"] == 40.0
    assert math.isclose(p["centroid"]["x"], 5.2, abs_tol=1e-9)
    assert math.isclose(p["centroid"]["y"], 5.2, abs_tol=1e-9)
    # Ix = 4768/3 - 5.2²*40 = 507.7333…，Ixy = 1096 - 5.2²*40 = 14.4
    assert math.isclose(p["ix"], 4768.0 / 3.0 - 1081.6, abs_tol=1e-6)
    assert math.isclose(p["iy"], 4768.0 / 3.0 - 1081.6, abs_tol=1e-6)
    assert math.isclose(p["ixy"], 14.4, abs_tol=1e-9)
    assert math.isclose(p["i_max"], 4768.0 / 3.0 - 1081.6 + 14.4, abs_tol=1e-6)
    assert math.isclose(p["i_min"], 4768.0 / 3.0 - 1081.6 - 14.4, abs_tol=1e-6)
    # 质量沿 y=x 分布，较大主值轴与其垂直 -> 135°
    assert math.isclose(p["axis_angle_deg"], 135.0, abs_tol=1e-6)


def test_four_level_alternation():
    # 外 100 - 孔 64 + 岛 16 - 岛中孔 4 = 48，全部同心 -> Ix=Iy=512，Ixy=0
    d = ("M0,0h10v10h-10Z M1,1h8v8h-8Z "
         "M3,3h4v4h-4Z M4,4h2v2h-2Z")
    p = _props(d)
    assert p["area"] == 48.0
    assert p["centroid"] == {"x": 5.0, "y": 5.0}
    assert math.isclose(p["ix"], 512.0, abs_tol=1e-6)
    assert math.isclose(p["iy"], 512.0, abs_tol=1e-6)
    assert p["ixy"] == 0.0
    assert p["axis_angle_deg"] == 0.0
    assert p["contributing_contour_count"] == 4


def test_rotated_rectangle_principal_axis():
    # 沿 30° 方向放置的 20x2 细长矩形：较大主值轴垂直长边 -> 120°
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    ux, uy = 10.0 * c, 10.0 * s          # 半长边方向
    vx, vy = -1.0 * s, 1.0 * c           # 半宽方向
    pts = [(ux + vx, uy + vy), (ux - vx, uy - vy),
           (-ux - vx, -uy - vy), (-ux + vx, -uy + vy)]
    d = "M" + " L".join(f"{x},{y}" for x, y in pts) + " Z"
    p = _props(d)
    assert math.isclose(p["area"], 40.0, abs_tol=1e-9)
    assert math.isclose(p["i_min"], 40.0 * 2.0 ** 2 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["i_max"], 40.0 * 20.0 ** 2 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["axis_angle_deg"], 120.0, abs_tol=1e-6)
    # 主轴角恒在 [0, 180) 内
    assert 0.0 <= p["axis_angle_deg"] < 180.0


def test_orientation_independent():
    # 两环同时反向书写，所有截面量完全一致
    d1 = "M0,0h10v10h-10Z M2,2h6v6h-6Z"
    d2 = "M0,10h10v-10h-10Z M2,8h6v-6h-6Z"
    assert _props(d1) == _props(d2)


def test_evenodd_same_result_as_nonzero():
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z M3,3h4v4h-4Z"
    assert _props(d, "nonzero") == _props(d, "evenodd")


def test_open_and_degenerate_skipped():
    # 外方 + 退化（往返零面积闭合）+ 开放折线：仅外方参与
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    p = _props(d)
    assert p["area"] == 100.0
    assert math.isclose(p["ix"], 10000.0 / 12.0, abs_tol=1e-6)
    assert p["contributing_contour_count"] == 1
    skipped = {(s["original_index"], s["reason"])
               for s in p["skipped_contours"]}
    assert skipped == {(1, "degenerate"), (2, "open")}


def test_empty_path_zero_and_nulls():
    p = _props("")
    assert p["area"] == 0.0
    assert p["centroid"] is None
    for key in ("ix", "iy", "ixy", "i_max", "i_min", "axis_angle_deg"):
        assert p[key] is None
    assert p["contributing_contour_count"] == 0
    assert p["contours"] == []


def test_only_open_contours_zero_and_nulls():
    p = _props("M0,0 L10,0 L10,10")
    assert p["area"] == 0.0
    assert p["ix"] is None and p["axis_angle_deg"] is None
    assert p["skipped_contours"][0]["reason"] == "open"


def test_only_degenerate_zero_and_nulls():
    p = _props("M0,0 L10,0 L0,0 Z")
    assert p["area"] == 0.0
    assert p["centroid"] is None
    assert p["i_max"] is None


def test_curve_flattened_circle():
    # 单位圆细分为 32 边形：Ix=Iy≈π/4（内接多边形略小），Ixy≈0，
    # 主值相等 -> 轴角 0
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    p = _props(d)
    assert 3.0 < p["area"] < math.pi
    assert p["centroid"] == {"x": 0.0, "y": 0.0}
    assert math.isclose(p["ix"], math.pi / 4.0, abs_tol=2e-2)
    assert p["ix"] == p["iy"]
    assert p["ixy"] == 0.0
    assert p["axis_angle_deg"] == 0.0


def test_quadratic_curve_positive_moments():
    # Q 描述的抛物线弓形同样走细分近似，惯性矩为正且主轴角在 [0,180)
    p = _props("M0,0 L10,0 Q10,10 0,10 Z")
    assert p["area"] > 0.0
    assert p["ix"] > 0.0 and p["iy"] > 0.0
    assert p["i_max"] >= p["i_min"] > 0.0
    assert 0.0 <= p["axis_angle_deg"] < 180.0


def test_negative_coordinates():
    p = _props("M-10,-10h20v20h-20Z")
    assert p["area"] == 400.0
    assert p["centroid"] == {"x": 0.0, "y": 0.0}
    assert math.isclose(p["ix"], 20.0 * 20.0 ** 3 / 12.0, abs_tol=1e-6)
    assert math.isclose(p["iy"], 20.0 * 20.0 ** 3 / 12.0, abs_tol=1e-6)
    assert p["ixy"] == 0.0


def test_no_negative_zero_in_output():
    p = _props("M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z")
    assert "-0.0" not in repr(p)
