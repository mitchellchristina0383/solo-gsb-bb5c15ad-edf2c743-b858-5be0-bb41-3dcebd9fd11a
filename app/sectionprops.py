"""截面惯性矩分析（section-properties，只读即时分析）。

对一份已存 SVG path 即时解析并完成轮廓包含层级分类后（与 metrics /
clearance / scanlines 等同一份解析与曲线细分折线，见
:func:`app.svggeo.analyze_contours` 与
:func:`app.svggeo.flatten_subpath`），对每个**有效闭合环**
（``outer`` / ``hole``）在细分多边形上按格林公式（鞋带推广）积分
面积及一、二阶面积矩，再按**包含层级交替**合成净填充截面：

- 外环（``outer``，深度偶数）**+** 计入；
- 孔洞（``hole``，深度奇数）**−** 扣减；
- 孔中“岛”（深度 2）重新计入，岛中孔继续扣减，依此类推。

合成净填充面积 ``area`` 与面积质心 ``centroid`` 后，返回**绕净填充
质心**的截面惯性张量：

- ``ix  = ∫(y−cy)² dA``（关于过质心、沿 x 方向的水平轴）
- ``iy  = ∫(x−cx)² dA``（关于过质心、沿 y 方向的竖直轴）
- ``ixy = ∫(x−cx)(y−cy) dA``（惯性积）

以及两项主惯性矩 ``i_max`` / ``i_min`` 和**较大主值对应主轴**的方向角
``axis_angle_deg``：在 SVG 屏幕坐标（y 轴向下）下从 +x 朝 +y 计量，
归一到 ``[0, 180)`` 度；两项主值相等（任意过质心的轴均为主轴，如
正方形/圆截面）时轴角返回 ``0.0``。

- 各环积分先按环自身书写方向求出有符号值，统一翻转到正方向后再按
  角色加减，因此**结果与环的书写方向无关**，``nonzero`` / ``evenodd``
  两种填充规则下一致；
- 开放（``open``）与退化（``degenerate``）轮廓不参与，列入
  ``skipped_contours``；
- 没有任何有效闭合区域，或净填充面积被孔洞完全抵消时，``area`` 为
  ``0.0``，``centroid`` 与全部惯性矩、主值、轴角均为 ``null``；
- 整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import math
from typing import Any

from .svggeo import Contour

# 净面积小于该阈值（按轮廓尺寸缩放）视为“无有效填充区域”（与 metrics 同口径）
_AREA_EPS = 1e-10
# 二阶矩相关阈值（按尺寸四次方缩放）：浮点噪声钳制与主值相等判定
_MOMENT_EPS = 1e-10
# 输出量化位数（仅影响 JSON 展示，不改变判定）
_DECIMALS = 9


def compute_section_properties(contours: list[Contour]) -> dict[str, Any]:
    """对已分类的轮廓列表计算净填充截面的惯性矩与主轴。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``depth``）。

    返回:
        ``{area, centroid, ix, iy, ixy, i_max, i_min, axis_angle_deg,
        contributing_contour_count, contours, skipped_contours}``；
        无有效净填充区域时 ``centroid`` 与六项截面量均为 ``None``。
    """
    valid = [c for c in contours if c.role in ("outer", "hole")]
    valid.sort(key=lambda c: c.original_index)

    scale = 1.0
    for c in valid:
        diag = math.hypot(c.bounds[2] - c.bounds[0], c.bounds[3] - c.bounds[1])
        scale = max(scale, diag)

    total_area = 0.0
    moment_x = 0.0   # Σ ±∫x dA（一阶面积矩）
    moment_y = 0.0   # Σ ±∫y dA
    ixx_origin = 0.0  # Σ ±∫y² dA（关于原点坐标轴的二阶面积矩）
    iyy_origin = 0.0  # Σ ±∫x² dA
    ixy_origin = 0.0  # Σ ±∫xy dA
    items: list[dict[str, Any]] = []

    for c in valid:
        a, sx, sy, ixx, iyy, ixy = _ring_moments(c.points)
        # 方向无关：把环的带符号积分统一翻转到正方向，再按角色加减
        orient = 1.0 if a >= 0.0 else -1.0
        factor = orient if c.role == "outer" else -orient
        total_area += factor * a
        moment_x += factor * sx
        moment_y += factor * sy
        ixx_origin += factor * ixx
        iyy_origin += factor * iyy
        ixy_origin += factor * ixy
        items.append({
            "original_index": c.original_index,
            "role": c.role,
            "depth": c.depth,
            "area": _num(abs(a)),
            "signed_contribution": _num(factor * a),
            "centroid": {"x": _num(sx / a), "y": _num(sy / a)},
        })

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if c.role not in ("outer", "hole")
    ]

    # 净面积被孔洞完全抵消（理论上的等面积异常嵌套）或本就没有闭合区域
    if abs(total_area) <= _AREA_EPS * scale ** 2:
        return {
            "area": 0.0,
            "centroid": None,
            "ix": None,
            "iy": None,
            "ixy": None,
            "i_max": None,
            "i_min": None,
            "axis_angle_deg": None,
            "contributing_contour_count": len(valid),
            "contours": items,
            "skipped_contours": skipped,
        }

    cx = moment_x / total_area
    cy = moment_y / total_area
    # 平行移轴：关于原点的二阶矩 -> 绕净填充质心的二阶矩
    ix = ixx_origin - cy * cy * total_area
    iy = iyy_origin - cx * cx * total_area
    ixy = ixy_origin - cx * cy * total_area

    # 数学上绕质心的惯性矩非负，钳制浮点噪声产生的微小负值
    eps4 = _MOMENT_EPS * scale ** 4
    if -eps4 < ix < 0.0:
        ix = 0.0
    if -eps4 < iy < 0.0:
        iy = 0.0

    i_avg = 0.5 * (ix + iy)
    radius = math.hypot(0.5 * (ix - iy), ixy)
    i_max = i_avg + radius
    i_min = i_avg - radius
    if -eps4 < i_min < 0.0:
        i_min = 0.0

    if radius <= eps4:
        # 主值相等（如正方形/圆截面）：任意方向均为主轴，约定取 0°
        axis_angle = 0.0
    else:
        # i_max 对应主轴方向：I(θ) 极大支满足 tan2θ = 2·ixy / (iy − ix)；
        # y 轴向下的坐标系中 atan2 即“从 +x 朝 +y”的计量方向
        theta = 0.5 * math.atan2(-2.0 * ixy, ix - iy)
        axis_angle = _num(math.degrees(theta % math.pi))
        if axis_angle >= 180.0:
            axis_angle = 0.0  # 量化后顶到 180° 等价于 0°（轴向无正负）

    return {
        "area": _num(total_area),
        "centroid": {"x": _num(cx), "y": _num(cy)},
        "ix": _num(ix),
        "iy": _num(iy),
        "ixy": _num(ixy),
        "i_max": _num(i_max),
        "i_min": _num(i_min),
        "axis_angle_deg": axis_angle,
        "contributing_contour_count": len(valid),
        "contours": items,
        "skipped_contours": skipped,
    }


def _ring_moments(points: list[tuple[float, float]]
                  ) -> tuple[float, float, float, float, float, float]:
    """闭合细分环的带符号面积矩积分（格林公式 / 鞋带公式推广）。

    返回 ``(A, Sx, Sy, Ixx, Iyy, Ixy)``：

    - ``A``   ：有符号面积（y 轴向下坐标系中正 = 顺时针）
    - ``Sx``  = ∫x dA，``Sy`` = ∫y dA（一阶矩）
    - ``Ixx`` = ∫y² dA，``Iyy`` = ∫x² dA，``Ixy`` = ∫xy dA（关于原点轴）

    全部量随环绕方向整体变号；调用方按角色与方向翻转后累加。
    """
    a = sx = sy = ixx = iyy = ixy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        a += cross
        sx += (x1 + x2) * cross
        sy += (y1 + y2) * cross
        ixx += (y1 * y1 + y1 * y2 + y2 * y2) * cross
        iyy += (x1 * x1 + x1 * x2 + x2 * x2) * cross
        ixy += (x1 * y2 + 2.0 * x1 * y1 + 2.0 * x2 * y2 + x2 * y1) * cross
    return (a / 2.0, sx / 6.0, sy / 6.0,
            ixx / 12.0, iyy / 12.0, ixy / 24.0)


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
