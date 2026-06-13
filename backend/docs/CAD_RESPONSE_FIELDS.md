# CAD 接口返回字段说明

本文档描述 CAD 特征识别 API 的 JSON 响应结构。

**单位**：所有长度字段均为 `"mm"`（毫米）。

---

## 一、接口与响应类型对照

| 接口 | 响应类型 | 说明 |
|------|----------|------|
| `POST /api/v1/cad/analyze/face` | **CadFaceAnalyzeResult** | 单面提取（**推荐**） |
| `POST /api/v1/cad/analyze/face_spread` | **CadFaceSpreadResult** | 选面 → 内/外表面扩散到整个装配体同侧表面（含 3D 深度） |
| `POST /api/v1/cad/analyze/face_and_path` | **CadFaceAnalyzeAndPathResponse** | `{ analyze, path }` |
| `POST /api/v1/cad/path/generate/face` | PathPlanResult | 基于单面 analyze JSON |
| `POST /api/v1/cad/analyze` | CadAnalyzeResult | 全模型提取（兼容） |
| `POST /api/v1/cad/analyze_and_path` | CadAnalyzeAndPathResponse | 全模型 + 刀路 |

---

## 二、CadFaceAnalyzeResult（单面 — 推荐）

`schema_version`: `"1.0"`

### 2.1 顶层字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `schema_version` | string | 数据结构版本，当前 `"1.0"` |
| `unit` | string | 固定 `"mm"` |
| `target_face_id` | string | 本次分析的面 ID，如 `face_3` |
| `model_bbox` | object | 零件整体包围盒（非单面 bbox） |
| `face` | object | 该面元数据（见第六节） |
| `reference_points` | array | 参考点（面心、轮廓中心、孔心等） |
| `polylines` | array | 折线（画轮廓线） |
| `wires` | array | 线环 |
| `contours` | array | 轮廓（类型 + 中心 + 参数） |
| `outer_contours` | string[] | 该面上外轮廓 contour id |
| `holes` | array | 孔/凸台特征（含 `kind`/`direction`/`through`/`depth`） |
| `pockets` | array | 型腔（非圆凹陷，3D 识别后填充） |
| **`feature_groups`** | object | **按类型分组的特征索引（路径规划用）** |
| `work_plane` | string | `auto` / `xy` / `yz` / `xz` |
| `work_plane_normal` | object | 加工平面法向 `{x,y,z}` |

### 2.2 feature_groups（按类型存储）

便于路径规划模块按类型过滤，无需前端自行遍历 `contours[]` / `holes[]`。

| 字段 | 类型 | 含义 |
|------|------|------|
| `contours_by_type` | `dict[str, ContourFeature[]]` | 按 `contour_type` 分组 |
| `holes_by_type` | `dict[str, HoleFeature[]]` | 按孔类型分组 |
| `wires_by_role` | `dict[str, WireFeature[]]` | 按 `outer` / `inner` 分组 |

#### contours_by_type 常见 key

| key | 含义 |
|-----|------|
| `outer` | 外轮廓 |
| `circle` | 圆孔/圆环 |
| `slot` | 槽 |
| `rectangle` | 矩形 |
| `hexagon` | 六边形 |
| `unknown` | 未分类（弧面、非平面环等） |

#### holes_by_type 常见 key

| key | 含义 |
|-----|------|
| `circle` | 圆孔 |
| `slot` | 槽孔 |
| `rectangle` | 矩形孔 |
| `hexagon` | 六边形孔 |

#### wires_by_role

| key | 含义 |
|-----|------|
| `outer` | 外环 wire |
| `inner` | 内环 wire（孔口、型腔口） |

示例：

```json
{
  "target_face_id": "face_4",
  "feature_groups": {
    "contours_by_type": {
      "outer": [{ "id": "contour_0", "contour_type": "outer", "polyline_id": "poly_wire_face_4_0", "...": "..." }],
      "circle": [{ "id": "contour_1", "contour_type": "circle", "parameters": { "diameter": 30 }, "...": "..." }]
    },
    "holes_by_type": {
      "circle": [{ "id": "hole_contour_1", "diameter": 30, "...": "..." }]
    },
    "wires_by_role": {
      "outer": [{ "id": "wire_face_4_0", "is_outer": true, "...": "..." }],
      "inner": [{ "id": "wire_face_4_1", "is_outer": false, "...": "..." }]
    }
  }
}
```

### 2.3 model_bbox（零件包围盒）

| 字段 | 含义 |
|------|------|
| `xmin`, `ymin`, `zmin` | 最小角 |
| `xmax`, `ymax`, `zmax` | 最大角 |
| `center` | 包围盒中心 `{x,y,z}` |

---

## 三、CadAnalyzeResult（全模型 — 兼容）

`schema_version`: `"1.1"`

与单面结果相比：

- 有 `summary`（体积、面数、边数等），无 `target_face_id` / `model_bbox` / `feature_groups`
- `faces[]` 包含所有面
- `contours[]` / `holes[]` 包含全零件特征

| 字段 | 类型 | 含义 |
|------|------|------|
| `schema_version` | string | `"1.1"` |
| `unit` | string | `"mm"` |
| `summary` | object | 零件整体统计 |
| `reference_points` | array | 参考点 |
| `polylines` | array | 折线 |
| `faces` | array | 所有面 |
| `wires` | array | 所有线环 |
| `contours` | array | 所有轮廓 |
| `outer_contours` | string[] | 全局最大外轮廓 id |
| `holes` | array | 所有孔/凸台（含 `kind`/`direction`/`through`/`depth`） |
| `pockets` | array | 型腔（非圆凹陷，3D 识别后填充） |
| `work_plane` | string | 加工坐标系 |
| `work_plane_normal` | object | 法向 |

> 全模型 `analyze` 默认也启用 3D 深度识别（`enable_depth=true`）。`faces[]` 含 `side`/`side_score`。

---

## 三-B、CadFaceSpreadResult（选面 → 内/外表面扩散）

`schema_version`: `"1.0"`。由 `POST /api/v1/cad/analyze/face_spread` 返回。

流程：选中一个面 → 用外法向正负判定其为外表面/内表面 → 扩散到整个装配体的全部同侧表面 →
逐面复用单面算法并叠加 3D 深度识别（通孔/盲孔/型腔/凸台 + 深度）。

| 字段 | 类型 | 含义 |
|------|------|------|
| `schema_version` | string | `"1.0"` |
| `unit` | string | `"mm"` |
| `target_face_id` | string | 选中的种子面 id |
| `side` | string | 种子面侧别：`outer` 外表面 / `inner` 内表面 |
| `side_score` | number | 种子面内外判定得分 |
| `model_bbox` | object | 零件/装配体包围盒 |
| `solid_count` | int | 装配体实体数 |
| `face_ids` | string[] | 本次扩散覆盖的同侧面 id 列表 |
| `faces` | array | 同侧所有面（含 `side`/`side_score`） |
| `holes` | array | 同侧表面上的孔/凸台（含 `kind`/`direction`/`through`/`depth`） |
| `pockets` | array | 型腔（非圆凹陷，见下） |
| `contours` / `wires` / `polylines` / `reference_points` | array | 同单面结构 |
| `feature_groups` | object | 按类型分组索引 |
| `work_plane` / `work_plane_normal` | string / object | 加工坐标系 |

典型用法：选外表面任一面 → 得到整机外蒙皮全部孔/凸台；选内表面任一面 → 得到全部内腔/孔壁特征。

### pockets（型腔）

非圆凹陷（矩形/槽等）在 3D 识别为 `recess` 时写入。

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 如 `pocket_contour_3` |
| `depth` | number | 型腔深度 mm |
| `through` | bool \| null | 是否贯通 |
| `center` | `{x,y,z}` \| null | 口部中心 |
| `axis` | `{x,y,z}` \| null | 口部外法向 |
| `contour_type` | string \| null | `rectangle` / `slot` / `unknown` |
| `face_id` | string \| null | 口部所在面 |
| `bottom_face_id` | string \| null | 底面 id（预留） |
| `wire_ids` | string[] | 口部 wire id |
| `parameters` | object \| null | 口部 L/W 等参数 |

---

## 四、contours（轮廓 — 核心输出）

单面与全模型共用 **ContourFeature** 结构。

| 字段 | 含义 |
|------|------|
| `id` | 轮廓编号，如 `contour_0` |
| `contour_type` | `outer` / `circle` / `slot` / `rectangle` / `hexagon` / `unknown` |
| `center` | 几何中心 `{x,y,z}` |
| `normal` | 轮廓法向 `{x,y,z}` |
| `polyline_id` | 对应 `polylines[].id` |
| `wire_id` | 对应 `wires[].id` |
| `face_id` | 所属面 |
| `is_outer` | 是否外轮廓 |
| `area` | 投影面积 mm² |
| `perimeter` | 周长 mm |

### parameters（特征参数，按类型填写）

| contour_type | 使用字段 | 说明 |
|--------------|----------|------|
| **circle** | `diameter` | Φ 直径 (mm) |
| **slot** | `length`, `width` | L 槽长、W 槽宽 (mm) |
| **rectangle** | `length`, `width` | L 长、W 宽 (mm) |
| **hexagon** | `across_flats` | 对边长 (mm) |
| **outer** | `length`, `width` | 外接长宽（OBB） |

示例：

```json
{
  "id": "contour_1",
  "contour_type": "circle",
  "center": { "x": 50, "y": 50, "z": 10 },
  "normal": { "x": 0, "y": 0, "z": 1 },
  "polyline_id": "poly_wire_face_4_1",
  "wire_id": "wire_face_4_1",
  "face_id": "face_4",
  "is_outer": false,
  "area": 706.86,
  "perimeter": 94.25,
  "parameters": {
    "diameter": 30,
    "length": null,
    "width": null,
    "across_flats": null
  }
}
```

---

## 五、summary（全模型专用）

| 字段 | 类型 | 含义 |
|------|------|------|
| `volume` | number \| null | 体积 mm³ |
| `surface_area` | number \| null | 总表面积 mm² |
| `bbox` | object | 包围盒 |
| `face_count` | int | 面数量 |
| `edge_count` | int | 边数量 |
| `solid_count` | int | 实体数量 |

---

## 六、faces（面）

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 如 `face_0`（与 `face_id` 参数一致） |
| `surface_type` | string | `plane` / `cylinder` / `cone` / `sphere` / `torus` / `other` |
| `area` | number | 面积 mm² |
| `normal` | `{x,y,z}` \| null | 平面法向 |
| `axis` | `{x,y,z}` \| null | 圆柱/圆锥轴线 |
| `center` | `{x,y,z}` \| null | 面心或轴上一点 |
| `radius` | number \| null | 圆柱/球半径 mm |
| `outer_wire_id` | string \| null | 外环 wire id |
| `inner_wire_ids` | string[] | 内环 wire id 列表 |
| `side` | string \| null | 内/外表面：`outer` 外 / `inner` 内 / `unknown` |
| `side_score` | number \| null | 内外判定得分 = 外法向·(面上点-所属实体质心)，>0 外、<0 内 |

**前端选面提示**：可先展示 GLB，用户点击面后映射到 `face_{index}`，再调用 `/analyze/face` 或 `/analyze/face_spread`。

---

## 七、wires（线环）

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 如 `wire_face_4_0` |
| `face_id` | string | 所属面 |
| `is_outer` | bool | 外环 / 内环 |
| `length` | number | 周长 mm |
| `area` | number \| null | 投影面积 mm² |
| `polyline_id` | string | 对应折线 id |
| `contour_id` | string \| null | 对应轮廓 id |
| `contour_type` | string \| null | 轮廓类型 |

---

## 八、holes（孔 / 凸台）

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 如 `hole_contour_1` |
| `kind` | string | `through` 通孔 / `blind` 盲孔 / `pocket` 型腔 / `boss` 凸台 / `circle` / `slot` 等 |
| `contour_type` | string \| null | 口部轮廓类型，与 contours 一致 |
| `direction` | string \| null | `recess` 凹陷(孔/型腔) / `protrusion` 凸出(凸台)；未启用 3D 时为 null |
| `through` | bool \| null | 凹陷是否贯通（通孔 true、盲孔/型腔 false） |
| `center` | `{x,y,z}` | 孔心 |
| `axis` | `{x,y,z}` | 孔轴线（口部外法向） |
| `diameter` | number \| null | 直径 mm |
| `depth` | number \| null | 凹陷深度或凸台高度 mm（3D 识别后填充；未启用/纯 Shell 时 null） |
| `face_id` | string \| null | 关联面 |
| `wire_id` | string \| null | 关联 wire |
| `parameters` | object \| null | 同 contours.parameters |

**说明**：
- `kind` / `direction` / `through` / `depth` 由 3D 深度识别填充（`enable_depth=true`，默认开启，需模型含实体 Solid）。
- 未启用 3D 或纯 Shell 模型时回退 2D：`kind` 取轮廓类型，`depth/direction/through` 为 null。
- 圆柱面合成孔需 `include_cylinder_holes: true`。

### holes_by_type 分组 key

`feature_groups.holes_by_type` 仍按 `contour_type`（circle/slot/rectangle/hexagon）分组，
便于路径规划；具体加工类型见每条孔的 `kind`/`direction`/`through`。

---

## 九、polylines（折线 — 画线）

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 如 `poly_wire_face_4_0` |
| `closed` | bool | 是否闭合 |
| `points` | array | `[{x,y,z}, ...]` |

---

## 十、reference_points（参考点）

| kind | 含义 |
|------|------|
| `face_center` | 面心 |
| `contour_center` | 轮廓中心 |
| `hole_center` | 孔心 |
| `datum` | 基准点（全模型 analyze 才有 bbox 基准） |

---

## 十一、PathPlanResult（刀路）

| 字段 | 类型 | 含义 |
|------|------|------|
| `schema_version` | string | `"1.0"` |
| `strategy` | string | `outer_contour` / `hole_circle` / `zigzag` / `combined` |
| `segments` | array | 刀路段 |
| `total_length` | number | 总长度 mm |
| `estimated_time_s` | number \| null | 估算时间 s |

### segments[]

| 字段 | 含义 |
|------|------|
| `id` | 段 id |
| `strategy` | 本段策略 |
| `feed` | 进给 mm/min |
| `points` | 刀路点列 |

单面刀路使用 `model_bbox` 作为高度参考（通过 `face_analyze_to_path_payload` 适配）。

---

## 十二、数据关系简图（单面）

```
用户选中 face_3
    │
    ▼
POST /analyze/face  (face_id=face_3)
    │
    ├── face                    面元数据
    ├── wires[]                 该面所有环
    │      ├── is_outer=true  ──→ feature_groups.wires_by_role.outer
    │      └── is_outer=false ──→ feature_groups.wires_by_role.inner
    ├── contours[]              轮廓分类结果
    │      └── contour_type   ──→ feature_groups.contours_by_type.*
    ├── holes[]                 孔特征
    │      └── kind/contour_type ──→ feature_groups.holes_by_type.*
    ├── polylines[]             通过 polyline_id 画线
    └── outer_contours[]        该面外轮廓 id → 外轮廓刀路
```

---

## 十三、100×60×20 长方体单面参考（face_0 顶面）

| 项目 | 期望值 |
|------|--------|
| `target_face_id` | `face_0` |
| `contours` | ≥ 1 |
| `feature_groups.contours_by_type.outer` | 1 条 |
| `holes` | `[]` |
| `outer_contours` | 1 个 id |

---

## 十四、带孔板单面参考

对含圆孔顶面（有 `inner_wire_ids` 的平面）：

| 项目 | 期望值 |
|------|--------|
| `holes` | ≥ 1 |
| `feature_groups.holes_by_type.circle` | ≥ 1 |
| `feature_groups.contours_by_type.circle` | ≥ 1 |
| `feature_groups.wires_by_role.inner` | ≥ 1 |

---

完整 API 调用见 [CAD_API_EXTERNAL.md](./CAD_API_EXTERNAL.md)，算法说明见 [CAD_ALGORITHM.md](./CAD_ALGORITHM.md)。
