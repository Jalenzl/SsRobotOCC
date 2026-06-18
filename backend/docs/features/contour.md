# 特征识别（外/内轮廓 + 法向）— 方案与层次

> 目标：在已重构完成的 CAD 视图上，把 **每个面（face）的外轮廓 / 内轮廓** 抽
> 象为有序 3D 折线 + 折线外法向，前端可直接用 Three.js `Line` / `LineSegments`
> 叠加在原 GLB 之上进行高亮渲染；点击「测试特征提取」按钮即可触发。

## 1. 总体层次

把特征识别彻底放在与 mesh 提取、GLB 导出同级别的「OCC 算法模块」里，HTTP
只负责入参 / 出参 / 缓存；OCC 计算全部在子进程 / 隔离函数内完成。

```
backend/app/
├── occ/                       # ── 纯算法（不依赖 FastAPI）
│   ├── loader.py              # STEP 字节 → TopoDS_Shape（已有）
│   ├── step_hierarchy.py      # 装配体树（已有）
│   ├── topology.py            # 边/面邻接、外法向采样（已有）
│   ├── geometry_utils.py      # 面信息、wire、work-plane 法向（已有）
│   ├── discretize.py          # edge / wire → 有序 3D 折线（已有）
│   ├── mesh_export.py         # GLB 网格导出（已有）
│   ├── contour.py             # ── NEW：face → wires/polylines/contours/holes
│   └── face_analyzer.py       # ── NEW：target face_id → 完整 CadFaceAnalyzeResult
├── services/
│   ├── cad_cache.py           # STEP 缓存（已有，model_id 复用）
│   └── feature_service.py     # ── NEW：编排 face_analyzer，组装响应
└── routers/
    ├── cad.py                 # 上传 / model_id（已有）
    └── feature.py             # ── NEW：POST /api/v1/cad/analyze/face_spread
```

**设计原则**
1. **`app/occ/` 只做算法**，不感知 HTTP / Pydantic；返回 plain dataclass 或
   `dict`，便于单元测试与子进程复用。
2. **`app/services/feature_service.py` 做管道编排**：从 `model_id` 读取
   STEP → 调用 `feature_pipeline` → 组装响应字段（`bbox` / `workplane` /
   `contours`），对应前端需要的坐标系。
3. **`app/routers/feature.py` 极薄**：参数校验、状态码、统一异常。
4. **前端用 `model_id` 复用上传**：`/api/v1/cad/upload` 已经返回 `model_id`，
   特征接口只需 POST `model_id` 即可，不必再传 STEP 字节；点击「测试特征
   提取」按钮时从 store 取当前 model_id 即可触发。

## 2. 坐标系约定（与前端 GLB 渲染一致）

- 后端输出的「轮廓点」和「折线法向」都在 **世界坐标**（即 GLB 已应用的
  装配体 matrix 之外，还要再叠 node matrix）。但因为「轮廓是绑定在 face 上
  的几何」，更稳妥的做法是：**后端返回 face 的局部 3D 折线 + face 法向**，
  前端在用 GLB 渲染时根据 `part_id` × `matrix` 自己把顶点搬到世界。
- 我们采用上述稳妥做法，字段约定如下：
  - `contour.points` = `[[x,y,z], ...]`，3D 世界坐标（与 GLB 中 mesh 顶点
    完全一致）—— 由 face 的 `TopLoc_Location`（face.Location）+ 节点
    `location_matrix` 一起叠出。如果 face 没有 `TopLoc`，与 mesh 同位置。
  - `contour.normal` = `[nx, ny, nz]`，3D 世界坐标，单向量（取线段中点
    处的 face 外法向 + 折线切向叉乘验证后取朝向一致的一面）。
  - `contour.kind` = `outer` | `inner` | `inner_nested`：分类结果，方便
    前端用不同颜色 / 线宽区分。
  - `contour.face_index` = 全局 face 序号（与 `iterate_faces` 同序）。
  - `contour.part_id` = 该 face 所属 part 节点 id（`step_hierarchy` 中
    已有 `part_id`）。

## 3. 算法流水线

```text
STEP bytes
   └─► (loader)              TopoDS_Shape
         └─► (step_hierarchy)  HierarchyNode + parts[ {part_id, matrix, shape} ]
               └─► for each part:
                     ├─ work plane: 最小包围盒厚度方向 / 指定 normal
                     ├─ collect planar faces ( |n·wp| > 0.999 )
                     │     └─ for each planar face:
                     │           ├─ wires = face.wires
                     │           ├─ for each wire:
                     │           │     ├─ polyline = wire_to_polyline(...)
                     │           │     ├─ normal = face_outward_normal(face)
                     │           │     └─ classify(outer/inner, workplane)
                     │           └─ emit ContourRecord
                     └─ (future) collect cylindrical/conical features
```

### 3.1 「外/内环」分类

对每一个工作平面的 face：
1. 拿到 face 上的所有 wire（`face_wires(face)`）。
2. 把每条 wire 投到工作平面（参见 §3.2）。
3. 沿工作平面的 +U 方向构造带符号面积；正 = 外（outer），负 = 内（inner）。
   - 与 OCCT 内部「外 wire 顺时针、内 wire 逆时针」约定对齐到统一符号：
     **面积绝对值最大的那一条 = outer**，其余 = inner。
4. 若多个 outer 并存（极少见的多 island），按面积绝对值从大到小排序，第一个
   标记 `outer`，其余 `inner_nested`。

### 3.2 工作平面

工作平面 = 顶面/底面（最薄方向 + 1mm 容差内的平面），决定轮廓投影坐标系。
投影：
- workplane normal = `(0,0,1)`（auto 默认：z 轴 = 厚度方向）
- workplane origin = 整个 shape 的 bbox min 的 (xmin, ymin, 0)
- workplane U = `(1,0,0)`，V = `(0,1,0)`
- 任意 3D 点 P 投到 workplane：`(P.x − origin.x, P.y − origin.y)`

> 当且仅当多个 face 的法向都接近 workplane normal 时（即顶面 + 底面 + 任何
> 法向指 z 的台阶面），它们都视为「加工平面」并都给出 outer/inner 轮廓。

### 3.3 法向

折线法向 = 该 face 在轮廓中点处的 **外法向**（`face_outward_normal`），
**不是** 折线切向 × 路径方向。理由：
- 折线切向 × 路径方向 仅当外环（CCW）才有意义；多个 inner 折线方向不一，
  容易误判。
- 整面外法向是唯一确定的，前端可以直接画箭头（`ArrowHelper`）展示「这条
  轮廓属于哪个方向的面」，对应你图里箭头方向（朝外 / 朝内）。

为了兼顾「轮廓上每段的方向性」（如内孔的径向方向），**额外**给每条折线
附一个 `tangent_hint`：取折线中点 + 中点处 face 外法向，组合成箭头；
前端可以一键画出。

## 4. 响应 Schema（与前端对齐）

```jsonc
{
  "schema": "robotlaser.feature.contours/v1",
  "model_id": "76fbe152fff2b5d2…",
  "unit": "mm",
  "workplane": {
    "origin": [x, y, z],   // 3D 世界坐标
    "u":      [ux, uy, uz],
    "v":      [vx, vy, vz],
    "normal": [nx, ny, nz]
  },
  "contours": [
    {
      "id": "c_0_outer",
      "part_id": "p1_s0",
      "face_index": 5,
      "kind": "outer",                 // outer | inner | inner_nested
      "closed": true,
      "points": [[x,y,z], ...],
      "normal": [nx, ny, nz],         // 面外法向，世界坐标
      "tangent_hint": {                // 中点 + 折线切向（仅 inner 必备）
        "point":  [x, y, z],
        "tangent":[tx, ty, tz]
      },
      "length": 312.4,
      "area":   5401.0
    }
  ],
  "stats": {
    "faces": 12, "outer": 4, "inner": 7, "inner_nested": 1, "elapsed_ms": 38
  }
}
```

前端 `LineSegments` / `Line` 渲染：
- `outer`   → 绿色 `LineLoop`，线宽 3
- `inner`   → 红色 `LineLoop`，线宽 2
- `inner_nested` → 橙色 `LineLoop`
- 折线外法向 → `ArrowHelper(normal, mid_point, 8, color)`，颜色与折线同

## 5. 接口

`POST /api/v1/feature/contours`
- body: `{ "model_id": "…", "workplane_mode": "auto|xy|yz|xz", "tolerance_mm": 0.1 }`
- response: 见 §4
- 错误：502（pythonOCC 不可用）/ 422（解析失败）/ 404（model_id 找不到）

## 6. 测试

- `tests/test_contour.py`：用 `make_plate_with_hole_shape` 验证
  1 个 outer + 1 个 inner，外环周长 ≈ 4 × 边长，内环周长 ≈ 2πr。
- `tests/test_contour_assembly.py`：双实体装配，每实体至少 1 outer。
- 集成测试：HTTP `POST /api/v1/feature/contours` + 已有 `plate_with_hole`，
  断言响应 schema 完整，轮廓闭合、`kind` 数量正确。

## 7. 风险 / 后续

- **多面共外环**（复杂 STEP 偶发）：目前按 OCCT 给的 wire 列表直接取，
  未做几何合并。下一版加「同平面、同 part、相邻 wire 合并」。
- **自由曲面上的轮廓**：当前只处理平面 face；Bezier 面 / NURBS 面输出
  `kind=unsupported`，前端用灰色虚线占位，便于排错。
- **法向翻转**：若用户传入 `workplane_mode` 显式指定 normal，将用传入
  normal 与 face 外法向做点积正负比较 → 法向朝上的标记 `side=top`、
  朝下 `side=bottom`，便于前端做俯视图 / 仰视图切换。
