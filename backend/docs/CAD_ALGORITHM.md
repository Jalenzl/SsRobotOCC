# pythonOCC CAD 特征识别与刀路规划算法说明

## 1. 系统架构

```
前端 (Vue/React/Three.js)
    │  multipart/form-data 或 fetch
    ▼
FastAPI  `/api/v1/cad/*`
    │
    ├─ routers/cad.py          HTTP 契约、参数校验
    ├─ services/cad_service.py 业务流程编排
    ├─ models/cad.py             Pydantic JSON Schema（前后端共用）
    └─ occ/                      pythonOCC 内核（可选依赖）
         ├─ loader.py            STEP → TopoDS_Shape
         ├─ discretize.py        Edge/Wire → 折线
         ├─ geometry_utils.py    面包围盒、曲面分类、加工坐标系
         ├─ topology.py          边-面/面-实体邻接图、实体质心、外法向（3D 基础设施）
         ├─ features/extractor.py        点/面/线/孔识别 + 内外扩散编排
         ├─ features/contour_classifier.py 2D 轮廓分类
         ├─ features/face_side.py        内/外表面判定（外法向正负）
         ├─ features/feature3d.py        3D 深度识别（通孔/盲孔/型腔/凸台）
         └─ path/planner.py      2.5D 刀路生成
```

设计原则：

- **分层**：路由不直接调用 OCC，便于单测与替换内核（如 OCP）。
- **JSON 友好**：所有几何输出为 `Point3D` / `Polyline3D`，前端可直接画线、标点。
- **可选依赖**：未安装 pythonOCC 时 `/cad/status` 仍可用，分析接口返回 `501`。

---

## 2. STEP 载入与拓扑规范化

**输入**：`.stp` / `.step` 字节流。

**步骤**：

1. `STEPControl_Reader.ReadFile` + `TransferRoots` → `OneShape()`。
2. 拓扑规范化 `_normalize_shape`：
   - 优先取第一个 `TopAbs_SOLID`；
   - 否则取第一个 `TopAbs_SHELL`；
   - 否则保留复合体。

**输出**：`TopoDS_Shape`，作为后续 `TopExp_Explorer` 的根。

---

## 3. 特征识别算法

### 3.0 单面提取（推荐 API 路径）

**入口**：`extract_face_features(shape, options, face_id=...)`

```
1. 解析 face_id → 整数索引（face_12 或 12）
2. TopExp_Explorer(shape, FACE) 遍历，取第 index 个面
3. 调用 _extract_face_payload() 只处理该面
4. _build_feature_groups() 按 contour_type / hole kind / wire role 分组
5. 返回 CadFaceAnalyzeResult（含 feature_groups）
```

与全模型 `extract_all_features()` 共用同一套 `_extract_face_payload()`，保证单面与全模型结果一致。

**face_id 与 OCC 遍历顺序一致**，与 per-face mesh 命名 `face_{idx}` 对齐。

### 3.1 面（Face）遍历

```
FOR face IN TopExp_Explorer(shape, TopAbs_FACE):
    adaptor = BRepAdaptor_Surface(face)
    分类: Plane | Cylinder | Cone | Sphere | Torus | other
    area = brepgprop.SurfaceProperties(face)
    wires = face 上所有 TopAbs_WIRE
```

记录字段：`surface_type`, `area`, `normal`/`axis`/`radius`, `outer_wire_id`, `inner_wire_ids`。

### 3.2 边离散与外轮廓线（Wire / Polyline）

对每条 `Wire`：

1. 用 **`BRepTools_WireExplorer`** 按 wire 拓扑顺序遍历边，再逐边离散（`GCPnts_QuasiUniformDeflection`）。
2. 边端点连接容差与 `linear_deflection` 自适应；**保留 wire 上全部边段**（不再只取最长链），避免圆孔折线被截成弧。
3. 若容差内拼接失败，仍顺序保留各边几何，避免丢边；必要时回退贪心拼链并串联所有分段。
4. 首尾相连判断 `closed`（容差与离散精度同量级）。
5. 平面面：投影到**面外法向**二维，用鞋带公式算面积；非平面面：PCA 投影 + 非平面度检测。

**平面外轮廓判定**（同一平面 Face）：

- 所有 **closed** _wire 按 `area` 降序排序；
- **面积最大** 者为 `outer_wire`（外轮廓线）；
- 其余 closed 内环 → 孔候选（见 3.3）。

全局外轮廓：在所有平面面中，取面积最大的 `outer_wire_id` 写入 `outer_contours[]`。

全局外轮廓：单面模式下在该面 `contours` 中取最大外轮廓；全模型模式在所有面中取最大。

### 3.3 轮廓分类（contour_classifier）

平面与非平面 wire 均进入 `classify_wire_contour()`：

| contour_type | 判定依据 |
|--------------|----------|
| `outer` | 外环 wire |
| `circle` | 圆度 ≥ 0.88，或兜底圆度 ≥ 0.82（近圆小孔） |
| `slot` | 长宽比 ≥ 1.75 且非圆 |
| `hexagon` | 约 6 拐角 + 边长均匀 |
| `rectangle` | 4–5 直角拐角，或 OBB 长宽比 ≤ 1.6 且圆度 ∈ [0.55, 0.88)（含圆角矩形） |
| `unknown` | 非平面度超阈值或无法分类 |

非平面面使用 `prefer_pca_plane=true`，投影基由 PCA 拟合（仅用于 2D 形状分类）。

**轮廓法向 `contours[].normal`**：始终为 **宿主面外法向**（`topology.face_point_and_outward_normal`），
不再使用 PCA/加工平面法向，避免自由曲面/邻接面上圆弧法向「贴着平面」。
2D 分类仍用 PCA 投影，与 API 输出的法向字段分离。

### 3.4 孔（Hole）识别

**路径 A — 平面内环（铣削孔、沉头孔口部）**

- 条件：`inner_wire`、closed、`area > 0`；
- 等效直径：`d = 2 * sqrt(area / π)`；
- 过滤：`hole_diameter_min ≤ d ≤ hole_diameter_max`；
- 孔心：折线顶点质心；孔轴：平面法向 `normal`。

**路径 B — 圆柱面（钻孔、镗孔壁）**

- 条件：`surface_type == cylinder` 且 `include_cylinder_holes=true`（默认关闭，避免外圆柱/圆角误判）；
- `diameter = 2 * radius`，同上直径过滤；
- 孔心：圆柱轴上一点；孔轴：圆柱轴线方向。

**去重**：孔心坐标按 0.01mm 网格量化，距离 < 1mm 视为同一孔。

**类型**：由 3D 深度识别填充（见 3.9）。圆形凹陷 → `through`（通孔）/ `blind`（盲孔）；
非圆凹陷 → `pocket`（型腔，同时登记到 `pockets[]`）；凸出特征 → `boss`（凸台）。
每个孔附带 `direction`（recess 凹 / protrusion 凸）、`through`（是否贯通）、`depth`（深度/高度，mm）。
若模型无实体（纯 Shell）或 `enable_depth=false`，则回退为 2D，`kind` 取轮廓类型、`depth=null`。

### 3.5 参考点（Reference Points）

| kind | 来源 |
|------|------|
| `face_center` | 平面/圆柱轴心 |
| `hole_center` | 孔心 |
| `datum` | 包围盒中心、min、max |
| `contour_vertex` | （预留）轮廓顶点 |

### 3.6 口袋（Pocket）

非圆凹陷特征（矩形/槽等内环）在 3D 识别为 `recess` 时，除写入 `holes[]`（kind=`pocket`）外，
同时登记到 `pockets[]`，附带 `depth`、`through`、`center`、`axis`、`contour_type`、口部 `parameters`。
`bottom_face_id` 预留（后续可由壁面→底面邻接链补全）。

### 3.7 内 / 外表面判定（features/face_side.py）

需求：前端选中一个面后，需要知道它是「外表面」还是「内表面」，从而扩散到整个装配体的同侧表面。

判定公式（外法向正负）：

```
score = n̂ · (P − C)
```

- `P`：面上一点（参数域中点，世界坐标）；
- `n̂`：该点处指向实体外部的单位**外法向**（几何法向按 `TopAbs_REVERSED` 取反）；
- `C`：该面**所属实体**的体积质心（装配体逐实体计算，`topology.build_face_solid_map` 定位 owning solid）。

| 条件 | 判定 | 典型 |
|------|------|------|
| `score ≥ 0` | `outer` 外表面 | 顶面、侧面、凸台外壁 |
| `score < 0` | `inner` 内表面 | 孔壁、镗孔、内腔、装配贴合面 |
| 法向不可定义 | `unknown` | 退化面 |

- 对自由曲面（B-spline）同样适用（只依赖法向与质心，不依赖解析曲面类型）。
- 极端凹形（深型腔底面）为近似启发式，`score` 越接近 0 越不确定，写入 `face.side_score` 供阈值过滤。

### 3.8 3D 深度识别（features/feature3d.py）

在 2.5D 口部轮廓基础上补齐**深度方向**语义，核心是 `BRepClass3d_SolidClassifier` 射线步进
（点在实体内/外判定）。对每条内环口部（中心 `C`、外法向 `N`）：

1. **方向判定（两侧探针）**：在 `C ± εN` 取点判定内外：
   - 外侧 `C+εN` 在实体内（IN）→ `protrusion` 凸台；
   - 内侧 `C−εN` 在实体外（OUT，空腔）→ `recess` 凹陷。
2. **通孔 / 盲孔（轴向步进）**：沿 `−N` 向材料内步进；命中材料（IN）即孔底 → `blind`；
   一直为空腔（OUT）直到穿出包围盒 → `through`。
3. **深度量化（壁面轴向跨度）**：取与口部 wire 共享边的侧壁面顶点，投影到轴 `N`，
   相对口部下方跨度 = 凹陷深度，上方跨度 = 凸台高度。壁面几何比步进更精确，作为深度首选；
   步进只负责类型判定与兜底。

```
通孔 through ──→ depth = 壁面贯穿厚度, through=true,  direction=recess
盲孔 blind  ──→ depth = 口部到孔底,    through=false, direction=recess
型腔 pocket ──→ 非圆凹陷, 另登记 pockets[]
凸台 boss   ──→ depth = 凸台高度,      direction=protrusion
```

- 不依赖曲面是平面还是自由曲面 → **自由曲面上的孔/凸台同样可识别**。
- 装配体：按口部 host 面的 owning solid 建立分类器并缓存（逐实体）。
- 健壮性：任何 3D 步骤异常都回退到 2D（`depth=null`），不影响轮廓/孔的 2D 输出。

### 3.9 内/外表面扩散（extract_face_spread_features）

入口：`extract_face_spread_features(shape, options, face_id=...)` → `POST /cad/analyze/face_spread`。

```
1. 解析种子面 face_id → 用 3.7 判定其 side（outer / inner）
2. 扩散：收集整个装配体中所有同侧的面（FaceSideClassifier.faces_on_side）
3. 逐面复用 _extract_face_payload（与单面/全模型同一套算法）
4. 叠加 3.8 的 3D 深度识别，聚合孔/型腔/凸台
5. 返回 CadFaceSpreadResult（含 side、solid_count、face_ids、holes、pockets…）
```

选中外表面的一个面 → 得到整机外蒙皮的全部孔/凸台；选中内表面 → 得到全部内腔/孔壁特征。

### 3.10 加工坐标系（Work Plane）

| 模式 | 法向 |
|------|------|
| `xy` | (0,0,1) |
| `yz` | (1,0,0) |
| `xz` | (0,1,0) |
| `auto` | 包围盒最短边方向（典型装夹面） |

---

## 4. 刀路规划算法（2.5D）

**输入**：

- 全模型：`CadAnalyzeResult` + `PathPlanOptions`
- 单面：`CadFaceAnalyzeResult` → `face_analyze_to_path_payload()` 适配后 + `PathPlanOptions`

**公共参数**：

- `safe_z = bbox.zmax + clearance_z`（快移平面）
- `z_cut`：默认取包围盒中间高度（可改为多层切片循环）

### 4.1 外轮廓 (`outer_contour`)

1. 取 `outer_contours[0]` 对应折线；
2. 先在 `safe_z` 走一圈闭合边界；
3. 下到 `z_cut` 再切一圈；
4. 进给 `feed_cut`。

### 4.2 孔加工 (`hole_circle`)

对每个孔：

1. 快移到孔心 `(x,y,safe_z)`；
2. 以 `diameter/2 - tool_offset` 为半径生成 `n` 点圆（`n ∝ 周长/step_over`）；
3. 在 `z_cut` 平面走圆；
4. 抬刀回 `safe_z`。

### 4.3 行切 (`zigzag`)

在包围盒 `[xmin,xmax]×[ymin,ymax]`：

- 行距 `step_over`；
- 奇偶行反向，形成弓字形；
- 每行：快移 → 下刀 → 切削 → 抬刀。

### 4.4 组合策略 (`combined`)

依次生成：外轮廓 → 各孔圆 → 行切；`total_length` 为各段折线长度之和；`estimated_time_s ≈ total_length / feed_cut * 60`。

---

## 5. 外部前端调用

本仓库不含前端实现。推荐流程：

1. `POST /stp/convert` → GLB 显示
2. 用户选面 → `face_id`
3. 选其一：
   - `POST /cad/analyze/face` → 仅该面的 `feature_groups`；
   - `POST /cad/analyze/face_spread` → 判定内/外表面并扩散到整个装配体同侧表面，
     返回带 `depth`/`direction`/`through` 的孔/型腔/凸台（`CadFaceSpreadResult`）。
4. `POST /cad/path/generate/face` 或 `/analyze/face_and_path` → 刀路

完整契约与示例见 **[CAD_API_EXTERNAL.md](./CAD_API_EXTERNAL.md)**，字段说明见 **[CAD_RESPONSE_FIELDS.md](./CAD_RESPONSE_FIELDS.md)**。

---

## 6. 安装 pythonOCC

Windows 推荐使用 Conda：

```bash
conda create -n occ python=3.10
conda activate occ
conda install -c conda-forge pythonocc-core
pip install -r backend/requirements.txt
```

可选文件：`backend/requirements-occ.txt`（说明性依赖）。

---

## 7. 已实现的 3D 能力与后续扩展

已实现（本版本）：

- **内/外表面判定**：外法向正负，逐实体质心（3.7）。
- **3D 深度识别**：通孔/盲孔/型腔/凸台 + 深度，`BRepClass3d_SolidClassifier` 射线步进（3.8）。
- **内外扩散**：选面 → 同侧表面全装配体扩散（3.9，`/cad/analyze/face_spread`）。
- **自由曲面**：深度/凸台识别不依赖解析曲面类型，B-spline 面上的孔同样可识别。
- **装配体**：`Compound` 多 `Solid` 逐实体质心与分类器，面全局索引保持一致。

后续扩展：

1. **多层粗精加工**：对 `z` 从 `zmax` 到 `zmin` 按 `ap` 步距切片，重复 4.1–4.3。
2. **轮廓偏置**：外轮廓 inward offset 使用 `BRepOffsetAPI_MakeOffset`。
3. **孔序优化**：孔心 TSP（最近邻 / 2-opt）减少空行程。
4. **沉头/台阶孔**：同轴多级圆环 + 不同深度 → `counterbore`。
5. **型腔底面**：壁面 → 底面邻接链补全 `pocket.bottom_face_id`。
6. **零件级标签**：为每个特征附带 `part_id`（owning solid 索引），便于装配体分零件加工。
