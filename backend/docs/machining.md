# CAM 路径规划（多孔洞刀路生成）— 文档

> 本文档描述 **特征识别完成之后** 的 CAM 路径规划子系统。 它把每个孔洞
> / 外轮廓转换为机器人可执行的"刀路"：`MachiningPath` 包含若干 `CAMLine`，
> 每条 `CAMLine` 描述机器人从 A 点移动到 B 点的一段加工动作。

## 1. 总体层次

```
backend/app/
├── models/
│   └── feature.py            # CAMLine / MachiningPath / MachiningGroup / MachiningResult
├── services/
│   ├── feature_service.py    # 特征识别 (analyze_face_spread / analyze_part_spread)
│   └── machining_service.py  # ★ 路径规划 + 段分类 + 引线 + 跨路径空驶
└── routers/
    └── feature.py            # HTTP 端点（/machining/paths、/machining/paths/multi）
```

数据流向：

```
STEP 字节 (model_id 缓存)
   │
   ▼  feature_service.analyze_face_spread / analyze_part_spread
   │
feature_result: {
    polylines: [{ id, points: [[x,y,z]...] }, ...],
    wires:     [{ id, polyline_id, contour_id, contour_type, ... }, ...],
    contours:  [{ id, contour_type, polyline_id, is_outer, ... }, ...],
    holes:     [{ id, kind, contour_type, wire_id, center, axis, ... }, ...],
}
   │
   ▼  machining_service.generate_machining_paths_multi(feature_result, hole_ids, ...)
   │
MachiningResult: {
    schema_version: "2.0",
    model_id,
    machining_groups: [{
        id, name,
        inner_paths: [MachiningPath, ...],   # ← 按点击顺序
        outer_paths: [MachiningPath, ...],   # ← 可选
        path_order:   [path_id, ...],        # ← 扁平加工序列
        transition_lines: [CAMLine, ...],    # ← 跨孔空驶
    }],
    total_path_count,
    total_line_count,
}
```

## 2. 数据模型（models/feature.py）

### 2.1 `CAMLine` — 一段加工动作

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 全局唯一 ID（如 `path_xxx_line_3`） |
| `line_type` | `Literal` | `machining` / `lead` / `cut_in` / `cut_out` / `fast` / `idle` / `location` |
| `path_type` | `Literal` | `outer` / `inner` |
| `inner_type` | `Literal\|None` | `circle` / `slot` / `rectangle` / `hexagon` / `irregular` |
| `out_type` | `Literal\|None` | `long_line` / `shorter_line` / `shortest_line` / `big_arc` / `small_arc` / `three_d_corner` / `point` |
| `start_point` / `end_point` | `Point3D` | 段起止 3D 点（世界坐标，与 GLB 顶点一致） |
| `normal` | `Vector3D\|None` | 该点法向（用于逆解时构建 Ax2） |
| `velocity` | `float` | 该段速度（mm/s），已按 out_type 衰减 |
| `power` / `duty` | `int` | 激光功率 / 占空比（idle 时为 0） |
| `is_clockwise` | `bool` | 切割方向（CW / CCW） |
| `order_index` | `int` | 段在所属 path 内的顺序（1-based） |
| `robot_joints` | `list[float]` | **逆解后填入**的 6 轴关节角，初始为 `[]` |

### 2.2 `MachiningPath` — 一条封闭加工单元

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | path 唯一 ID |
| `name` | `str` | 人类可读名（如 `hole_circle`） |
| `path_type` | `Literal` | `outer` / `inner` |
| `contour_id` / `source_hole_id` / `source_contour_id` | `str\|None` | 关联到特征识别结果中的 ID（前端高亮用） |
| `cam_lines` | `list[CAMLine]` | 加工段序列（**加工顺序**） |
| `lead_line` / `lead_out_line` | `CAMLine\|None` | 引线 / 退刀线（`line_type="lead"`） |
| `idle_lines` | `list[CAMLine]` | path 内段间过渡（目前未用，留作扩展） |
| `order_index` | `int` | **0-based** 在所属 inner_paths / outer_paths 列表中的位置（=前端点击顺序） |
| `thickness` | `float` | 板材厚度（mm） |
| `normal_reversed` / `is_removed` | `bool` | 状态标志 |

### 2.3 `MachiningGroup` — 一个加工组

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | group 唯一 ID |
| `name` | `str` | 描述名（如 `MultiHole[3]+Outer[1]`） |
| `inner_paths` | `list[MachiningPath]` | **按前端点击顺序排列** |
| `outer_paths` | `list[MachiningPath]` | 可选，按特征识别顺序 |
| `process_face_ids` | `list[str]` | 关联 face ID |
| `path_order` | `list[str]` | **扁平加工序列**（`inner_paths` + `outer_paths` 的 ID 列表） |
| `transition_lines` | `list[CAMLine]` | **跨路径空驶线**（每个 `line_type="idle"`，数量 = `len(inner_paths) - 1`） |

### 2.4 `MachiningResult` — 顶层响应

```python
class MachiningResult(BaseModel):
    schema_version: str = "2.0"
    unit: str = "mm"
    model_id: str
    feature_result: Any | None = None        # 原始特征识别结果（冗余以便前端单次拿到全部数据）
    machining_groups: list[MachiningGroup]
    total_path_count: int
    total_line_count: int                    # = 所有 cam_lines + transition_lines
```

## 3. 多孔洞选择 + 点击顺序

### 3.1 业务流程

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. 用户在 3D viewer 中点击 / 框选 / Shift+Click 多选孔洞             │
│     前端维护一个**有序**列表: hole_selection: list[str]               │
│     （第 0 个就是用户最先点击的）                                      │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2. 前端调用 POST /api/v1/cad/machining/paths/multi                  │
│     一次性提交 model_id + face_id + hole_ids（按点击顺序）            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  3. 后端 machining_service.generate_machining_paths_multi()          │
│     - 调 feature_service.analyze_face_spread 拿全量特征              │
│     - 按 hole_ids 顺序筛选 holes                                       │
│     - 为每个 hole 调 hole_to_machining_path() 拿到 MachiningPath      │
│     - 为相邻 hole 之间生成 1 条 transition_line (idle)               │
│     - 把所有 path.id 串成 path_order                                  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  4. 前端按 inner_paths[i].order_index 渲染每条 path                   │
│     用 path_order 驱动 3D 仿真播放（沿 transition_lines 跳转）        │
│     之后调用 /ikfast/m20ia-35m/inverse 批量算关节角                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 与前端的**数据配合**约定

| 步骤 | 前端做什么 | 后端返回什么 |
|------|------------|--------------|
| 1. 多选 | 维护 `hole_selection: ["hole_3", "hole_1", "hole_7"]` | — |
| 2. 触发 | 按钮点击 → POST `/machining/paths/multi` | — |
| 3. 渲染 | 按 `inner_paths[i].order_index` 顺序显示 | 返回 `MachiningResult.machining_groups[0]` |
| 4. 高亮 | 用 `path.source_hole_id` 反向高亮模型上的孔 | — |
| 5. 仿真播放 | 沿 `path_order` 顺序串接各 `MachiningPath` | — |
| 6. 跳转 | 在 `transition_lines[i]` 期间画"虚线"（idle/快进） | — |

> **关键约定**：`inner_paths` 列表的**顺序**就是加工顺序。 前端无需解析
> `order_index` 字段，但**可以用**该字段做调试 / UI 排序。

### 3.3 前端调用示例

```javascript
// 假设 hole_selection 是用户在 3D viewer 中按点击顺序维护的列表
const holeIds = ["hole_3", "hole_1", "hole_7"];

const response = await fetch("/api/v1/cad/machining/paths/multi", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model_id:  currentModelId,
    face_id:   selectedFaceId,        // 例如 "face_12"
    hole_ids:  holeIds,               // ← 关键：保留用户点击顺序
    include_outer: true,              // 可选：把外轮廓也带进来
    apply_craft_params: true,
    idle_velocity: 350,               // mm/s，跨孔空驶速度
  }),
});

const result = await response.json();
const group = result.machining_groups[0];

// 加工顺序：先内后外（如果 include_outer=true）
console.log("path_order:", group.path_order);

// 渲染每条 path
group.inner_paths.forEach((path, idx) => {
  console.log(`[${path.order_index}] ${path.name}: ${path.cam_lines.length} 段`);
  // 例如：[0] hole_circle: 32 段
  //       [1] hole_slot: 24 段
  //       [2] hole_hexagon: 28 段
});

// 跨孔空驶
group.transition_lines.forEach((line, idx) => {
  console.log(`空驶 ${idx}: ${line.start_point} → ${line.end_point}, 速度 ${line.velocity} mm/s`);
});
```

## 4. HTTP 接口

### 4.1 `POST /api/v1/cad/machining/paths/multi`（**新增**）

**请求体**（`application/json` 推荐，也支持 `multipart/form-data`）：

```json
{
  "model_id":  "9f1c8b...",
  "face_id":   "face_12",          // 或 part_id
  "part_id":   "part_0",
  "hole_ids":  ["hole_3", "hole_1", "hole_7"],
  "include_outer": false,
  "apply_craft_params": true,
  "idle_velocity": 350.0
}
```

**关键参数**：

- `hole_ids` (**必填**) — 用户点击顺序的孔洞 ID 列表。后端**严格按这个顺序**生成 `inner_paths`。未识别的 ID 会被**静默跳过**；若全部不匹配返回 400。
- `include_outer` (默认 `false`) — 是否把外轮廓也包含进来
- `idle_velocity` (默认 `300.0`) — 跨孔空驶速度（mm/s）
- `apply_craft_params` (默认 `true`) — 是否按轮廓类型应用默认工艺

**返回**：JSON, 200 OK

```json
{
  "schema_version": "2.0",
  "unit": "mm",
  "model_id": "9f1c8b...",
  "feature_result": { ... },
  "machining_groups": [
    {
      "id": "group_xxx",
      "name": "MultiHole[3]",
      "inner_paths": [ ... 3 个 MachiningPath，按 hole_ids 顺序 ... ],
      "outer_paths": [],
      "process_face_ids": ["face_12"],
      "path_order": ["path_aaa", "path_bbb", "path_ccc"],
      "transition_lines": [
        { "line_type": "idle", "start_point": ..., "end_point": ..., "velocity": 350, "power": 0 }
      ],
      "is_merged": false
    }
  ],
  "total_path_count": 3,
  "total_line_count": 96
}
```

**错误码**：

| 状态码 | 含义 |
|--------|------|
| 400 | `model_id` / `face_id` / `hole_ids` 缺失，或 `hole_ids` 全部不存在 |
| 404 | `model_id` 缓存已失效 |
| 503 | pythonOCC 不可用或 ikfast 未编译 |

### 4.2 `POST /api/v1/cad/machining/paths`（**原有**，保持兼容）

不带 `hole_ids`，处理**当前 face 的所有孔洞**。**保留**以兼容已有调用方。

### 4.3 其它相关接口

- `GET /api/v1/cad/machining/path_types` — 枚举所有 path / line 类型
- `GET /api/v1/cad/machining/craft_params?contour_type=circle&thickness=1.0`
  — 查询某类型轮廓的默认工艺参数
- `GET /api/v1/cad/feature/status` — 端点列表

## 5. 段分类规则

`_classify_segment()` 按下面规则判定每条 `CAMLine` 的 `out_type`：

| 规则 | 条件 | out_type |
|------|------|----------|
| 1 | 段长 < 0.05mm | `point` |
| 2 | 3 点圆心角 ∈ (0°, 360°)，**非** 180° 退化 | `big_arc` (≥60°) / `small_arc` (≥30°) |
| 3 | 相邻段夹角 ≥ 60°（且非共线） | `three_d_corner` |
| 4 | 段长/总周长 ≥ 0.4 | `long_line` |
| 5 | 段长/总周长 ≥ 0.1 | `shorter_line` |
| 6 | 其余 | `shortest_line` |

> ⚠️ **设计权衡**：当前实现是基于**离散点云**的纯几何判别（不依赖 OCC
> 边类型），因为上游 `discretize.py` 已经把 OCC 边离散成 polyline。 优势
> 是快、无需 pythonOCC 也能测试；劣势是稠密圆周上的 32 段小直线全部
> 归 `shortest_line`。后续若要更精准分类，可以让上层直接传 OCC edge。

## 6. 速度表（`_SEGMENT_VELOCITY`）

| out_type | 速度系数 | 圆孔示例 (base=100) |
|----------|---------|---------------------|
| `long_line` | 1.00 | 100 mm/s |
| `shorter_line` | 0.80 | 80 mm/s |
| `shortest_line` | 0.60 | 60 mm/s |
| `big_arc` | 0.70 | 70 mm/s |
| `small_arc` | 0.50 | 50 mm/s |
| `three_d_corner` | 0.40 | 40 mm/s |
| `point` | 0.30 | 30 mm/s |

> 参考 SmartLaser `CAMLine.cs:LongLine → 250 mm/s、ShorterLine → 150 mm/s`，
> 当前实现把基础速度统一为 `100 mm/s` 方便测试。生产环境应该读
> `Database/Files/CraftRecipesOutline.ini`（TODO: 后续 PR）。

## 7. 测试

`tests/test_machining_multi.py` 9 项测试覆盖：

- `test_click_order_preserved` — `inner_paths` 顺序 == `hole_ids` 顺序
- `test_idle_lines_between_holes` — N 孔产生 N-1 条 idle 线
- `test_unknown_hole_id_raises` — 全部未知时抛 ValueError
- `test_partial_unknown_silently_skipped` — 部分未知时静默跳过
- `test_segment_velocity_uses_table` — 不同段类型速度衰减
- `test_segment_velocity_with_sharp_corner` — 90° 拐角 → `three_d_corner`
- `test_hole_resolves_to_correct_polyline` — wire→polyline 映射正确（regression）
- `test_legacy_entry_point_still_works` — 原 `/machining/paths` 兼容
- `test_response_serializable_to_json` — 完整 JSON round-trip

运行：

```bash
cd E:/SsRobotOCC/backend
python tests/test_machining_multi.py
```

## 8. 已知限制 / 后续工作

- [ ] **Z 轴抬刀 / 下刀轨迹**：`CAMLine` 当前无 `z_entry` / `z_retract` 字段。
      需要在仿真阶段用 ikfast 把 (x,y) 配 Z = 上方安全高度 + 切深切换。
- [ ] **逆解集成**：每条 `CAMLine.robot_joints=[]` 仍是空的。下一步
      编排：构造 Ax2 → 调 `ikfast_service.inverse_kinematics()` → 写回。
- [ ] **工艺库**：`_DEFAULT_CRAFT_PARAMS` 是硬编码。生产应从
      `Database/Files/CraftRecipes*.ini` 读。
- [ ] **段级 OutType 与 OBB 包围盒**：仿真阶段需要根据 `out_type` 调整
      工具朝向（corner 处偏转）。
- [ ] **多边形 / 异形孔**：当前 `_classify_segment` 在不规则多边形上的
      表现取决于离散密度，可能需要 OCC 边类型作为补充信号。
