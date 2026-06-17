# CAD 接口 — 外部前端调用说明

本仓库**仅提供 HTTP API**，不包含前端页面。任意 Web / 桌面 / 移动端项目通过 REST 调用即可。

- 服务地址（本地开发）：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/openapi.json`
- Swagger UI：`http://127.0.0.1:8000/swagger`
- 单位：毫米（`mm`）

> 特征识别（analyze / path / face / face_spread）子系统已被移除，相关端点暂未恢复；
> pythonOCC 算法部分（拓扑遍历、几何量、wire 离散、GLB 导出、STEP 加载）保留在
> `backend/app/occ/` 下，等待重新设计特征识别模块。

---

## 1. 接口一览

| 方法 | 路径 | Content-Type | 说明 |
|------|------|--------------|------|
| GET  | `/api/v1/cad/status`         | —                       | 是否安装 pythonOCC |
| POST | `/api/v1/cad/upload`         | `multipart/form-data`   | 上传 STEP 一次 → 返回 `model_id` |
| POST | `/api/v1/cad/upload/binary`  | `application/octet-stream` | 直接传 STEP 字节，绕过 multipart |
| GET  | `/api/v1/stp/status`         | —                       | STEP 转换服务状态 |
| POST | `/api/v1/stp/convert`        | `multipart/form-data`   | STEP → GLB（保留层级）|
| GET  | `/api/v1/stp/convert/{model_id}` | —                     | 使用缓存 STEP 转 GLB |
| GET  | `/api/v1/stp/hierarchy`      | `multipart/form-data`   | 查询层级结构（JSON）|
| GET  | `/api/v1/stp/hierarchy/{model_id}` | —                  | 使用缓存查询层级结构 |

## 2. `POST /api/v1/cad/upload`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | `.stp` / `.step` |

返回：

```json
{ "model_id": "9f1c...", "filename": "part.step", "suffix": ".step", "size": 123456 }
```

- `model_id` 由文件内容哈希得到：**同一文件重复上传会复用同一 id**。
- 缓存默认保留约 24 小时；过期后调用会返回 404，重新 `upload` 即可。

## 3. `POST /api/v1/cad/upload/binary`

直接传 STEP 原始字节（`Content-Type: application/octet-stream`），绕过 multipart 解析。
文件名优先取 query 参数 `filename`，其次请求头 `X-Filename`，最后回退到 `model.stp`。

## 4. STEP → GLB（`/api/v1/stp/convert`）

用于 3D 预览显示，**保留 STEP 文件中的层级结构**。

```
GET  /api/v1/stp/status     → { convert_ready, engine, features, ... }
POST /api/v1/stp/convert    → multipart file → 二进制 GLB
GET  /api/v1/stp/convert/{model_id} → 使用缓存转 GLB
```

Query/Form 参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `linear_deflection` | `0.1` | 网格精度 (mm)，越小越精细 |
| `angular_deflection` | `0.5` | 角度偏差 (rad) |
| `preserve_hierarchy` | `true` | 是否保留层级结构 |

返回：

- Content-Type: `model/gltf-binary`
- Header `X-Meta-Node-Count`: 层级节点数
- Header `X-Meta-Mesh-Count`: 网格数量

**层级结构说明**：

GLB 中的每个 GLTF node 对应 STEP 文件中的一个 Product（零件/部件）：

```json
// GLTF nodes 示例
{
  "name": "model",           // 根节点
  "children": [0, 1],        // 子节点索引
  "matrix": [...]            // 变换矩阵
}
{
  "name": "Base_Part",       // STEP Product 名称
  "mesh": 0,                 // 指向 meshes[0]
  "children": [2]
}
{
  "name": "Arm_Part",        // 子零件
  "mesh": 1,
  "matrix": [...]            // 局部变换
}
```

## 5. 查询层级结构（`/api/v1/stp/hierarchy`）

在不生成 GLB 的情况下查询 STEP 文件的层级信息。

```
GET  /api/v1/stp/hierarchy    → 上传文件
GET  /api/v1/stp/hierarchy/{model_id} → 使用缓存
```

返回示例：

```json
{
  "filename": "assembly.step",
  "node_count": 3,
  "mesh_count": 3,
  "nodes": [
    {
      "name": "Main_Assembly",
      "shape_type": "COMPOUND",
      "location": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
      "has_mesh": false,
      "face_count": 0,
      "children": [1, 2]
    },
    {
      "name": "Base_Part",
      "shape_type": "SOLID",
      "has_mesh": true,
      "face_count": 12,
      "children": []
    },
    {
      "name": "Arm_Part",
      "shape_type": "SOLID",
      "has_mesh": true,
      "face_count": 8,
      "children": []
    }
  ]
}
```

## 6. CORS（跨域）

后端默认允许所有来源（`allow_origins=["*"]`）。生产环境在 `backend/.env` 配置：

```env
CORS_ALLOW_ORIGINS=https://your-frontend.com,http://localhost:5173
CORS_ALLOW_CREDENTIALS=true
```

## 7. 后端启动（供外部联调）

```bat
cd RobotLaserNew
run_server.cmd
```

需 Conda 环境 `occ`（含 pythonOCC）。
