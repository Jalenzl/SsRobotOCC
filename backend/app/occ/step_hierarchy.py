"""读取 STEP 文件的层级结构 (产品树) via pythonOCC XDE.

XDE 路径对装配树缺失 / `GetComponents` 退化的 STEP（如部分 AP214 文件，
产品关系写成 `MANIFOLD_SOLID_BREP` 挂在 root shape 上而非 product 组件），
提供 3 层兜底以保证至少能拿到完整几何：
1) XDE 组件树遍历（标准路径）
2) XDE 当前 shape 内部按 Solid 拆分为虚拟 part 子节点
3) 退到 `STEPControl_Reader` 直接拿所有 Solid，每个 Solid 当一个 part
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Shape, topods

try:
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
    from OCC.Core.TDF import TDF_LabelSequence
    from OCC.Core.TDocStd import TDocStd_Document
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorTool, XCAFDoc_ShapeTool
    XDE_AVAILABLE = True
except ImportError:
    XDE_AVAILABLE = False


@dataclass
class HierarchyNode:
    """单个产品节点."""
    name: str
    part_id: str                    # 唯一标识 (用于前端 metadata)
    label_path: str                 # 标签路径，用于调试
    shape: TopoDS_Shape | None = None  # 几何体
    location_matrix: list[float] = field(default_factory=lambda: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
    color: tuple[float, float, float] | None = None  # RGB
    children: list[HierarchyNode] = field(default_factory=list)
    is_assembly: bool = False       # 是否有子节点

    def to_dict(self) -> dict[str, Any]:
        """转换为前端可用的字典."""
        result: dict[str, Any] = {
            "name": self.name,
            "part_id": self.part_id,
            "is_assembly": self.is_assembly,
            "matrix": self.location_matrix,
        }
        if self.color:
            result["color"] = list(self.color)
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


class StepHierarchyReader:
    """读取 STEP 文件的层级结构 (XDE/产品树)."""

    def __init__(self):
        if not XDE_AVAILABLE:
            raise ImportError(
                "pythonOCC XDE 模块不可用。需要安装完整版 pythonOCC: "
                "conda install -c conda-forge pythonocc-core"
            )

    def read(self, data: bytes, filename_hint: str = "model.stp") -> tuple[HierarchyNode, list[TopoDS_Shape]]:
        """读取 STEP 文件并返回层级树和所有几何体.

        Returns:
            (root_node, shapes): 根节点和所有 TopoDS_Shape 列表
        """
        import tempfile
        from pathlib import Path

        suffix = ".step" if filename_hint.lower().endswith(".step") else ".stp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            path = tmp.name

        try:
            root, shapes = self._read_file(Path(path), filename_hint)
        finally:
            Path(path).unlink(missing_ok=True)

        if self._tree_has_geometry(root) or not shapes:
            return root, shapes

        # 兜底 1：XDE 把形状吸进文档了，但产品树拆不出 part — 把 shapes 拆
        # 成虚拟 part 挂到 root 下。
        fallback_parts = self._split_solids(shapes)
        if fallback_parts:
            if root is None:
                root = HierarchyNode(
                    name=filename_hint.rsplit(".", 1)[0] if filename_hint else "Assembly",
                    part_id="root",
                    label_path="",
                )
            root.is_assembly = True
            for i, solid in enumerate(fallback_parts):
                # 关键：solid 自身可能还携带 Location（XDE 在
                # label 上挂的 location 会沿拓扑传播到 solid），
                # 这里必须 strip 掉，否则 tessellate 后顶点 + 节点
                # matrix 会双倍叠加。
                local_solid = self._strip_location(solid)
                root.children.append(
                    HierarchyNode(
                        name=f"{root.name}_part_{i}",
                        part_id=f"{root.part_id}_p{i}",
                        label_path="",
                        location_matrix=self._location_to_matrix(solid.Location()),
                        shape=local_solid,
                    )
                )
            shapes = fallback_parts
            return root, shapes

        # 兜底 2：完全退到非 XDE 读取，从根 shape 拆所有 Solid。
        solids = self.read_shapes(data, filename_hint)
        if solids:
            root = HierarchyNode(
                name=filename_hint.rsplit(".", 1)[0] if filename_hint else "Assembly",
                part_id="root",
                label_path="",
                is_assembly=True,
            )
            for i, solid in enumerate(solids):
                # read_shapes 返回的 solid 同样可能携带 Location，
                # strip 后再下发。
                local_solid = self._strip_location(solid)
                root.children.append(
                    HierarchyNode(
                        name=f"{root.name}_part_{i}",
                        part_id=f"{root.part_id}_p{i}",
                        label_path="",
                        location_matrix=self._location_to_matrix(solid.Location()),
                        shape=local_solid,
                    )
                )
            return root, solids

        return root, shapes

    @staticmethod
    def _tree_has_geometry(root: HierarchyNode | None) -> bool:
        """递归检查层级树里是否有任何节点带几何."""
        if root is None:
            return False
        if root.shape is not None and not root.shape.IsNull():
            return True
        return any(StepHierarchyReader._tree_has_geometry(c) for c in root.children)

    @staticmethod
    def _split_solids(shapes: list[TopoDS_Shape]) -> list[TopoDS_Shape]:
        """从 XDE shape 列表里把所有 Solid 展平出来."""
        solids: list[TopoDS_Shape] = []
        for shape in shapes:
            if shape is None or shape.IsNull():
                continue
            exp = TopExp_Explorer(shape, TopAbs_SOLID)
            while exp.More():
                solids.append(topods.Solid(exp.Current()))
                exp.Next()
        return solids

    def _strip_location(self, shape: TopoDS_Shape | None) -> TopoDS_Shape | None:
        """Return a copy of the shape with its accumulated Location removed.

        ``XCAFDoc_ShapeTool.GetShape(label)`` returns the shape **with the
        label's accumulated transform already applied** (i.e. in the
        assembly's world frame). If we then tessellate that shape and also
        store the label's Location as the node's matrix, every vertex gets
        the transform applied twice — the part ends up far from its true
        position and the assembly looks like its components are flying
        apart. To get a correct render we must keep the matrix (so the
        front-end can still query position / rotation) but tessellate the
        shape **without** its Location, i.e. in the label's local frame.
        Vertices are then ``local × matrix`` = world.
        """
        if shape is None or shape.IsNull():
            return shape
        try:
            return shape.Located(TopLoc_Location())
        except Exception:
            return shape

    def read_shapes(self, data: bytes, filename_hint: str) -> list[TopoDS_Shape]:
        """非 XDE 兜底：用 ``STEPControl_Reader`` 直接取 STEP 中的所有 Solid.

        在 XDE 完全失败（root 为空 + shapes 为空）时使用。

        注意：``read_step_bytes`` 返回的 shape 处于世界坐标；我们这里
        要拿到每个 Solid 的局部坐标，调用方会在循环里 ``_strip_location``。
        """
        from app.occ.loader import read_step_bytes

        shape = read_step_bytes(data, filename_hint)
        if shape is None or shape.IsNull():
            return []
        return self._split_solids([shape])

    def _read_file(self, path: Path, filename_hint: str) -> tuple[HierarchyNode, list[TopoDS_Shape]]:
        reader = STEPCAFControl_Reader()
        status = reader.ReadFile(str(path))
        if status != IFSelect_RetDone:
            raise ValueError(f"STEP read failed: {path}")

        # 创建 XDE 文档
        doc = TDocStd_Document("step-xcaf")
        
        # Transfer 到文档
        if not reader.Transfer(doc):
            raise ValueError("STEP transfer failed")

        # 从文档获取主标签
        main_label = doc.Main()
        
        shape_tool = XCAFDoc_DocumentTool.ShapeTool(main_label)
        color_tool = XCAFDoc_DocumentTool.ColorTool(main_label)

        shapes: list[TopoDS_Shape] = []
        root = self._build_tree(shape_tool, color_tool, shapes)
        return root, shapes

    def _mat_multiply(
        self,
        a: list[float],
        b: list[float],
    ) -> list[float]:
        """Multiply two 4x4 matrices stored as 16-element column-major lists."""
        # glTF stores matrices in column-major order: m[0..3] = column 0,
        # m[4..7] = column 1, m[8..11] = column 2, m[12..15] = column 3.
        out = [0.0] * 16
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += a[k * 4 + row] * b[col * 4 + k]
                out[col * 4 + row] = s
        return out

    def _build_tree(
        self,
        shape_tool: "XCAFDoc_ShapeTool",
        color_tool: "XCAFDoc_ColorTool",
        shapes: list[TopoDS_Shape],
        label=None,
        parent_path: str = "",
        depth: int = 0,
        cumulative_matrix: list[float] | None = None,
    ) -> HierarchyNode | None:
        if label is None:
            labels = TDF_LabelSequence()
            shape_tool.GetShapes(labels)
            if labels.Length() == 0:
                return None
            label = labels.Value(1)

        name = self._get_label_name(label, depth, parent_path)
        part_id = self._make_part_id(label, depth)

        loc = shape_tool.GetLocation(label)
        local_matrix = self._location_to_matrix(loc)

        # 把 parent 累积的矩阵乘上本 label 的 local transform，
        # 才是这个 label 上的 shape 真正应该出现的世界 transform。
        # 当 ``cumulative_matrix`` 为 None（递归顶层）时，父级视为 identity。
        if cumulative_matrix is None:
            cumulative_matrix = self._identity_matrix_list()
        node_matrix = self._mat_multiply(cumulative_matrix, local_matrix)

        color = self._get_label_color(label, color_tool)

        node = HierarchyNode(
            name=name,
            part_id=part_id,
            label_path=str(parent_path),
            location_matrix=node_matrix,
            color=color,
        )

        sub_labels = TDF_LabelSequence()
        shape_tool.GetComponents(label, sub_labels)

        if sub_labels.Length() > 0:
            node.is_assembly = True
            for i in range(1, sub_labels.Length() + 1):
                sub_label = sub_labels.Value(i)
                child_node = self._build_tree(
                    shape_tool,
                    color_tool,
                    shapes,
                    sub_label,
                    f"{parent_path}/{name}",
                    depth + 1,
                    node_matrix,
                )
                if child_node:
                    node.children.append(child_node)
            return node

        shape = shape_tool.GetShape(label)
        if shape.IsNull():
            return node

        # 关键：``GetShape(label)`` 返回的 shape **已经**把 label 的
        # 累积 Location 应用到几何里了（XDE 语义）。如果直接拿去
        # tessellate，再把 ``label.GetLocation()`` 写到节点的
        # ``location_matrix``，每个顶点会被矩阵双倍变换，导致
        # 装配体各零件相互错位。这里把 Location 剥掉，只保留
        # 节点 ``location_matrix``（已是 parent_cumulative × local
        # 的累积矩阵），最终 ``vertex × matrix`` 才等于真实世界坐标。
        shape = self._strip_location(shape)

        # 兜底 1：XDE 在当前 label 上没有组件，但 label 直接挂了一个 shape
        # （典型场景：AP214 把装配体作为 compound 直接挂在 root label，
        # 没有拆成 product components）。把 shape 里的每个 Solid 当作
        # 一个虚拟 part 子节点；当前 node 升格为 assembly。
        solids: list[TopoDS_Shape] = []
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solids.append(topods.Solid(exp.Current()))
            exp.Next()

        if len(solids) > 1:
            node.is_assembly = True
            for si, solid in enumerate(solids):
                # 关键：solid 自身可能还带 Location（compound 里各
                # solid 之间的相对位移），必须 strip 后顶点才能进
                # 入 solid-local 坐标系；矩阵要包含 parent 累积 ×
                # solid 自身的 Location，才是 sub-solid 的世界 transform。
                local_solid = self._strip_location(solid)
                sub_matrix = self._mat_multiply(
                    node_matrix, self._location_to_matrix(solid.Location())
                )
                sub = HierarchyNode(
                    name=f"{name}_part_{si}",
                    part_id=f"{part_id}_s{si}",
                    label_path=f"{parent_path}/{name}",
                    location_matrix=sub_matrix,
                    shape=local_solid,
                )
                node.children.append(sub)
                shapes.append(local_solid)
            node.shape = None
            return node

        if len(solids) == 1:
            node.shape = solids[0]
            shapes.append(solids[0])
            return node

        # shape 既不是 compound 也不是单 solid（罕见），保留原始 shape
        # 让下游有机会处理。
        node.shape = shape
        shapes.append(shape)
        return node

    @staticmethod
    def _identity_matrix_list() -> list[float]:
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def _get_label_name(self, label, depth: int, parent_path: str) -> str:
        """获取标签的名称."""
        name_tool = None
        try:
            from OCC.Core.TDataStd import TDataStd_Name
            name_tool = TDataStd_Name
        except ImportError:
            pass

        if name_tool:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    name = name_tool.Get(label)
                    if name.Length() > 0:
                        return str(name.ToCString())
            except Exception:
                pass

        if depth == 0:
            return "Assembly"
        return f"Part_{depth}"

    def _make_part_id(self, label, depth: int) -> str:
        """生成唯一的 part_id."""
        import hashlib
        label_str = str(label)
        short_hash = hashlib.md5(label_str.encode()).hexdigest()[:8]
        return f"p{depth}_{short_hash}"

    def _location_to_matrix(self, loc) -> list[float]:
        """将 TopLoc_Location 转换为 4x4 矩阵."""
        identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        if loc is None or loc.IsIdentity():
            return identity

        try:
            trsf = loc.Transformation()
            values = [
                trsf.Value(1, 1), trsf.Value(1, 2), trsf.Value(1, 3), trsf.Value(1, 4),
                trsf.Value(2, 1), trsf.Value(2, 2), trsf.Value(2, 3), trsf.Value(2, 4),
                trsf.Value(3, 1), trsf.Value(3, 2), trsf.Value(3, 3), trsf.Value(3, 4),
            ]
            return [float(v) for v in values] + [0.0, 0.0, 0.0, 1.0]
        except Exception:
            return identity

    def _get_label_color(self, label, color_tool) -> tuple[float, float, float] | None:
        """获取标签的颜色 (如果有)."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from OCC.Core.Quantity import Quantity_Color
                from OCC.Core.TColStd import TColStd_HSequenceOfColor
                colors = TColStd_HSequenceOfColor()
                if color_tool.GetInstanceColor(label, colors):
                    c = colors.Value(1)
                    if c.ColorType() == 1:
                        return (c.Red(), c.Green(), c.Blue())
        except Exception:
            pass
        return None


def read_step_hierarchy(data: bytes, filename_hint: str = "model.stp") -> dict[str, Any]:
    """便捷函数：读取 STEP 层级结构.

    Returns:
        {
            "schema": "robotlaser.step.hierarchy/v1",
            "root": HierarchyNode.to_dict(),
            "total_parts": int,
            "total_assemblies": int,
        }
    """
    if not XDE_AVAILABLE:
        raise ImportError("需要 pythonOCC XDE 模块")

    reader = StepHierarchyReader()
    root, shapes = reader.read(data, filename_hint)

    def count_nodes(node: HierarchyNode) -> tuple[int, int]:
        parts = 0 if node.is_assembly else 1
        assemblies = 1 if node.is_assembly else 0
        for child in node.children:
            cp, ca = count_nodes(child)
            parts += cp
            assemblies += ca
        return parts, assemblies

    total_parts, total_assemblies = count_nodes(root)

    return {
        "schema": "robotlaser.step.hierarchy/v1",
        "unit": "mm",
        "filename": filename_hint,
        "root": root.to_dict(),
        "total_parts": total_parts,
        "total_assemblies": total_assemblies,
        "total_shapes": len(shapes),
    }
