"""Load mesh / point cloud vertices from PLY (ASCII or binary LE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply_vertices(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Vertex x,y,z and optional RGB from an ARKit-style 3DOD .ply mesh."""
    mesh_path = Path(mesh_path)
    with mesh_path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line_b = f.readline()
            if not line_b:
                raise ValueError(f"Invalid PLY, missing end_header: {mesh_path}")
            line = line_b.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt = None
        vertex_count = None
        vertex_props: list[tuple[str, str]] = []
        current_element = None
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[:1] == ["format"]:
                fmt = parts[1]
            elif parts[:1] == ["element"]:
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[:1] == ["property"] and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("List properties are not expected in the vertex element")
                vertex_props.append((parts[2], parts[1]))

        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {mesh_path}")
        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"Unsupported PLY format {fmt!r}; expected ascii or binary_little_endian")

        names = [name for name, _ in vertex_props]
        for required in ("x", "y", "z"):
            if required not in names:
                raise ValueError(f"PLY missing vertex property {required!r}: {mesh_path}")

        if fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append(f.readline().decode("ascii").split())
            arr = np.asarray(rows, dtype=np.float64)
            col = {name: i for i, name in enumerate(names)}
            points = arr[:, [col["x"], col["y"], col["z"]]].astype(np.float32)
            if {"red", "green", "blue"}.issubset(names):
                colors = arr[:, [col["red"], col["green"], col["blue"]]].clip(0, 255).astype(np.uint8)
            else:
                colors = None
            return points, colors

        dtype = np.dtype([(name, PLY_DTYPES[typ]) for name, typ in vertex_props])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        points = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
        if {"red", "green", "blue"}.issubset(names):
            colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
        else:
            colors = None
        return points, colors
