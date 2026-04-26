#!/usr/bin/env python3
"""
Utility for generating Eval3D-style multi‑view renders from an arbitrary mesh.

You already have a bunch of evaluation tools in this repo but none of them
produce the renders themselves; they assume you have a `save/.../rgb_images`
folder full of views.  This script takes a `.obj`/`.glb` file and produces the
same directory layout automatically, so you can evaluate models created with
other frameworks.

Example usage (run from the workspace root):

    python Eval3D/render_multiview.py \
        --mesh /path/to/model.glb \
        --out $DATA_PATH/my_algo/a_prompt/save/it0-test \
        --n-views 120 \
        --radius 2.0

The output directory will contain the three subdirectories required by the
metrics:

    rgb_images/0000.png ... 0119.png
    opacity/   0000.png ... 0119.png
    batch_data/0000.npy ... 0119.npy

`batch_data` files include the same fields that the evaluation code consumes
(`c2w`, `camera_positions`, `proj_mtx`, `elevation`, `azimuth`,
`camera_distances`).  You can then point the geometric/semantic/structural
scripts at the parent `my_algo` folder and they will operate on the newly
rendered asset.
"""

import argparse
import os
import sys

# Add the Eval3D/ directory to sys.path so that sibling packages like
# `semantic_consistency` can be imported regardless of cwd.
eval3d_dir = os.path.abspath(os.path.dirname(__file__))
if eval3d_dir not in sys.path:
    sys.path.insert(0, eval3d_dir)

import numpy as np
import torch
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    MeshRenderer,
    PointLights,
    TexturesVertex,
)


def load_mesh(mesh_path: str, device: str) -> Meshes:
    """Load .obj or any trimesh-supported format (glb, ply, stl, …)."""
    if mesh_path.lower().endswith(".obj"):
        mesh = load_objs_as_meshes([mesh_path], device=torch.device(device))
    else:
        import trimesh
        loaded = trimesh.load(mesh_path, force="mesh")
        tm: trimesh.Trimesh = loaded if isinstance(loaded, trimesh.Trimesh) else trimesh.util.concatenate(loaded)
        verts = torch.from_numpy(tm.vertices.astype(np.float32)).unsqueeze(0).to(device)
        faces = torch.from_numpy(tm.faces.astype(np.int64)).unsqueeze(0).to(device)
        mesh = Meshes(verts=verts, faces=faces)

    # Apply the same YZX axis permutation that utils_3d.load_mesh uses so that
    # the mesh orientation is consistent with what the evaluators expect.
    v: torch.Tensor = mesh.verts_padded()  # type: ignore[assignment]
    vx, vy, vz = v.unbind(dim=-1)  # each (1, V)
    mesh = mesh.update_padded(
        new_verts_padded=torch.stack([vy, vz, vx], dim=-1)
    )

    # Ensure a texture exists (white vertex colours) so the shader doesn't fail.
    if mesh.textures is None:
        vp: torch.Tensor = mesh.verts_padded()  # type: ignore[assignment]
        mesh.textures = TexturesVertex(torch.ones_like(vp))

    return mesh


def build_renderer(elev: float, azim: float, dist: float, device: str):
    """Create a pytorch3d renderer for a single camera pose."""
    R, T = look_at_view_transform(
        dist=dist, elev=elev, azim=azim, degrees=True,
        at=((0, 0, 0),), up=((0, 1, 0),), device=device,
    )
    # FoVPerspectiveCameras requires only R/T + fov; no projection matrix needed.
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60.0)
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(
        device=device,
        cameras=cameras,
        lights=PointLights(device=device, location=[[0.0, 3.0, 3.0]]),
    )
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer, cameras, R, T


def main():
    parser = argparse.ArgumentParser(
        description="Render an object from many viewpoints for Eval3D metrics",
    )
    parser.add_argument("--mesh", required=True, help="input mesh (.obj / .glb / …)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n-views", type=int, default=120)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    mesh = load_mesh(args.mesh, args.device)

    # Export an OBJ next to the test folder so semantic_consistency/evaluate.py
    # can find it at  …/save/it0-export/model.obj  without needing threestudio.
    export_dir = os.path.join(os.path.dirname(args.out), "it0-export")
    obj_path   = os.path.join(export_dir, "model.obj")
    if not os.path.exists(obj_path):
        import trimesh
        os.makedirs(export_dir, exist_ok=True)
        loaded = trimesh.load(args.mesh, force="mesh")
        tm_export: trimesh.Trimesh = loaded if isinstance(loaded, trimesh.Trimesh) else trimesh.util.concatenate(loaded)
        tm_export.export(obj_path)
        print(f"OBJ exported to: {obj_path}")

    rgb_dir = os.path.join(args.out, "rgb_images")
    opa_dir = os.path.join(args.out, "opacity")
    bd_dir  = os.path.join(args.out, "batch_data")
    for d in (rgb_dir, opa_dir, bd_dir):
        os.makedirs(d, exist_ok=True)

    for idx, az in enumerate(np.linspace(0, 360, args.n_views, endpoint=False)):
        elev = 0.0
        dist = float(args.radius)
        az   = float(az)

        renderer, cameras, R, T = build_renderer(elev, az, dist, args.device)
        images = renderer(mesh)  # (1, H, W, 4) RGBA in [0, 1]

        img  = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        mask = (images[0, ...,  3].cpu().numpy() > 0).astype(np.uint8) * 255

        name = f"{idx:04d}.png"
        cv2.imwrite(os.path.join(rgb_dir, name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(opa_dir, name), mask)

        # Reconstruct c2w (camera-to-world) from R, T.
        # In pytorch3d: X_cam = R @ X_world + T  →  X_world = R^T @ (X_cam - T)
        c2w = torch.eye(4)
        c2w[:3, :3] = R[0].cpu().T
        c2w[:3,  3] = -(R[0].cpu().T @ T[0].cpu())

        # Build a 4×4 perspective projection matrix compatible with the
        # evaluator's `proj_mtx` field (stored as numpy for np.save).
        fov_rad = np.deg2rad(60.0)
        f = 1.0 / np.tan(fov_rad / 2.0)
        proj_mtx = np.array([[f, 0, 0, 0],
                              [0, f, 0, 0],
                              [0, 0, -1, -2*dist],
                              [0, 0, -1, 0]], dtype=np.float32)[None]  # (1.4.4)

        batch = {
            "c2w":              c2w.numpy(),
            "camera_positions": c2w[:3, 3].numpy()[None],
            "proj_mtx":         proj_mtx,
            "elevation":        np.array([elev], dtype=np.float32),
            "azimuth":          np.array([az],   dtype=np.float32),
            "camera_distances": np.array([dist], dtype=np.float32),
        }
        np.save(os.path.join(bd_dir, f"{idx:04d}.npy"), batch, allow_pickle=True)  # type: ignore


if __name__ == "__main__":
    main()
