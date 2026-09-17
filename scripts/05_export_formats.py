#!/usr/bin/env python3
"""
Stage 5 — Export All Formats

Exports the aligned reconstruction to all required output formats:
OBJ, PLY, LAS, GeoTIFF (DSM), glTF/glb, FBX, and COLMAP text.

Input:
    --aligned_points  Path to aligned_points.npz (from Stage 4)
    --frames_meta     Path to frames_meta.csv (for frame filenames in COLMAP)
    --output_dir      Directory for exports
    --formats         Comma-separated list or "all"

Output:
    See README.md § Output Directory Structure for full listing.
"""

import argparse
import json
import logging
import struct
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("05_export_formats")

ALL_FORMATS = ["ply", "obj", "las", "geotiff", "glb", "gltf", "fbx", "colmap"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 5: Export aligned reconstruction to all required formats"
    )
    parser.add_argument(
        "--aligned_points", required=True, type=str,
        help="Path to aligned_points.npz (from Stage 4)"
    )
    parser.add_argument(
        "--frames_meta", type=str, default=None,
        help="Path to frames_meta.csv (for COLMAP image names)"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory for exported files"
    )
    parser.add_argument(
        "--formats", type=str, default="all",
        help="Comma-separated formats to export, or 'all' (default: all)"
    )
    parser.add_argument(
        "--mesh_depth", type=int, default=9,
        help="Poisson reconstruction depth (default: 9)"
    )
    parser.add_argument(
        "--dsm_resolution", type=float, default=0.5,
        help="GeoTIFF DSM grid resolution in meters (default: 0.5)"
    )
    parser.add_argument(
        "--colmap_max_points", type=int, default=500_000,
        help="Max points for COLMAP points3D.txt (default: 500k)"
    )
    return parser.parse_args()


# =============================================================================
# PLY Export
# =============================================================================

def export_ply(points, colors, output_path):
    """Export colored point cloud as binary PLY."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(n):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            f.write(struct.pack("<BBB", *colors[i].astype(np.uint8)))
    log.info("PLY exported: %s (%d points)", output_path, n)


# =============================================================================
# LAS Export
# =============================================================================

def export_las(points, colors, output_path, utm_zone=None):
    """Export georeferenced point cloud as LAS 1.4 with color."""
    try:
        import laspy
    except ImportError:
        log.warning("laspy not installed — skipping LAS export")
        return

    header = laspy.LasHeader(point_format=2, version="1.4")

    # Compute offset (min values) and scale
    mins = points.min(axis=0)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    # Set CRS via VLR if UTM zone is known
    if utm_zone is not None:
        try:
            from pyproj import CRS
            mean_lat = 0  # Rough: sign of first z determines hemisphere
            epsg = 32600 + utm_zone if points[:, 1].mean() >= 0 else 32700 + utm_zone
            crs = CRS.from_epsg(epsg)
            # GeoTIFF VLR for CRS
            header.add_crs(crs)
        except Exception as e:
            log.warning("Could not embed CRS in LAS: %s", e)

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    # LAS color is 16-bit
    las.red = colors[:, 0].astype(np.uint16) * 256
    las.green = colors[:, 1].astype(np.uint16) * 256
    las.blue = colors[:, 2].astype(np.uint16) * 256

    las.write(str(output_path))
    log.info("LAS exported: %s (%d points)", output_path, len(points))


# =============================================================================
# GeoTIFF DSM Export
# =============================================================================

def export_geotiff_dsm(points, output_path, resolution=0.5, utm_zone=None):
    """Rasterize point cloud to a Digital Surface Model GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        log.warning("rasterio not installed — skipping GeoTIFF export")
        return

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    cols = max(1, int(np.ceil((x_max - x_min) / resolution)))
    rows = max(1, int(np.ceil((y_max - y_min) / resolution)))

    log.info("DSM grid: %d × %d (%.1f m resolution)", cols, rows, resolution)

    # Rasterize: assign each point to a grid cell, keep max elevation
    grid = np.full((rows, cols), np.nan, dtype=np.float32)
    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / resolution).astype(int), 0, rows - 1)  # flip y

    for i in range(len(z)):
        r, c = row_idx[i], col_idx[i]
        if np.isnan(grid[r, c]) or z[i] > grid[r, c]:
            grid[r, c] = z[i]

    # Fill small holes with nearest-neighbor
    from scipy.ndimage import maximum_filter
    mask = np.isnan(grid)
    if mask.any():
        filled = maximum_filter(grid, size=3)
        grid[mask] = filled[mask]

    transform = from_bounds(x_min, y_min, x_max, y_max, cols, rows)

    # Determine CRS
    crs_dict = None
    if utm_zone is not None:
        epsg = 32600 + utm_zone  # Assumes northern hemisphere; refine if needed
        crs_dict = f"EPSG:{epsg}"

    with rasterio.open(
        str(output_path), "w",
        driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs=crs_dict, transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(grid, 1)

    log.info("GeoTIFF DSM exported: %s (%d×%d)", output_path, cols, rows)


# =============================================================================
# Mesh Reconstruction (shared by OBJ, glb, gltf, fbx)
# =============================================================================

def build_mesh(points, colors, depth=9):
    """
    Poisson surface reconstruction from colored point cloud.
    Returns an open3d TriangleMesh, or None on failure.
    """
    try:
        import open3d as o3d
    except ImportError:
        log.warning("open3d not installed — skipping mesh reconstruction")
        return None

    log.info("Building mesh from %d points (Poisson depth=%d)...", len(points), depth)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    # Voxel downsample for speed (adaptive: ~0.5% of extent)
    extent = np.ptp(points, axis=0).max()
    voxel_size = max(extent * 0.002, 0.1)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    log.info("Downsampled %d → %d points (voxel %.2f m)", len(points), len(pcd_down.points), voxel_size)

    # Estimate normals
    radius = voxel_size * 4
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_down, depth=depth
    )

    # Trim low-density vertices (noise at boundary)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    log.info("Mesh: %d vertices, %d triangles",
             len(mesh.vertices), len(mesh.triangles))
    return mesh


def export_obj(mesh, output_path):
    """Export mesh as Wavefront OBJ."""
    import open3d as o3d
    success = o3d.io.write_triangle_mesh(str(output_path), mesh,
                                         write_vertex_colors=True)
    if success:
        log.info("OBJ exported: %s", output_path)
    else:
        log.error("OBJ export failed")


def export_glb(mesh, output_path):
    """Export mesh as glTF binary using trimesh."""
    try:
        import trimesh
    except ImportError:
        log.warning("trimesh not installed — skipping glb export")
        return

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v_colors = np.asarray(mesh.vertex_colors)

    vertex_colors = None
    if len(v_colors) > 0:
        rgba = np.hstack([
            (v_colors * 255).astype(np.uint8),
            np.full((len(v_colors), 1), 255, dtype=np.uint8)
        ])
        vertex_colors = rgba

    tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles,
                            vertex_colors=vertex_colors)
    tmesh.export(str(output_path), file_type="glb")
    log.info("GLB exported: %s", output_path)


def export_gltf(mesh, output_path):
    """Export mesh as glTF text using trimesh."""
    try:
        import trimesh
    except ImportError:
        log.warning("trimesh not installed — skipping gltf export")
        return

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    v_colors = np.asarray(mesh.vertex_colors)

    vertex_colors = None
    if len(v_colors) > 0:
        rgba = np.hstack([
            (v_colors * 255).astype(np.uint8),
            np.full((len(v_colors), 1), 255, dtype=np.uint8)
        ])
        vertex_colors = rgba

    tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles,
                            vertex_colors=vertex_colors)
    tmesh.export(str(output_path), file_type="gltf")
    log.info("GLTF exported: %s", output_path)


def export_fbx(mesh, output_path):
    """
    Export mesh as FBX. Requires pyfbx or Blender headless.
    Warns and skips if neither is available.
    """
    try:
        import trimesh
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        tmesh.export(str(output_path), file_type="fbx")
        log.info("FBX exported: %s", output_path)
    except Exception as e:
        log.warning(
            "FBX export failed (%s). Install pyfbx or Blender headless for FBX support.", e
        )


# =============================================================================
# COLMAP Text Export
# =============================================================================

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    # Shepperd's method
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def export_colmap(extrinsics, intrinsics, points, colors,
                  output_dir, frame_names=None, max_points=500_000):
    """
    Write COLMAP text-format files: cameras.txt, images.txt, points3D.txt.
    Uses PINHOLE camera model (fx, fy, cx, cy).
    """
    colmap_dir = Path(output_dir) / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    S = extrinsics.shape[0]

    # --- cameras.txt ---
    # Assume shared intrinsics (first frame's K). One camera entry.
    K = intrinsics[0]  # (3,3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Guess image dimensions from principal point (rough)
    img_w = int(cx * 2)
    img_h = int(cy * 2)

    cam_path = colmap_dir / "cameras.txt"
    with open(cam_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {img_w} {img_h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    log.info("COLMAP cameras.txt: 1 PINHOLE camera (%dx%d, f=%.1f)", img_w, img_h, fx)

    # --- images.txt ---
    img_path = colmap_dir / "images.txt"
    with open(img_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(S):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            name = frame_names[i] if frame_names and i < len(frame_names) else f"frame_{i:04d}.jpg"
            f.write(f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {name}\n")
            f.write("\n")  # Empty POINTS2D line
    log.info("COLMAP images.txt: %d images", S)

    # --- points3D.txt ---
    if len(points) > max_points:
        idx = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        idx.sort()
        pts = points[idx]
        clrs = colors[idx]
    else:
        pts, clrs = points, colors

    pts_path = colmap_dir / "points3D.txt"
    with open(pts_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i in range(len(pts)):
            x, y, z = pts[i]
            r, g, b = clrs[i]
            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")
    log.info("COLMAP points3D.txt: %d points", len(pts))


# =============================================================================
# Main
# =============================================================================

def run_exports(args):
    """Load aligned data and export requested formats."""

    data_path = Path(args.aligned_points)
    if not data_path.exists():
        log.error("Aligned points not found: %s", data_path)
        sys.exit(1)

    log.info("Loading aligned points from %s", data_path)
    data = np.load(str(data_path))
    points = data["points"]
    colors = data["colors"]
    confidence = data["confidence"]
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    utm_zone = int(data["utm_zone"][0]) if "utm_zone" in data else None

    log.info("Loaded %d points, %d frames", len(points), extrinsics.shape[0])

    # Parse requested formats
    if args.formats.strip().lower() == "all":
        formats = ALL_FORMATS
    else:
        formats = [f.strip().lower() for f in args.formats.split(",")]
        unknown = set(formats) - set(ALL_FORMATS)
        if unknown:
            log.error("Unknown formats: %s (supported: %s)", unknown, ALL_FORMATS)
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Frame names for COLMAP
    frame_names = None
    if args.frames_meta and Path(args.frames_meta).exists():
        import pandas as pd
        meta_df = pd.read_csv(args.frames_meta)
        if "frame_file" in meta_df.columns:
            frame_names = meta_df["frame_file"].tolist()

    # --- Point cloud exports ---
    if "ply" in formats:
        export_ply(points, colors, output_dir / "cloud.ply")

    if "las" in formats:
        export_las(points, colors, output_dir / "cloud.las", utm_zone)

    if "geotiff" in formats:
        export_geotiff_dsm(points, output_dir / "dsm.tif",
                           resolution=args.dsm_resolution, utm_zone=utm_zone)

    if "colmap" in formats:
        export_colmap(extrinsics, intrinsics, points, colors,
                      output_dir, frame_names, args.colmap_max_points)

    # --- Mesh exports (build once, export multiple) ---
    mesh_formats = {"obj", "glb", "gltf", "fbx"}
    needed_mesh = mesh_formats & set(formats)
    mesh = None
    if needed_mesh:
        mesh = build_mesh(points, colors, depth=args.mesh_depth)

    if mesh is not None:
        if "obj" in formats:
            export_obj(mesh, output_dir / "model.obj")
        if "glb" in formats:
            export_glb(mesh, output_dir / "model.glb")
        if "gltf" in formats:
            export_gltf(mesh, output_dir / "model.gltf")
        if "fbx" in formats:
            export_fbx(mesh, output_dir / "model.fbx")
    elif needed_mesh:
        log.warning("Mesh reconstruction failed — skipping %s", needed_mesh)

    log.info("All exports complete → %s", output_dir)


if __name__ == "__main__":
    args = parse_args()
    run_exports(args)
    log.info("Stage 5 complete")
