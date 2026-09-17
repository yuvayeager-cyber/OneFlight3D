"""
utils/geo_utils.py — Geospatial utility functions for OneFlight3D

Contains:
    - WGS84 → local ENU (East-North-Up) coordinate conversion
    - WGS84 → UTM zone auto-detection and conversion
    - Umeyama similarity transform solver (scale + rotation + translation)
    - Transform application
    - Spatial spread validation for alignment quality
"""

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("geo_utils")

try:
    from pyproj import CRS, Transformer
except ImportError:
    log.warning(
        "pyproj not installed — GPS coordinate transforms will fail. "
        "Install with: pip install pyproj"
    )


# =============================================================================
# Coordinate Conversions
# =============================================================================

def wgs84_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float,
) -> np.ndarray:
    """
    Convert WGS84 (lat, lon, alt) to local East-North-Up (ENU) meters,
    relative to a reference point.

    Uses pyproj for accurate geodetic → ECEF → ENU conversion.

    Args:
        lat, lon, alt: Arrays of WGS84 coordinates (degrees, degrees, meters)
        ref_lat, ref_lon, ref_alt: Reference point (typically first GPS fix)

    Returns:
        enu: (N, 3) array of [East, North, Up] coordinates in meters
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    # WGS84 → ECEF
    crs_wgs84 = CRS.from_epsg(4326)  # WGS84 geographic
    crs_ecef = CRS.from_epsg(4978)   # WGS84 ECEF (geocentric)
    transformer_to_ecef = Transformer.from_crs(crs_wgs84, crs_ecef, always_xy=True)

    # Convert all points to ECEF
    x, y, z = transformer_to_ecef.transform(lon, lat, alt)
    x_ref, y_ref, z_ref = transformer_to_ecef.transform(ref_lon, ref_lat, ref_alt)

    # ECEF → ENU rotation matrix at the reference point
    lat_r = np.radians(ref_lat)
    lon_r = np.radians(ref_lon)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    # ENU rotation matrix (3x3)
    R_enu = np.array([
        [-sin_lon,            cos_lon,             0       ],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,   cos_lat ],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,   sin_lat ],
    ])

    # Translate to reference origin, then rotate to ENU
    dx = np.stack([x - x_ref, y - y_ref, z - z_ref], axis=-1)  # (N, 3)
    enu = dx @ R_enu.T  # (N, 3)

    return enu


def auto_utm_zone(lon: float) -> int:
    """Determine UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1


def wgs84_to_utm(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    utm_zone: Optional[int] = None,
) -> Tuple[np.ndarray, int, str]:
    """
    Convert WGS84 to UTM coordinates.

    Args:
        lat, lon, alt: WGS84 coordinates
        utm_zone: UTM zone (auto-detected from mean longitude if None)

    Returns:
        utm_coords: (N, 3) array of [easting, northing, altitude]
        zone_number: UTM zone used
        crs_wkt: WKT string of the UTM CRS (for metadata embedding)
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    alt = np.asarray(alt, dtype=np.float64)

    if utm_zone is None:
        utm_zone = auto_utm_zone(float(np.mean(lon)))

    # Determine hemisphere
    mean_lat = float(np.mean(lat))
    hemisphere = "north" if mean_lat >= 0 else "south"

    # Build UTM CRS
    crs_wgs84 = CRS.from_epsg(4326)
    epsg_utm = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
    crs_utm = CRS.from_epsg(epsg_utm)

    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
    easting, northing = transformer.transform(lon, lat)

    utm_coords = np.stack([easting, northing, alt], axis=-1)

    log.info(
        "UTM zone %d%s (EPSG:%d) — easting range: %.1f–%.1f, northing range: %.1f–%.1f",
        utm_zone,
        "N" if hemisphere == "north" else "S",
        epsg_utm,
        easting.min(), easting.max(),
        northing.min(), northing.max(),
    )

    return utm_coords, utm_zone, crs_utm.to_wkt()


# =============================================================================
# Umeyama Similarity Transform
# =============================================================================

def umeyama_similarity(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Compute the optimal similarity transform (scale + rotation + translation)
    that maps src points to dst points, using the Umeyama algorithm.

    Minimizes: sum_i || s * R @ src_i + t - dst_i ||^2

    This is the standard algorithm for registering VGGT's arbitrary-scale
    camera positions to real-world GPS (ENU) coordinates.

    Args:
        src: (N, 3) source points (VGGT camera positions)
        dst: (N, 3) destination points (GPS in ENU meters)

    Returns:
        s: Scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        rmse: Root mean square error after alignment

    Reference:
        S. Umeyama, "Least-squares estimation of transformation parameters
        between two point patterns," IEEE TPAMI, 1991.
    """
    assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"
    assert src.shape[0] >= 3, f"Need at least 3 points, got {src.shape[0]}"
    assert src.shape[1] == 3, f"Expected 3D points, got {src.shape[1]}D"

    n = src.shape[0]

    # Step 1: Compute centroids
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Step 2: Center the points
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # Step 3: Compute variances
    var_src = np.sum(src_centered ** 2) / n

    if var_src < 1e-10:
        raise ValueError(
            "Source points have near-zero variance — they may be coincident. "
            "Cannot compute a meaningful similarity transform."
        )

    # Step 4: Cross-covariance matrix
    cov = (dst_centered.T @ src_centered) / n  # (3, 3)

    # Step 5: SVD of cross-covariance
    U, D, Vt = np.linalg.svd(cov)

    # Step 6: Handle reflection (ensure proper rotation, det(R) = +1)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    # Step 7: Rotation
    R = U @ S @ Vt

    # Step 8: Scale
    s = np.trace(np.diag(D) @ S) / var_src

    # Step 9: Translation
    t = mu_dst - s * R @ mu_src

    # Step 10: Compute RMSE
    transformed = s * (src @ R.T) + t
    residuals = transformed - dst
    rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))

    log.info("Umeyama result — scale: %.6f, RMSE: %.4f m", s, rmse)

    return s, R, t, rmse


def apply_transform(
    points: np.ndarray,
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply a similarity transform: p' = s * R @ p + t

    Args:
        points: (N, 3) or (H, W, 3) array of 3D points
        s: Scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Returns:
        transformed: Same shape as input, with transform applied
    """
    original_shape = points.shape

    # Flatten to (N, 3) for transformation
    if points.ndim > 2:
        points_flat = points.reshape(-1, 3)
    else:
        points_flat = points

    # Apply: p' = s * R @ p + t
    transformed = s * (points_flat @ R.T) + t

    return transformed.reshape(original_shape)


# =============================================================================
# Alignment Quality Checks
# =============================================================================

def check_spatial_spread(
    points: np.ndarray,
    min_spread_meters: float = 10.0,
    min_points: int = 5,
) -> Tuple[bool, dict]:
    """
    Check that the GPS points are spatially well-distributed enough for
    a reliable Umeyama transform. Don't just check the count — also check
    that they span a reasonable spatial extent (not all clustered).

    Args:
        points: (N, 3) ENU coordinates
        min_spread_meters: Minimum spatial extent in any single axis
        min_points: Minimum number of points required

    Returns:
        is_valid: Whether the distribution passes validation
        info: Dict with diagnostic information
    """
    n = len(points)

    # Per-axis extent
    extent = points.max(axis=0) - points.min(axis=0)
    max_extent = float(extent.max())
    min_extent = float(extent.min())

    # Convex hull volume proxy: product of extents (rough)
    volume_proxy = float(np.prod(np.maximum(extent, 1e-6)))

    # Check if points are roughly collinear (bad for 3D alignment)
    # Use PCA — if smallest eigenvalue is very small relative to largest,
    # the points are roughly planar or collinear
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(n - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    # Condition ratio — if ratio is very high, points are degenerate
    condition_ratio = eigenvalues[0] / max(eigenvalues[-1], 1e-10)
    is_near_collinear = condition_ratio > 1000 and n < 20

    info = {
        "num_points": n,
        "extent_east_m": float(extent[0]),
        "extent_north_m": float(extent[1]),
        "extent_up_m": float(extent[2]),
        "max_extent_m": max_extent,
        "condition_ratio": float(condition_ratio),
        "is_near_collinear": bool(is_near_collinear),
    }

    # Validation
    issues = []
    if n < min_points:
        issues.append(f"Too few points: {n} < {min_points}")
    if max_extent < min_spread_meters:
        issues.append(
            f"Spatial spread too small: {max_extent:.1f}m < {min_spread_meters:.1f}m"
        )
    if is_near_collinear:
        issues.append(
            f"Points are nearly collinear (condition ratio: {condition_ratio:.0f})"
        )

    is_valid = len(issues) == 0
    info["is_valid"] = is_valid
    info["issues"] = issues

    if not is_valid:
        for issue in issues:
            log.warning("Alignment quality check: %s", issue)
    else:
        log.info(
            "Alignment quality OK: %d points, spread %.1f×%.1f×%.1f m",
            n, extent[0], extent[1], extent[2],
        )

    return is_valid, info
