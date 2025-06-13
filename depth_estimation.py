"""
Minimal bounding-box-based depth estimation module.

Provides a single function to compute object distance from bounding-box pixel width,
using the pinhole camera model.

Constants:
    FOCAL_LENGTH: focal length in pixel units (calibrated)
    REAL_WIDTH: real-world object width in meters (e.g. average human shoulder width)

Function:
    estimate_distance_bbox(px_width, focal_length, real_width) -> float
        Compute distance in meters. Returns None if px_width <= 0.
"""

def estimate_distance_bbox(
    bbox_width_px: int,
    focal_length: float = 580.0,
    real_width: float = 0.45,
) -> float:
    """
    Estimate distance to an object based on its bounding-box width in pixels.

    Args:
        bbox_width_px: width of the bounding box in pixels
        focal_length: camera focal length in pixel units
        real_width: real-world width of the object in meters

    Returns:
        Estimated distance in meters, or raises ValueError if bbox_width_px <= 0
    """
    if bbox_width_px <= 0:
        raise ValueError(f"Invalid bbox width: {bbox_width_px}. Must be > 0.")
    # distance = (real_width * focal_length) / pixel_width
    return (real_width * focal_length) / bbox_width_px
