"""
Minimal DeepSORT-based human tracking module.

Provides simple functions to initialize a DeepSort tracker and update it
with new detections each frame. No I/O, display, or Spot SDK code included.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Default DeepSORT configuration
DEFAULT_CONFIG = {
    'max_age': 60,                # Keep tracks alive for 60 frames
    'n_init': 3,                  # Require 3 hits to confirm new tracks
    'nn_budget': 100,             # Store up to 100 embeddings per track
    'max_cosine_distance': 0.25,  # Stricter appearance matching
    'override_track_class': None, # No class override
    'embedder': 'mobilenet',      # Embedder model
    'half': False,                # Full precision embeddings
    'bgr': True,
    'embedder_gpu': False,
}


def init_tracker(config: dict = None) -> DeepSort:
    """
    Initialize and return a DeepSort tracker instance.

    Args:
        config: Optional override for default config keys:
            max_age, n_init, nn_budget,
            max_cosine_distance, override_track_class,
            embedder, half, bgr, embedder_gpu
    Returns:
        DeepSort tracker object
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
    tracker = DeepSort(
        max_age=cfg['max_age'],
        n_init=cfg['n_init'],
        nn_budget=cfg['nn_budget'],
        max_cosine_distance=cfg['max_cosine_distance'],
        override_track_class=cfg['override_track_class'],
        embedder=cfg['embedder'],
        half=cfg['half'],
        bgr=cfg['bgr'],
        embedder_gpu=cfg['embedder_gpu'],
    )
    return tracker


def update_tracks(
    tracker: DeepSort,
    raw_detections: list,
    frame: np.ndarray = None,
) -> list:
    """
    Update tracker with new detections for the current frame.

    Args:
        tracker: DeepSort instance
        raw_detections: List of detections in format [
            ([x, y, w, h], confidence, class_name)
        ]
        frame: Optional frame array (for appearance features)

    Returns:
        List of confirmed tracks, each as dict:
            {'track_id': int, 'bbox': (x1, y1, x2, y2)}
    """
    # Run the tracker update
    tracks = tracker.update_tracks(raw_detections, frame=frame)
    results = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        # Convert LTRB to tuple of ints
        l, t, r, b = track.to_ltrb()
        bbox = (int(l), int(t), int(r), int(b))
        results.append({
            'track_id': track.track_id,
            'bbox': bbox,
        })
    return results
