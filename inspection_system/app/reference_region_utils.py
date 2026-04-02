from __future__ import annotations


def build_reference_regions(reference_mask, inspection_cfg: dict, dilate_mask, erode_mask):
    allowed_iters = int(inspection_cfg.get("allowed_dilate_iterations", 2))
    required_iters = int(inspection_cfg.get("required_erode_iterations", 1))
    allowed_mask = dilate_mask(reference_mask, allowed_iters)
    required_mask = erode_mask(reference_mask, required_iters)
    return allowed_mask, required_mask