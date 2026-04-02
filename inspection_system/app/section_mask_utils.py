from __future__ import annotations


def compute_section_masks(required_mask, section_columns: int, cv2, np):
    white = (required_mask > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)

    sections = []
    for label_id in range(1, stats.shape[0]):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        component = np.zeros_like(required_mask, dtype=np.uint8)
        component[labels == label_id] = 255

        splits = min(section_columns, max(1, w // 8))
        step = max(1, w // splits)

        sx = x
        while sx < x + w:
            ex = min(x + w, sx + step)
            section = np.zeros_like(required_mask, dtype=np.uint8)
            section[y:y + h, sx:ex] = component[y:y + h, sx:ex]
            if (section > 0).sum() > 0:
                sections.append(section)
            sx = ex

    return sections