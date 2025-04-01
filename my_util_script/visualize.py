import cv2
import numpy as np

def show_mask_partial(mask, image, color=None, alpha=0.45):
    if color is None:
        color = np.random.randint(0, 255, 3)

    overlay = np.zeros_like(image)
    for i in range(3):
        overlay[:, :, i] = color[i]

    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return combined

def show_mask(mask, image, colors=None, alpha=0.45):
    '''
    mask: [H, W]
    image: [H, W, 3]
    colors: list oflist of 3 in [0, 255]
    require len(colors) == len(np.unique(mask)) - 1
    alpha: float in [0, 1]
    '''
    label_vals = np.unique(mask)
    label_vals = label_vals[label_vals > 0]

    if colors is None:
        label_cols = [
            np.random.randint(0, 255, 3)
            for _ in range(len(label_vals))
        ]
    else:
        assert len(colors) == len(label_vals)
        label_cols = colors

    overlay = np.zeros_like(image)
    for i in range(len(label_vals)):
        overlay_i = np.zeros_like(image)
        overlay_i[mask == label_vals[i], :] = image[mask == label_vals[i], :]
        overlay_i = show_mask_partial(
            (mask == label_vals[i]).astype(np.uint8),
            overlay_i,
            color=label_cols[i]
        )
        overlay[mask == label_vals[i], :] = overlay_i[mask == label_vals[i], :]
    
    overlay[mask == 0, :] = image[mask == 0, :]

    return overlay