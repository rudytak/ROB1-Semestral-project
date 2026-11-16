#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa
import matplotlib.pyplot as plt  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """

    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    # todo HW03: Detect circle in each image
    circle_color_samples = [
        (48, 84, 111),
        (52, 78, 96),
        (53, 82, 112),
        (82, 104, 135),
        (88, 105, 117),
        (83, 111, 137)
    ]
    target_color_rgb = np.mean(circle_color_samples, axis=0).astype(np.uint8)
    target_color = cv2.cvtColor(target_color_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3,)

    max_hue_dist = 40
    print(f"Target color in HSV: {target_color}, max hue distance: {max_hue_dist}")
    lower = np.array([target_color[0] - max_hue_dist, 50, 50])
    upper = np.array([target_color[0] + max_hue_dist, 255, 255])
    
    min_circle_area = 300
    
    centers = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        
        contours_all, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = [c for c in contours_all if cv2.contourArea(c) > min_circle_area]
        print(f"Found {len(contours_all)} contours in the image of which {len(contours)} are larger than {min_circle_area} pixels")
        
        # find the most circular contour
        f = lambda c: abs(4*np.pi*cv2.contourArea(c) / (cv2.arcLength(c, True)**2 + 1e-6) - 1)
        best = min(contours, key=f)
        
        # cv2.drawContours(img, [best], -1, (255, 0, 0), 1)
        # plt.figure(figsize=(9, 6))
        # plt.imshow(img)
        # plt.tight_layout()
        # plt.show()
        
        center = np.mean(best.reshape(-1, 2), axis=0)
        centers.append(center)
    centers = np.array(centers).reshape(-1, 2)
    
    print(f"Detected centers: {centers}")
    
    hoop_centers = []
    for pos in hoop_positions:
        hoop_pos = np.array(pos["translation_vector"])[:2]
        hoop_centers.append(hoop_pos)
    hoop_centers = np.array(hoop_centers).reshape(-1, 2)
    
    print(f"Hoop centers: {hoop_centers}")
    
    # todo HW03: Find homography using cv2.findHomography. Use the hoop positions and circle centers.
    
    H, _ = cv2.findHomography(centers, hoop_centers, method=0)

    return H
