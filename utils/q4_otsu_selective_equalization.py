"""
Q4: Otsu thresholding + selective histogram equalization.
    (a) Otsu binary mask – foreground (woman + room)
    (b) Equalize only the foreground region
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class OtsuSelectiveEqualization:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.img_gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)

    def run(self):
        # ── (a) Otsu thresholding ─────────────────────────────────────────────
        # Automatically finds the optimal threshold that separates the bright
        # overexposed window (background) from the dark woman + room (foreground).
        thresh_val, binary_mask = cv2.threshold(
            self.img_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"  [Q4] Otsu threshold value: {thresh_val:.1f}")

        # binary_mask == 255  →  bright pixels  →  window (background)
        # binary_mask == 0    →  dark pixels    →  woman + room (foreground)
        fg_mask = (binary_mask == 0)   # True where pixel belongs to foreground

        # Save Otsu binary mask
        cv2.imwrite(os.path.join(self.output_dir, "q4a_otsu_mask.png"), binary_mask)

        # ── (b) Selective histogram equalization (foreground only) ────────────
        result = self.img_gray.copy()
        fg_pixels = self.img_gray[fg_mask]

        # Build histogram over foreground pixels only (full 0–255 range)
        H, _ = np.histogram(fg_pixels, bins=256, range=(0, 256))
        cdf = np.cumsum(H)

        # Standard equalization formula: scale CDF to [0, 255]
        cdf_min = int(cdf[cdf > 0].min())
        N = int(fg_pixels.size)
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = round((int(cdf[i]) - cdf_min) / max(N - cdf_min, 1) * 255)

        # Apply LUT exclusively to foreground pixels; background unchanged
        result[fg_mask] = lut[fg_pixels]

        # Save equalized result
        cv2.imwrite(os.path.join(self.output_dir, "q4b_selective_eq.png"), result)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(self.img_gray, cmap="gray")
        axes[0].set_title("Original (Grayscale)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(binary_mask, cmap="gray")
        axes[1].set_title(
            f"Otsu Mask (thresh={thresh_val:.0f})\nWhite=window | Black=woman+room",
            fontsize=9
        )
        axes[1].axis("off")

        axes[2].imshow(fg_mask.astype(np.uint8) * 255, cmap="gray")
        axes[2].set_title("Foreground Mask\n(White = woman + room)", fontsize=9)
        axes[2].axis("off")

        axes[3].imshow(result, cmap="gray")
        axes[3].set_title("Selective Equalization\n(Foreground Only)", fontsize=9)
        axes[3].axis("off")

        plt.suptitle("Q4 – Otsu Selective Histogram Equalization", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q4_otsu_results.png"), dpi=150)
        plt.close()

        print("  [Q4] Saved: q4a_otsu_mask.png, q4b_selective_eq.png, q4_otsu_results.png")
