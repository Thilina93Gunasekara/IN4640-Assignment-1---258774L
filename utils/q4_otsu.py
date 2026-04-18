"""
Q4: Otsu thresholding + selective histogram equalization on highlights_and_shadows.jpg.
    (a) Otsu binary mask – foreground (woman + room)
    (b) Equalize only the foreground pixels
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
        # (a) Otsu thresholding – automatically finds optimal threshold
        thresh_val, binary_mask = cv2.threshold(
            self.img_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"  [Q4] Otsu threshold value: {thresh_val:.1f}")

        # Save binary mask
        cv2.imwrite(os.path.join(self.output_dir, "q4a_otsu_mask.png"), binary_mask)

        # (b) Selective histogram equalization – only foreground pixels
        # We define foreground as the brighter region (woman+room, non-window area)
        # Otsu separates bright foreground from the overexposed bright window
        # The mask keeps pixels BELOW threshold (darker indoor region)
        fg_mask = (binary_mask == 0).astype(np.uint8)  # foreground = darker interior

        result = self.img_gray.copy()
        fg_pixels = result[fg_mask == 1]

        # Build histogram for foreground pixels only
        H, _ = np.histogram(fg_pixels, bins=256, range=(0, 255))
        cdf = np.cumsum(H)
        cdf_min = cdf[cdf > 0].min()
        N = fg_pixels.size
        lut = np.round((cdf - cdf_min) / max(N - cdf_min, 1) * 255).astype(np.uint8)

        # Apply LUT only to foreground
        result[fg_mask == 1] = lut[fg_pixels]

        # Save
        cv2.imwrite(os.path.join(self.output_dir, "q4b_selective_eq.png"), result)

        # --- Figure ---
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(self.img_gray, cmap="gray")
        axes[0].set_title("Original (Grayscale)", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(binary_mask, cmap="gray")
        axes[1].set_title(f"Otsu Mask (thresh={thresh_val:.0f})", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(fg_mask * 255, cmap="gray")
        axes[2].set_title("Foreground Mask", fontsize=10)
        axes[2].axis("off")

        axes[3].imshow(result, cmap="gray")
        axes[3].set_title("Selective Equalization\n(Foreground Only)", fontsize=10)
        axes[3].axis("off")

        plt.suptitle("Q4 – Otsu Selective Histogram Equalization", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q4_otsu_results.png"), dpi=150)
        plt.close()

        print("  [Q4] Saved: q4a_otsu_mask.png, q4b_selective_eq.png, q4_otsu_results.png")
