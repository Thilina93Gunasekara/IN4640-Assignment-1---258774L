"""
Q3: Manual histogram equalization on runway.png.
    Implements the full equalization pipeline from scratch (no cv2.equalizeHist).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ManualHistogramEqualization:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    def equalize(self, img):
        """
        Manual histogram equalization.
        Steps:
          1. Compute histogram H[k] for k in [0..255]
          2. Compute CDF (cumulative distribution function)
          3. Normalize CDF to [0..255]
          4. Apply the mapping to the image
        """
        H, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))

        # CDF – cumulative sum of histogram
        cdf = np.cumsum(H)

        # Normalize using CDF min-value method to avoid black pixels
        cdf_min = cdf[cdf > 0].min()
        N = img.size  # total number of pixels

        # Equalization formula: s(k) = round((cdf(k) - cdf_min) / (N - cdf_min) * 255)
        cdf_eq = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)

        # Map original pixel values through the lookup table
        equalized = cdf_eq[img]
        return equalized, H, cdf_eq

    def run(self):
        eq_img, hist_orig, lut = self.equalize(self.img)

        # Also compute equalized histogram for display
        H_eq, _ = np.histogram(eq_img.ravel(), bins=256, range=(0, 255))

        # Save equalized image
        cv2.imwrite(os.path.join(self.output_dir, "q3_hist_equalized.png"), eq_img)

        # --- Figure: Original vs Equalized + Histograms ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Original Image", fontsize=11)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(eq_img, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Histogram Equalized", fontsize=11)
        axes[0, 1].axis("off")

        axes[1, 0].bar(range(256), hist_orig, color='steelblue', width=1)
        axes[1, 0].set_title("Histogram – Original", fontsize=11)
        axes[1, 0].set_xlabel("Intensity")
        axes[1, 0].set_ylabel("Pixel Count")

        axes[1, 1].bar(range(256), H_eq, color='tomato', width=1)
        axes[1, 1].set_title("Histogram – After Equalization", fontsize=11)
        axes[1, 1].set_xlabel("Intensity")
        axes[1, 1].set_ylabel("Pixel Count")

        plt.suptitle("Q3 – Manual Histogram Equalization (runway.png)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q3_hist_eq_results.png"), dpi=150)
        plt.close()

        print("  [Q3] Saved: q3_hist_equalized.png, q3_hist_eq_results.png")
