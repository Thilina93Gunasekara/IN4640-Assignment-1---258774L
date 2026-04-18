"""
Q1: Intensity Transformations on the runway image.
    (a) Gamma correction γ=0.5
    (b) Gamma correction γ=2
    (c) Contrast stretching with r1=0.2, r2=0.8
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class IntensityTransformations:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        # Load as grayscale – runway.png is used for Q1
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        # Normalize to [0,1] float for processing
        self.img_norm = self.img.astype(np.float64) / 255.0

    def gamma_correction(self, gamma):
        """Apply s = r^gamma (power-law / gamma transform)."""
        corrected = np.power(self.img_norm, gamma)
        # Back to uint8
        return (corrected * 255).clip(0, 255).astype(np.uint8)

    def contrast_stretching(self, r1=0.2, r2=0.8):
        """
        Piecewise linear contrast stretch:
          s = 0              if r < r1
          s = (r-r1)/(r2-r1) if r1 <= r <= r2
          s = 1              if r > r2
        """
        r = self.img_norm.copy()
        s = np.zeros_like(r)
        # Middle segment
        mid = (r >= r1) & (r <= r2)
        s[mid] = (r[mid] - r1) / (r2 - r1)
        # High end
        s[r > r2] = 1.0
        return (s * 255).clip(0, 255).astype(np.uint8)

    def run(self):
        # (a) gamma = 0.5 (brightens dark regions)
        g05 = self.gamma_correction(0.5)
        # (b) gamma = 2 (darkens / enhances bright regions)
        g2 = self.gamma_correction(2.0)
        # (c) contrast stretching
        cs = self.contrast_stretching(r1=0.2, r2=0.8)

        # Save individual outputs
        cv2.imwrite(os.path.join(self.output_dir, "q1a_gamma_0.5.png"), g05)
        cv2.imwrite(os.path.join(self.output_dir, "q1b_gamma_2.0.png"), g2)
        cv2.imwrite(os.path.join(self.output_dir, "q1c_contrast_stretch.png"), cs)

        # Composite figure for report
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Original", "γ = 0.5", "γ = 2.0", "Contrast Stretch\n(r1=0.2, r2=0.8)"]
        images = [self.img, g05, g2, cs]
        for ax, im, title in zip(axes, images, titles):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Q1 – Intensity Transformations (runway.png)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q1_all_transforms.png"), dpi=150)
        plt.close()
        print("  [Q1] Saved: q1a_gamma_0.5.png, q1b_gamma_2.0.png, q1c_contrast_stretch.png, q1_all_transforms.png")
