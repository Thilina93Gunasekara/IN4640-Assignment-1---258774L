"""
Q1: Intensity Transformations
- Gamma correction (γ=0.5, γ=2)
- Contrast stretching (r1=0.2, r2=0.8)

Applied to runway image. Gamma < 1 brightens darker regions (useful for
underexposed aerial imagery), while gamma > 1 compresses bright highlights.
Contrast stretching clamps intensities outside [r1, r2] to expand mid-range contrast.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class IntensityTransformations:
    def __init__(self, image_path: str, output_dir: str = "./outputs"):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.image = img

    def gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction: s = r^gamma (normalized to [0,1]).
        gamma < 1 -> brightens; gamma > 1 -> darkens.
        """
        normalized = img.astype(np.float64) / 255.0
        corrected = np.power(normalized, gamma)
        return np.clip(corrected * 255, 0, 255).astype(np.uint8)

    def contrast_stretch(self, img: np.ndarray, r1: float = 0.2, r2: float = 0.8) -> np.ndarray:
        """
        Piecewise linear contrast stretching:
            s = 0           if r < r1
            s = (r-r1)/(r2-r1)  if r1 <= r <= r2
            s = 1           if r > r2
        All values normalized to [0,1] before applying.
        """
        r = img.astype(np.float64) / 255.0
        s = np.where(r < r1, 0.0,
             np.where(r > r2, 1.0,
                      (r - r1) / (r2 - r1)))
        return np.clip(s * 255, 0, 255).astype(np.uint8)

    def run(self):
        img = self.image

        gamma_05 = self.gamma_correction(img, gamma=0.5)
        gamma_2  = self.gamma_correction(img, gamma=2.0)
        stretched = self.contrast_stretch(img, r1=0.2, r2=0.8)

        # ---- Plot ----
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle("Q1: Intensity Transformations (Runway Image)", fontsize=14, fontweight='bold')

        images = [img, gamma_05, gamma_2, stretched]
        titles = ["Original", "Gamma γ=0.5\n(Brighter)", "Gamma γ=2\n(Darker)", "Contrast Stretch\nr1=0.2, r2=0.8"]

        for ax, im, title in zip(axes, images, titles):
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "q1_transformations.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Q1] Saved → {out_path}")

        # Print mean brightness values for observation reporting
        print(f"  Mean brightness — Original: {img.mean():.1f} | "
              f"γ=0.5: {gamma_05.mean():.1f} | "
              f"γ=2: {gamma_2.mean():.1f} | "
              f"Stretched: {stretched.mean():.1f}")

        return {"original": img, "gamma_05": gamma_05, "gamma_2": gamma_2, "stretched": stretched}
