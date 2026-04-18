"""
Q9: Image sharpening on emma.jpg (image of choice).
    Two techniques:
      (a) Unsharp Masking  – sharpened = original + k * (original − blurred)
      (b) Laplacian Sharpening – sharpened = original − k * Laplacian(original)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ImageSharpening:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    def run(self):
        # ── (a) Unsharp Masking ───────────────────────────────────────────────
        # Step 1: blur to extract low-frequency (smooth) component
        blurred = cv2.GaussianBlur(self.img, (5, 5), sigmaX=1.0)

        # Step 2: high-frequency detail = original − blurred
        high_freq = self.img.astype(np.float64) - blurred.astype(np.float64)

        # Step 3: add detail back scaled by factor k
        k = 1.5  # sharpening strength
        sharpened = np.clip(
            self.img.astype(np.float64) + k * high_freq, 0, 255
        ).astype(np.uint8)

        # ── (b) Laplacian Sharpening ──────────────────────────────────────────
        # Laplacian captures second-order edges (fine detail / transitions)
        # Subtract it to enhance edges without boosting noise as much
        laplacian = cv2.Laplacian(self.img, cv2.CV_64F)
        lap_sharp = np.clip(
            self.img.astype(np.float64) - 0.5 * laplacian, 0, 255
        ).astype(np.uint8)

        # Save outputs
        cv2.imwrite(os.path.join(self.output_dir, "q9_unsharp_mask.png"),    sharpened)
        cv2.imwrite(os.path.join(self.output_dir, "q9_laplacian_sharp.png"), lap_sharp)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        images = [self.img, sharpened, lap_sharp]
        titles = [
            "Original (emma.jpg)",
            "Unsharp Masking (k=1.5)",
            "Laplacian Sharpening (k=0.5)"
        ]
        for ax, im, title in zip(axes, images, titles):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.suptitle("Q9 – Image Sharpening (emma.jpg)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q9_sharpening_results.png"), dpi=150)
        plt.close()

        print("  [Q9] Saved: q9_unsharp_mask.png, q9_laplacian_sharp.png, q9_sharpening_results.png")