"""
Q8: Noise removal on emma.jpg (salt & pepper noise).
    (a) Gaussian smoothing
    (b) Median filtering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class NoiseRemoval:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    def run(self):
        # (a) Gaussian smoothing – smooths noise but may blur edges
        # kernel size 5x5, sigma=1.5 gives a reasonable smoothing
        gaussian = cv2.GaussianBlur(self.img, (5, 5), sigmaX=1.5)

        # (b) Median filtering – excellent for salt & pepper noise
        # because it replaces each pixel with the median of neighbors
        # (outlier salt/pepper values get discarded)
        median = cv2.medianBlur(self.img, 5)

        # Save
        cv2.imwrite(os.path.join(self.output_dir, "q8a_gaussian_denoised.png"), gaussian)
        cv2.imwrite(os.path.join(self.output_dir, "q8b_median_denoised.png"), median)

        # Comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(
            axes,
            [self.img, gaussian, median],
            ["Original (Salt & Pepper Noise)", "Gaussian Filter (5×5, σ=1.5)",
             "Median Filter (5×5)"]
        ):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Q8 – Salt & Pepper Noise Removal (emma.jpg)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q8_noise_removal.png"), dpi=150)
        plt.close()

        print("  [Q8] Saved: q8a_gaussian_denoised.png, q8b_median_denoised.png, q8_noise_removal.png")
