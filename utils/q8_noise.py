"""
Q8: Salt & Pepper noise corruption + removal on emma.jpg.
    (a) Apply Gaussian smoothing to the noisy image
    (b) Apply median filtering to the noisy image
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

    @staticmethod
    def _add_salt_pepper(image, amount=0.15, seed=42):
        """Corrupt image with salt (255) and pepper (0) noise."""
        np.random.seed(seed)
        noisy = image.copy()
        total = image.size
        n_salt   = int(total * amount / 2)
        n_pepper = int(total * amount / 2)

        # Salt – set random pixels to 255
        r, c = np.random.randint(0, image.shape[0], n_salt), \
               np.random.randint(0, image.shape[1], n_salt)
        noisy[r, c] = 255

        # Pepper – set random pixels to 0
        r, c = np.random.randint(0, image.shape[0], n_pepper), \
               np.random.randint(0, image.shape[1], n_pepper)
        noisy[r, c] = 0

        return noisy

    def run(self):
        # Corrupt the clean image with salt & pepper noise (15%)
        noisy = self._add_salt_pepper(self.img, amount=0.15)

        # ── (a) Gaussian smoothing ────────────────────────────────────────────
        # Blurs the whole image; reduces noise but also blurs edges
        gaussian = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.0)

        # ── (b) Median filtering ──────────────────────────────────────────────
        # Replaces each pixel with the median of its neighbourhood;
        # highly effective for salt & pepper noise while preserving edges
        median = cv2.medianBlur(noisy, 5)

        # Save outputs
        cv2.imwrite(os.path.join(self.output_dir, "q8_noisy.png"),       noisy)
        cv2.imwrite(os.path.join(self.output_dir, "q8a_gaussian.png"),   gaussian)
        cv2.imwrite(os.path.join(self.output_dir, "q8b_median.png"),     median)

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        images = [self.img, noisy, gaussian, median]
        titles = [
            "Original",
            "Salt & Pepper Noise\n(15%)",
            "Gaussian Smoothing\n(5×5, σ=1.0)",
            "Median Filtering\n(5×5)"
        ]
        for ax, im, title in zip(axes, images, titles):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.suptitle("Q8 – Salt & Pepper Noise Removal", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q8_noise_results.png"), dpi=150)
        plt.close()

        print("  [Q8] Saved: q8_noisy.png, q8a_gaussian.png, q8b_median.png, q8_noise_results.png")