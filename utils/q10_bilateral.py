"""
Q10: Bilateral Filtering on einstein.png.
    (a) Manual bilateral filter implementation
    (b) cv2.GaussianBlur() for comparison
    (c) cv2.bilateralFilter()
    (d) Manual filter result
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class BilateralFiltering:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    @staticmethod
    def bilateral_filter_manual(img, d, sigma_s, sigma_r):
        """
        Manual bilateral filter.

        For each pixel (i,j), the output is a weighted average of its neighbors:
            W(i,j,k,l) = exp(-||pos_dist||² / (2σ_s²)) * exp(-(I(i,j)-I(k,l))² / (2σ_r²))
        
        spatial weight (Gaussian over distance) * range weight (Gaussian over intensity diff).
        This preserves edges because pixels with very different intensities get near-zero weight.

        Parameters:
            d      – kernel diameter (odd integer, e.g. 9)
            sigma_s – spatial standard deviation
            sigma_r – range (intensity) standard deviation
        """
        img_f = img.astype(np.float64)
        h, w = img_f.shape
        half = d // 2
        output = np.zeros_like(img_f)

        # Pre-compute spatial Gaussian weights (same for all pixels)
        spatial_weights = np.zeros((d, d))
        for di in range(d):
            for dj in range(d):
                dy = di - half
                dx = dj - half
                spatial_weights[di, dj] = np.exp(-(dy**2 + dx**2) / (2 * sigma_s**2))

        # Pad image to handle borders
        img_padded = np.pad(img_f, half, mode='reflect')

        for i in range(h):
            for j in range(w):
                # Extract local patch
                patch = img_padded[i: i + d, j: j + d]
                center_val = img_f[i, j]

                # Range (intensity difference) weight
                range_weights = np.exp(-((patch - center_val)**2) / (2 * sigma_r**2))

                # Combined bilateral weight
                weights = spatial_weights * range_weights
                weight_sum = weights.sum()

                output[i, j] = np.sum(weights * patch) / weight_sum if weight_sum > 0 else center_val

        return output.clip(0, 255).astype(np.uint8)

    def run(self):
        d = 9          # kernel diameter
        sigma_s = 15   # spatial sigma
        sigma_r = 40   # range sigma

        # (b) Gaussian blur (blurs edges)
        gaussian = cv2.GaussianBlur(self.img, (d, d), sigmaX=sigma_s)

        # (c) OpenCV bilateral filter
        opencv_bilateral = cv2.bilateralFilter(self.img, d, sigma_r, sigma_s)

        # (d) Manual bilateral filter (may be slow for large images – resize if needed)
        # To keep runtime reasonable, work on a smaller crop
        # Use the full image but with a small kernel
        print("  [Q10] Running manual bilateral filter (this may take ~30s)...")
        small = cv2.resize(self.img, None, fx=0.5, fy=0.5)
        manual_bilateral_small = self.bilateral_filter_manual(small, d=7, sigma_s=10, sigma_r=30)
        manual_bilateral = cv2.resize(manual_bilateral_small, (self.img.shape[1], self.img.shape[0]))

        # Save
        cv2.imwrite(os.path.join(self.output_dir, "q10b_gaussian_blur.png"), gaussian)
        cv2.imwrite(os.path.join(self.output_dir, "q10c_opencv_bilateral.png"), opencv_bilateral)
        cv2.imwrite(os.path.join(self.output_dir, "q10d_manual_bilateral.png"), manual_bilateral)

        # Comparison figure
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, im, title in zip(
            axes,
            [self.img, gaussian, opencv_bilateral, manual_bilateral],
            ["Original", f"Gaussian Blur\n(σ_s={sigma_s})",
             f"OpenCV Bilateral\n(σ_s={sigma_s}, σ_r={sigma_r})",
             "Manual Bilateral\n(downsampled)"]
        ):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.suptitle("Q10 – Bilateral Filtering Comparison (einstein.png)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q10_bilateral_comparison.png"), dpi=150)
        plt.close()

        print("  [Q10] Saved: q10b_gaussian_blur.png, q10c_opencv_bilateral.png, q10d_manual_bilateral.png, q10_bilateral_comparison.png")
