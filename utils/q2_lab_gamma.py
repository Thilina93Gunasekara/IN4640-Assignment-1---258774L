"""
Q2: Gamma correction in L*a*b* color space on sapphire.jpg.
    (a) Apply gamma to the L (luminance) channel – choose γ that best enhances it.
    (b) Show histograms of original and corrected images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class LABGammaCorrection:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

    def run(self):
        # Convert RGB → L*a*b* (OpenCV expects uint8 in range 0-255)
        img_lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

        L = img_lab[:, :, 0]  # L channel in OpenCV is [0, 255]
        a = img_lab[:, :, 1]
        b = img_lab[:, :, 2]

        # Normalize L to [0,1], apply gamma, scale back
        # γ = 0.45 brightens the dark sapphire gemstone (underexposed illumination)
        gamma = 0.45
        L_norm = L / 255.0
        L_corrected = np.power(L_norm, gamma) * 255.0

        # Reconstruct LAB image and convert back to BGR
        corrected_lab = np.stack([L_corrected, a, b], axis=2).clip(0, 255).astype(np.uint8)
        corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)

        # Save corrected image
        cv2.imwrite(os.path.join(self.output_dir, "q2_lab_gamma_corrected.png"), corrected_bgr)

        # --- Figure 1: Side-by-side comparison ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(self.img_rgb)
        axes[0].set_title("Original (sapphire.jpg)", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(corrected_rgb)
        axes[1].set_title(f"Gamma corrected on L* channel (γ = {gamma})", fontsize=11)
        axes[1].axis("off")
        plt.suptitle("Q2 – LAB Gamma Correction", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q2_comparison.png"), dpi=150)
        plt.close()

        # --- Figure 2: Histograms ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Histogram of original (use L channel, normalized to 0-255 scale)
        axes[0].hist(L.ravel(), bins=256, range=(0, 255), color='steelblue', alpha=0.8)
        axes[0].set_title("Histogram of L* (Original)", fontsize=11)
        axes[0].set_xlabel("L* value")
        axes[0].set_ylabel("Pixel Count")

        axes[1].hist(L_corrected.ravel(), bins=256, range=(0, 255), color='tomato', alpha=0.8)
        axes[1].set_title(f"Histogram of L* (γ = {gamma})", fontsize=11)
        axes[1].set_xlabel("L* value")
        axes[1].set_ylabel("Pixel Count")

        plt.suptitle("Q2 – Luminance Histograms Before and After Gamma Correction", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q2_histograms.png"), dpi=150)
        plt.close()

        print(f"  [Q2] Applied gamma={gamma} on L* channel. Saved: q2_lab_gamma_corrected.png, q2_comparison.png, q2_histograms.png")
