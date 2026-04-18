"""
Q12: Homomorphic Filtering for illumination correction on runway.png.
     f(x,y) = i(x,y) * r(x,y)  →  log → FFT → filter → IFFT → exp
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class HomomorphicFiltering:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    @staticmethod
    def homomorphic_filter(img, sigma=30, gamma_L=0.3, gamma_H=1.5):
        """
        Homomorphic filtering pipeline:
        1. Log transform: separate illumination (low-freq) from reflectance (high-freq)
        2. FFT into frequency domain
        3. Apply high-emphasis Gaussian filter H(u,v):
               H(u,v) = (γ_H - γ_L) * [1 - exp(-D²/(2σ²))] + γ_L
           where D is distance from center. This boosts high frequencies (reflectance)
           and suppresses low frequencies (illumination).
        4. IFFT back to spatial domain
        5. Exp to reverse the log transform
        """
        # Step 1: log transform (add 1 to avoid log(0))
        img_log = np.log1p(img.astype(np.float64))

        # Step 2: FFT
        F = np.fft.fft2(img_log)
        F_shift = np.fft.fftshift(F)

        # Step 3: Build the high-emphasis filter
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        V, U = np.meshgrid(v, u)  # note: meshgrid transposes
        D2 = U**2 + V**2

        # High-emphasis Gaussian filter
        H = (gamma_H - gamma_L) * (1 - np.exp(-D2 / (2 * sigma**2))) + gamma_L

        # Step 4: Apply filter and IFFT
        F_filtered = F_shift * H
        F_back = np.fft.ifftshift(F_filtered)
        img_back = np.fft.ifft2(F_back).real

        # Step 5: Exp to invert log
        result = np.expm1(img_back)

        # Normalize to [0, 255]
        result = (result - result.min()) / (result.max() - result.min() + 1e-8) * 255
        return result.clip(0, 255).astype(np.uint8)

    def run(self):
        # Apply homomorphic filter
        filtered = self.homomorphic_filter(self.img, sigma=30, gamma_L=0.3, gamma_H=1.5)

        # Also show histogram equalized for comparison
        eq = cv2.equalizeHist(self.img)

        # Save
        cv2.imwrite(os.path.join(self.output_dir, "q12_homomorphic.png"), filtered)

        # Comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(
            axes,
            [self.img, filtered, eq],
            ["Original (runway.png)",
             "Homomorphic Filter\n(σ=30, γ_L=0.3, γ_H=1.5)",
             "Histogram Equalization\n(for comparison)"]
        ):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Q12 – Homomorphic Filtering vs Histogram Equalization", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q12_homomorphic_results.png"), dpi=150)
        plt.close()

        # Theory text
        theory = """\
Q12 Theoretical Answers
=======================

(a) Multiplicative Model f(x,y) = i(x,y) · r(x,y):
    - i(x,y) is illumination: slowly varying, large-area lighting gradients (low-freq)
    - r(x,y) is reflectance: the actual surface properties, edges, texture (high-freq)
    - This model is physically motivated: the camera captures light reflected from a surface,
      so the two components multiply. Separating them allows correcting one without
      destroying the other.

(b) Log transformation separates the two:
    log f(x,y) = log i(x,y) + log r(x,y)
    Multiplication becomes addition in the log domain. We can then use linear filtering
    to suppress log i (low-freq) and boost log r (high-freq).

(c) Algorithm:
    1. Log: z(x,y) = log(f(x,y) + 1)
    2. FFT: Z(u,v) = F{z(x,y)}
    3. Filter: S(u,v) = H(u,v) · Z(u,v)  with high-emphasis Gaussian H
    4. IFFT: s(x,y) = F^{-1}{S(u,v)}
    5. Exp: output = exp(s(x,y)) − 1

(d) Homomorphic vs Histogram Equalization:
    - Histogram equalization redistributes global intensities but cannot separate
      illumination and reflectance – it may over-enhance or introduce artifacts.
    - Homomorphic filtering physically separates lighting from surface detail.
    - Preferred when: illumination is strongly non-uniform (e.g., face lit from one side,
      nighttime scenes), or when preserving natural appearance of reflectance is important.

(e) Comments on runway.png result:
    - Illumination: the runway lighting gradients are reduced, giving a more uniform look.
    - Contrast: details in dark shadow areas become more visible.
    - Artifacts: possible slight halo effect near strong edges due to ringing of the
      Gaussian filter near high-contrast transitions.
"""
        with open(os.path.join(self.output_dir, "q12_theory.txt"), "w") as f:
            f.write(theory)
        print(theory)
        print("  [Q12] Saved: q12_homomorphic.png, q12_homomorphic_results.png, q12_theory.txt")
