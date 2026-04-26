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

a)	Multiplicative Model f(x,y) = i(x,y).r(x,y) 
•	i(x,y) represents illumination, which is defined as area lighting gradient (varies slowly) with low frequency content over large areas. 
•	r(x,y) is the reflectance value of the surface, containing both surface properties (texture/edges) and other types of detail (i.e., high frequency). 
•	This is a physically motivated model because the camera only senses what is reflected from the surface, so these two components multiply each other. When we separate the two components, we can modify one of them without affecting the other. 

b)	Use of logarithmic transformation to separate the two: log(f(x,y)) = log(i(x,y)) + log(r(x,y)) 
•	When you take the logarithm off, you convert the multiplication of the two components into addition in the logarithm domain. Thus, if we take a linear filter and low-pass filter the log(i) (i.e., low frequency) and high-pass filter the log(r) (i.e., high frequency), we can manipulate both components independently. 
c)	Procedure 
•	Log: z(x,y) = log(f(x,y)+1) 
•	FFT: Z(u,v) = F{z(x,y)} 
•	Filter: S(u,v) = H(u,v).Z(u,v) , Where H is a gaussian filter that has a large high emphasis region.
•	IFFT: s(x,y) = F^{-1}{S(u,v)} 
•	Exp: Output = exp(s(x,y)) - 1.
d)	Homomorphic vs Histogram Equalization
Histogram equalization is a method that will redistribute counts of global intensities across a histogram but will not separate illumination and reflectance - meaning it may create over-enhanced values or artifacts in some cases. On the other hand, homomorphic filtering allows for the physical separation of illumination from surface detail. Homomorphic filtering is preferred for use in instances where the illumination is significantly non-uniform (example: face being lit by only one side, outdoor scenes at night) or when reflecting the natural appearance of reflectance is important.
e)	Evaluating Results from runway.png
Illumination - The gradients of the runway lighting have been reduced to provide a more uniform look across the image.
•	Contrast - Detail in the darker shadow areas have improved visibility.
•	Artifacts - There may be a slight halo effect created by the ringing of the Gaussian filter near any strong edge due to the transition from high to low contrast.

"""
        with open(os.path.join(self.output_dir, "q12_theory.txt"), "w") as f:
            f.write(theory)
        print(theory)
        print("  [Q12] Saved: q12_homomorphic.png, q12_homomorphic_results.png, q12_theory.txt")
