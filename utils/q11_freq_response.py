"""
Q11: Relationship between spatial filtering and frequency response.
     This is a theoretical question – we generate visualizations of the 
     frequency responses of box, Gaussian, and Laplacian filters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class SpatialFrequencyResponse:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def freq_response(kernel, N=256):
        """Compute 2D DFT of a small kernel padded to NxN, return magnitude spectrum."""
        padded = np.zeros((N, N))
        kh, kw = kernel.shape
        # Place kernel at center
        top = N // 2 - kh // 2
        left = N // 2 - kw // 2
        padded[top:top + kh, left:left + kw] = kernel
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
        return np.abs(F)

    def run(self):
        N = 256

        # --- Box (averaging) filter ---
        box = np.ones((9, 9), dtype=np.float64) / 81.0

        # --- Gaussian filter ---
        sigma = 2
        k = 9
        half = k // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, x)
        gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gauss /= gauss.sum()

        # --- Laplacian filter ---
        laplacian = np.array([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]], dtype=np.float64)

        filters = [("Box Filter\n(9×9 averaging)", box),
                   (f"Gaussian Filter\n(9×9, σ={sigma})", gauss),
                   ("Laplacian Filter\n(3×3)", laplacian)]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for col, (title, filt) in enumerate(filters):
            # Spatial domain
            axes[0, col].imshow(filt, cmap='viridis', interpolation='nearest')
            axes[0, col].set_title(f"{title}\nSpatial Domain", fontsize=10)
            axes[0, col].axis("off")

            # Frequency domain magnitude
            F = self.freq_response(filt, N=N)
            F_log = np.log1p(F)  # log scale for visibility
            axes[1, col].imshow(F_log, cmap='inferno', extent=[-N//2, N//2, -N//2, N//2])
            axes[1, col].set_title(f"{title}\nFrequency Response (log scale)", fontsize=10)
            axes[1, col].set_xlabel("u (freq)")
            axes[1, col].set_ylabel("v (freq)")

        plt.suptitle("Q11 – Spatial Filters and Their Frequency Responses", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q11_freq_response.png"), dpi=150)
        plt.close()

        # Print theoretical answers to txt file
        theory = """\
Q11 Theoretical Answers
=======================

(a) Convolution Theorem:
    Spatial domain convolution g(x,y) = f(x,y) * h(x,y) corresponds to pointwise
    multiplication in the frequency domain: G(u,v) = F(u,v) · H(u,v).
    This means applying a filter spatially is equivalent to shaping the spectrum
    of the image by multiplying with the filter's transfer function H(u,v).

(b) Filter frequency domain effects:
    - Averaging (box) filter: Acts as a low-pass filter. Attenuates high frequencies,
      passes low frequencies. Since it has a rectangular shape in spatial domain,
      its Fourier transform is a sinc function – causing ringing artifacts.
    
    - Gaussian filter: Also a low-pass filter, but with a smooth Gaussian-shaped
      frequency response. No abrupt cutoff, so no ringing. Very natural edge-preserving
      smoothing at low σ values.
    
    - Laplacian filter: Acts as a high-pass filter. Amplifies high-frequency components
      (edges, noise) and suppresses DC/low frequencies. Used for edge detection and
      image sharpening.

(c) Why Gaussian avoids ringing:
    The Fourier transform of a Gaussian is also a Gaussian – a smooth, monotonically
    decreasing function. It has no sidelobes, unlike the sinc response of the box filter.
    Ringing occurs when a filter has sharp transitions in frequency (like an ideal
    low-pass filter), which produce oscillating spatial artifacts at edges.
    Since the Gaussian is infinitely differentiable and has no sharp cutoff, it
    introduces no such oscillations.

(d) Best filter for high-frequency noise:
    Gaussian filtering is most suitable:
    - Spatial: it spreads each pixel's value over neighbors, smoothing out isolated
      noise spikes (which are spatially uncorrelated).
    - Frequency: it gently attenuates high-frequency noise while preserving the
      low-frequency structure of the image.
    - The box filter also reduces noise but introduces ringing. Laplacian would
      amplify noise, making it unsuitable.
"""
        with open(os.path.join(self.output_dir, "q11_theory.txt"), "w") as f:
            f.write(theory)
        print(theory)
        print("  [Q11] Saved: q11_freq_response.png, q11_theory.txt")
