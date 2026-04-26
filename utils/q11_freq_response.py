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


a)	The Convolution Theorem states that applying convolution to a filtered image f(x,y) using a filter h(x,y) in the spatial domain results in applying pointwise multiplication between their corresponding frequency representations, G(u,v) representing the filtered image and H(u,v) representing the filter, in the frequency domain. Therefore, a filter's Fourier transform H(u,v) indicates directly which frequencies are passed or attenuated through that filter and how much.

b)	Qualitative Frequency Domain Impacts
Box/averaging Filter: Low Frequency filter with a Sinc Function DFT. The DFT of a Box/Averaging Filter passes low frequency smooth areas of an image and reduces (attenuates) high frequencies (such as edges and noise). Because the Sinc has side lobes, it will produce ringing artifacts.

c)	Tuning forks ring because of frequency discontinuities — a phenomenon known as the Gibbs effect. An ideal low-pass filter can be represented by a rectangular cutoff in the frequency domain, while its inverse discrete Fourier transform (IDFT) will exhibit a sinc wave with oscillating tails in the spatial domain. The Gaussian function has a smooth, infinitely differentiable spectrum in the frequency domain, which means that there is no abrupt cutoff to give rise to oscillations in the spatial domain.

d)	Gaussian Filter: Also, low frequency, but a Gaussian DFT is also Gaussian (smooth). The Gaussian filter has no side lobes, so it gradually attenuates high frequencies. Therefore, the Gaussian is an excellent filter for reducing image noise with no ringing artifacts.
Figure 14 (located at outputs/q11_freq_response.png) depicts 2-dimensional spatial domain filter kernels (top) and their log-magnitude frequency response (bottom). The Gaussian filter shows a smooth and balanced frequency response whereas the Laplacian filter shows amplification of high frequency signals.

"""
        with open(os.path.join(self.output_dir, "q11_theory.txt"), "w") as f:
            f.write(theory)
        print(theory)
        print("  [Q11] Saved: q11_freq_response.png, q11_theory.txt")
