"""
Q5: Gaussian Filtering on brain_proton_density_slice.png.
    (a) Compute 5x5 Gaussian kernel manually (σ=2)
    (b) Visualize 51x51 Gaussian kernel as 3D surface
    (c) Apply manual Gaussian smoothing
    (d) Apply cv2.GaussianBlur() and compare
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os


class GaussianFiltering:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        Compute a 2D Gaussian kernel of shape (size x size) for given sigma.
        We evaluate the Gaussian on a grid centered at 0.
        """
        # Half extent of the kernel
        half = size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        y = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        # 2D Gaussian: G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Normalize so the sum equals 1 (so we don't change overall brightness)
        kernel /= kernel.sum()
        return kernel

    def run(self):
        sigma = 2

        # (a) 5x5 kernel
        kernel_5 = self.gaussian_kernel(5, sigma)
        print("  [Q5a] 5×5 Gaussian kernel (σ=2):")
        print(np.round(kernel_5, 6))

        # Save kernel values as text
        with open(os.path.join(self.output_dir, "q5a_gaussian_kernel_5x5.txt"), "w") as f:
            f.write(f"5x5 Gaussian kernel, sigma={sigma}\n\n")
            f.write(np.array2string(kernel_5, precision=6, separator=", "))

        # (b) Visualize 51x51 kernel as 3D surface
        kernel_51 = self.gaussian_kernel(51, sigma)
        size51 = 51
        half51 = size51 // 2
        x_vals = np.arange(-half51, half51 + 1)
        y_vals = np.arange(-half51, half51 + 1)
        X, Y = np.meshgrid(x_vals, y_vals)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, kernel_51, cmap=cm.viridis, linewidth=0, antialiased=True)
        ax.set_title(f"51×51 Gaussian Kernel (σ={sigma})", fontsize=13, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Kernel value")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q5b_gaussian_3d.png"), dpi=150)
        plt.close()

        # (c) Manual convolution with 5x5 kernel using cv2.filter2D
        # cv2.filter2D applies a linear filter – this uses our manually computed kernel
        manual_blur = cv2.filter2D(self.img, -1, kernel_5.astype(np.float32))

        # (d) OpenCV built-in GaussianBlur (kernel size must be odd)
        opencv_blur = cv2.GaussianBlur(self.img, (5, 5), sigmaX=sigma, sigmaY=sigma)

        # Save outputs
        cv2.imwrite(os.path.join(self.output_dir, "q5c_manual_gaussian.png"), manual_blur)
        cv2.imwrite(os.path.join(self.output_dir, "q5d_opencv_gaussian.png"), opencv_blur)

        # Comparison figure
        diff = np.abs(manual_blur.astype(np.int16) - opencv_blur.astype(np.int16)).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        titles = ["Original", "Manual Gaussian (σ=2)", "OpenCV GaussianBlur", "Absolute Difference"]
        imgs = [self.img, manual_blur, opencv_blur, diff]
        cmaps = ["gray", "gray", "gray", "hot"]
        for ax, im, t, c in zip(axes, imgs, titles, cmaps):
            ax.imshow(im, cmap=c, vmin=0, vmax=255)
            ax.set_title(t, fontsize=10)
            ax.axis("off")
        plt.suptitle("Q5 – Gaussian Filtering Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q5_gaussian_comparison.png"), dpi=150)
        plt.close()

        print("  [Q5] Saved: q5a_gaussian_kernel_5x5.txt, q5b_gaussian_3d.png, q5c_manual_gaussian.png, q5d_opencv_gaussian.png, q5_gaussian_comparison.png")
