"""
Q6: Derivative of Gaussian on einstein.png.
    (a) Analytical proof (printed as text output)
    (b) Compute 5x5 DoG kernels in x and y directions
    (c) 3D surface plot of 51x51 DoG kernel
    (d) Apply kernels to get image gradients
    (e) Apply cv2.Sobel() and compare
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os


class DerivativeOfGaussian:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    @staticmethod
    def dog_kernel(size, sigma, direction='x'):
        """
        Compute the derivative of Gaussian kernel.
        dG/dx = -(x/σ²) * G(x,y)
        dG/dy = -(y/σ²) * G(x,y)
        Then normalize so the sum of positive values = 1 (standard normalization).
        """
        half = size // 2
        x = np.arange(-half, half + 1, dtype=np.float64)
        y = np.arange(-half, half + 1, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        G = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        if direction == 'x':
            kernel = -(xx / sigma**2) * G
        else:  # y direction
            kernel = -(yy / sigma**2) * G

        # Normalize: divide by sum of absolute positive values
        pos_sum = kernel[kernel > 0].sum()
        if pos_sum > 0:
            kernel /= pos_sum
        return kernel

    def run(self):
        sigma = 2

        # (a) Print analytical derivation
        proof = (
            "Q6(a) – Derivation of partial derivatives of G(x,y):\n"
            "G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))\n\n"
            "∂G/∂x = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²)) * (-2x/(2σ²))\n"
            "       = -(x/σ²) * G(x,y)  ✓\n\n"
            "∂G/∂y = -(y/σ²) * G(x,y)  (by symmetry) ✓\n"
        )
        print(proof)
        with open(os.path.join(self.output_dir, "q6a_dog_derivation.txt"), "w") as f:
            f.write(proof)

        # (b) 5x5 DoG kernels
        dog_x_5 = self.dog_kernel(5, sigma, 'x')
        dog_y_5 = self.dog_kernel(5, sigma, 'y')
        with open(os.path.join(self.output_dir, "q6b_dog_kernels_5x5.txt"), "w") as f:
            f.write("5x5 DoG Kernel – X direction:\n")
            f.write(np.array2string(dog_x_5, precision=6, separator=", "))
            f.write("\n\n5x5 DoG Kernel – Y direction:\n")
            f.write(np.array2string(dog_y_5, precision=6, separator=", "))

        # (c) 3D surface of 51x51 DoG-x kernel
        dog_x_51 = self.dog_kernel(51, sigma, 'x')
        half51 = 25
        x_vals = np.arange(-half51, half51 + 1)
        y_vals = np.arange(-half51, half51 + 1)
        X, Y = np.meshgrid(x_vals, y_vals)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, dog_x_51, cmap=cm.RdBu, linewidth=0, antialiased=True)
        ax.set_title(f"51×51 Derivative of Gaussian – X Direction (σ={sigma})", fontsize=12, fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("Kernel value")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q6c_dog_3d.png"), dpi=150)
        plt.close()

        # (d) Apply DoG kernels to get horizontal and vertical gradients
        grad_x = cv2.filter2D(self.img, cv2.CV_64F, dog_x_5)
        grad_y = cv2.filter2D(self.img, cv2.CV_64F, dog_y_5)
        mag = np.sqrt(grad_x**2 + grad_y**2)

        # Scale for visualization
        def norm_disp(arr):
            a = np.abs(arr)
            if a.max() == 0:
                return np.zeros_like(arr, dtype=np.uint8)
            return (a / a.max() * 255).astype(np.uint8)

        gx_disp = norm_disp(grad_x)
        gy_disp = norm_disp(grad_y)
        mag_disp = norm_disp(mag)

        cv2.imwrite(os.path.join(self.output_dir, "q6d_grad_x_dog.png"), gx_disp)
        cv2.imwrite(os.path.join(self.output_dir, "q6d_grad_y_dog.png"), gy_disp)
        cv2.imwrite(os.path.join(self.output_dir, "q6d_magnitude_dog.png"), mag_disp)

        # (e) Compare with Sobel
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        sx_disp = norm_disp(sobel_x)
        sy_disp = norm_disp(sobel_y)
        sm_disp = norm_disp(sobel_mag)

        cv2.imwrite(os.path.join(self.output_dir, "q6e_sobel_x.png"), sx_disp)
        cv2.imwrite(os.path.join(self.output_dir, "q6e_sobel_y.png"), sy_disp)

        # Comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        row0 = [self.img, gx_disp, gy_disp]
        row1 = [mag_disp, sx_disp, sm_disp]
        titles0 = ["Original", "DoG Grad-X", "DoG Grad-Y"]
        titles1 = ["DoG Magnitude", "Sobel Grad-X", "Sobel Magnitude"]
        for ax, im, t in zip(axes[0], row0, titles0):
            ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=10); ax.axis("off")
        for ax, im, t in zip(axes[1], row1, titles1):
            ax.imshow(im, cmap="gray"); ax.set_title(t, fontsize=10); ax.axis("off")
        plt.suptitle("Q6 – Derivative of Gaussian vs Sobel (einstein.png)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q6_dog_vs_sobel.png"), dpi=150)
        plt.close()

        print("  [Q6] Saved: q6a-e outputs (3D plot, kernels, gradients, comparison)")
