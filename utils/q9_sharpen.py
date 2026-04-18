"""
Q9: Image sharpening on daisy.jpg.
    Technique: Unsharp Masking (add scaled Laplacian / high-frequency back)
    sharpened = original + k * (original - blurred)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ImageSharpening:
    def __init__(self, image_path, output_dir):
        self.output_dir = output_dir
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

    def run(self):
        # Unsharp masking – classic sharpening approach
        # Step 1: blur the image to extract the low-frequency (smooth) component
        blurred = cv2.GaussianBlur(self.img, (5, 5), sigmaX=1.0)

        # Step 2: high-frequency detail = original − blurred (the "mask")
        high_freq = self.img.astype(np.float64) - blurred.astype(np.float64)

        # Step 3: add the detail back with a sharpening factor k
        k = 1.5  # sharpening strength – tuned for daisy image
        sharpened = np.clip(self.img.astype(np.float64) + k * high_freq, 0, 255).astype(np.uint8)

        # Also try a Laplacian-based sharpening for comparison
        laplacian = cv2.Laplacian(self.img, cv2.CV_64F)
        lap_sharp = np.clip(self.img.astype(np.float64) - 0.5 * laplacian, 0, 255).astype(np.uint8)

        # Save
        cv2.imwrite(os.path.join(self.output_dir, "q9_unsharp_mask.png"), sharpened)
        cv2.imwrite(os.path.join(self.output_dir, "q9_laplacian_sharp.png"), lap_sharp)

        # Comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(
            axes,
            [self.img, sharpened, lap_sharp],
            ["Original", "Unsharp Masking (k=1.5)", "Laplacian Sharpening"]
        ):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        plt.suptitle("Q9 – Image Sharpening (daisy.jpg)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "q9_sharpening_results.png"), dpi=150)
        plt.close()

        print("  [Q9] Saved: q9_unsharp_mask.png, q9_laplacian_sharp.png, q9_sharpening_results.png")
