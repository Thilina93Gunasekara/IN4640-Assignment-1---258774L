"""
Q2: LAB Gamma Correction
Convert image to L*a*b* color space, apply gamma correction on the L channel only,
then show histograms before and after. Doing gamma in LAB avoids hue shift that would
occur if we gamma-corrected the RGB channels independently.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


class LABGammaCorrection:
    def __init__(self, image_path: str, output_dir: str = "./outputs"):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        # OpenCV loads BGR; convert to RGB for display
        self.image_bgr = img
        self.image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def apply_gamma_to_L(self, img_bgr: np.ndarray, gamma: float) -> np.ndarray:
        """
        Convert BGR → L*a*b*, apply gamma to L channel, convert back.
        L channel in OpenCV's 8-bit LAB is in [0, 255] (mapped from [0, 100]).
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        L = lab[:, :, 0] / 255.0            # normalize to [0,1]
        L_corrected = np.power(L, gamma)
        lab[:, :, 0] = np.clip(L_corrected * 255, 0, 255)
        lab_corrected = lab.astype(np.uint8)
        bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        return bgr_corrected

    def get_L_channel(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return the L channel (0-255 scale) of a BGR image."""
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    def run(self):
        gamma = 0.5     # chosen to brighten the sapphire image
        corrected_bgr = self.apply_gamma_to_L(self.image_bgr, gamma)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)

        L_before = self.get_L_channel(self.image_bgr)
        L_after  = self.get_L_channel(corrected_bgr)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"Q2: LAB Gamma Correction (γ={gamma})", fontsize=14, fontweight='bold')

        # Row 0: images
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title("Original (RGB)", fontsize=11)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(corrected_rgb)
        axes[0, 1].set_title(f"Gamma γ={gamma} on L channel", fontsize=11)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(L_before, cmap='gray')
        axes[0, 2].set_title("L Channel (Before)", fontsize=11)
        axes[0, 2].axis('off')

        # Row 1: histograms
        axes[1, 0].hist(L_before.ravel(), bins=256, range=(0, 255), color='steelblue', alpha=0.8)
        axes[1, 0].set_title("Histogram of L (Before)", fontsize=11)
        axes[1, 0].set_xlabel("Pixel Value")
        axes[1, 0].set_ylabel("Frequency")

        axes[1, 1].hist(L_after.ravel(), bins=256, range=(0, 255), color='coral', alpha=0.8)
        axes[1, 1].set_title("Histogram of L (After)", fontsize=11)
        axes[1, 1].set_xlabel("Pixel Value")
        axes[1, 1].set_ylabel("Frequency")

        axes[1, 2].hist(L_before.ravel(), bins=256, range=(0, 255), color='steelblue', alpha=0.6, label='Before')
        axes[1, 2].hist(L_after.ravel(), bins=256, range=(0, 255), color='coral', alpha=0.6, label='After')
        axes[1, 2].set_title("L Histogram Overlay", fontsize=11)
        axes[1, 2].set_xlabel("Pixel Value")
        axes[1, 2].legend()

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "q2_lab_gamma.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Q2] Gamma γ={gamma} applied on L channel in LAB space → {out_path}")
        print(f"  L mean before: {L_before.mean():.1f} → after: {L_after.mean():.1f}")

        return {"original": self.image_rgb, "corrected": corrected_rgb,
                "L_before": L_before, "L_after": L_after}
