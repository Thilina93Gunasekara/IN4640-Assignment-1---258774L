"""
Q7: Image Zooming by factor s using:
    (a) Nearest-neighbor interpolation (manual)
    (b) Bilinear interpolation (manual)
    Test on provided image pairs: original + small, compute SSD.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class ImageZooming:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def nearest_neighbor_zoom(img, scale):
        """
        Scale up (or down) an image using nearest-neighbor interpolation.
        For each output pixel (i,j), we map back to the source: (i/s, j/s),
        and round to the nearest integer to pick the source pixel.
        """
        h, w = img.shape[:2]
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))

        # Build coordinate arrays for output image
        row_indices = np.round(np.arange(new_h) / scale).clip(0, h - 1).astype(np.int32)
        col_indices = np.round(np.arange(new_w) / scale).clip(0, w - 1).astype(np.int32)

        # Index the source image (fancy indexing handles both grayscale and color)
        zoomed = img[np.ix_(row_indices, col_indices)]
        return zoomed

    @staticmethod
    def bilinear_zoom(img, scale):
        """
        Scale up an image using bilinear interpolation.
        For each output pixel (i,j):
          - Map to source coordinates (y_s, x_s) = (i/s, j/s)
          - Find the 4 surrounding pixels
          - Weighted average based on fractional distances
        """
        h, w = img.shape[:2]
        new_h = max(1, round(h * scale))
        new_w = max(1, round(w * scale))

        # Source coordinates for each output pixel
        y_src = np.arange(new_h) / scale
        x_src = np.arange(new_w) / scale

        # Floor (top-left corner of interpolation cell)
        y0 = np.floor(y_src).clip(0, h - 2).astype(np.int32)
        x0 = np.floor(x_src).clip(0, w - 2).astype(np.int32)
        y1 = (y0 + 1).clip(0, h - 1)
        x1 = (x0 + 1).clip(0, w - 1)

        # Fractional parts (how far into the cell we are)
        dy = (y_src - y0).clip(0, 1)  # shape: (new_h,)
        dx = (x_src - x0).clip(0, 1)  # shape: (new_w,)

        # Expand dims for broadcasting: dy -> (new_h,1), dx -> (1,new_w)
        dy = dy[:, np.newaxis]
        dx = dx[np.newaxis, :]

        # Bilinear weights
        w00 = (1 - dy) * (1 - dx)
        w01 = (1 - dy) * dx
        w10 = dy * (1 - dx)
        w11 = dy * dx

        if img.ndim == 3:
            # Color image – apply per channel
            result = np.zeros((new_h, new_w, img.shape[2]), dtype=np.float64)
            for c in range(img.shape[2]):
                ch = img[:, :, c].astype(np.float64)
                result[:, :, c] = (
                    w00 * ch[np.ix_(y0, x0)] +
                    w01 * ch[np.ix_(y0, x1)] +
                    w10 * ch[np.ix_(y1, x0)] +
                    w11 * ch[np.ix_(y1, x1)]
                )
        else:
            # Grayscale
            img_f = img.astype(np.float64)
            result = (
                w00 * img_f[np.ix_(y0, x0)] +
                w01 * img_f[np.ix_(y0, x1)] +
                w10 * img_f[np.ix_(y1, x0)] +
                w11 * img_f[np.ix_(y1, x1)]
            )
        return result.clip(0, 255).astype(np.uint8)

    @staticmethod
    def normalized_ssd(img_a, img_b):
        """
        Resize img_b to match img_a if needed, then compute normalized SSD.
        SSD = sum((a-b)^2) / N  where N is total pixels.
        """
        # Resize to same dimensions if needed
        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        a = img_a.astype(np.float64)
        b = img_b.astype(np.float64)
        ssd = np.sum((a - b) ** 2) / a.size
        return ssd

    def run(self, images_dir):
        # Image pairs: (original, small version) – matching stems
        pairs = [
            ("im01.png", "im01small.png"),
            ("im02.png", "im02small.png"),
            ("im03.png", "im03small.png"),
            ("taylor.jpg", "taylor_small.jpg"),
        ]

        ssd_results = []

        for orig_name, small_name in pairs:
            orig_path = os.path.join(images_dir, orig_name)
            small_path = os.path.join(images_dir, small_name)

            orig = cv2.imread(orig_path)
            small = cv2.imread(small_path)
            if orig is None or small is None:
                print(f"  [Q7] Skipping pair {orig_name}/{small_name} – file not found")
                continue

            h_orig, w_orig = orig.shape[:2]
            h_small, w_small = small.shape[:2]

            # Compute the zoom factor needed to bring small → original size
            scale_h = h_orig / h_small
            scale_w = w_orig / w_small
            scale = (scale_h + scale_w) / 2  # use average if they differ slightly

            # Apply both interpolation methods
            nn_zoom = self.nearest_neighbor_zoom(small, scale)
            bl_zoom = self.bilinear_zoom(small, scale)

            # Resize to exact original size for fair SSD comparison
            nn_zoom_resized = cv2.resize(nn_zoom, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            bl_zoom_resized = cv2.resize(bl_zoom, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            ssd_nn = self.normalized_ssd(orig, nn_zoom_resized)
            ssd_bl = self.normalized_ssd(orig, bl_zoom_resized)
            ssd_results.append((orig_name, scale, ssd_nn, ssd_bl))

            # Save zoomed images
            stem = os.path.splitext(orig_name)[0]
            cv2.imwrite(os.path.join(self.output_dir, f"q7_{stem}_nn_zoom.png"), nn_zoom_resized)
            cv2.imwrite(os.path.join(self.output_dir, f"q7_{stem}_bl_zoom.png"), bl_zoom_resized)

            # Per-pair comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, im, title in zip(
                axes,
                [cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(nn_zoom_resized, cv2.COLOR_BGR2RGB),
                 cv2.cvtColor(bl_zoom_resized, cv2.COLOR_BGR2RGB)],
                ["Original", f"NN Zoom (×{scale:.2f})\nSSD={ssd_nn:.2f}",
                 f"Bilinear Zoom (×{scale:.2f})\nSSD={ssd_bl:.2f}"]
            ):
                ax.imshow(im); ax.set_title(title, fontsize=10); ax.axis("off")
            plt.suptitle(f"Q7 – Zoom Comparison: {orig_name}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"q7_{stem}_zoom_comparison.png"), dpi=120)
            plt.close()

        # Print SSD summary
        print("  [Q7] Normalized SSD (lower is better):")
        print(f"  {'Image':<30} {'Scale':>7} {'SSD-NN':>12} {'SSD-Bilinear':>14}")
        print("  " + "-" * 65)
        for name, s, snn, sbl in ssd_results:
            print(f"  {name:<30} {s:>7.2f} {snn:>12.4f} {sbl:>14.4f}")

        # Summary table figure
        if ssd_results:
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.axis("off")
            col_labels = ["Image", "Scale", "SSD (NN)", "SSD (Bilinear)"]
            table_data = [[n, f"{s:.2f}", f"{snn:.4f}", f"{sbl:.4f}"]
                          for n, s, snn, sbl in ssd_results]
            tbl = ax.table(cellText=table_data, colLabels=col_labels,
                           cellLoc="center", loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.2, 1.8)
            plt.title("Q7 – Normalized SSD Summary", fontsize=13, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "q7_ssd_summary.png"), dpi=150)
            plt.close()
