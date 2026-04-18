"""
IT5437 / IN4640 — Assignment 1: Intensity Transformations and Neighborhood Filtering
Student: [Your Index Number]

main.py — Run all questions sequentially.
Usage:
    cd assignment
    python main.py

Outputs are saved to ./outputs/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.q1_transformations  import IntensityTransformations
from utils.q2_lab_gamma         import LABGammaCorrection
from utils.q3_hist_eq           import ManualHistogramEqualization
from utils.q4_otsu              import OtsuSelectiveEqualization
from utils.q5_gaussian          import GaussianFiltering
from utils.q6_dog               import DerivativeOfGaussian
from utils.q7_zoom              import ImageZooming
from utils.q8_noise             import NoiseRemoval
from utils.q9_sharpen           import ImageSharpening
from utils.q10_bilateral        import BilateralFiltering
from utils.q11_freq_response    import SpatialFrequencyResponse
from utils.q12_homomorphic      import HomomorphicFiltering

IMAGES  = os.path.join(os.path.dirname(__file__), "images")
OUTPUTS = os.path.join(os.path.dirname(__file__), "outputs")

os.makedirs(OUTPUTS, exist_ok=True)


def separator(n):
    print("\n" + "=" * 60)
    print(f"  QUESTION {n}")
    print("=" * 60)


def run_q1():
    separator(1)
    IntensityTransformations(f"{IMAGES}/runway.png", OUTPUTS).run()

def run_q2():
    separator(2)
    LABGammaCorrection(f"{IMAGES}/sapphire.jpg", OUTPUTS).run()

def run_q3():
    separator(3)
    ManualHistogramEqualization(f"{IMAGES}/runway.png", OUTPUTS).run()

def run_q4():
    separator(4)
    OtsuSelectiveEqualization(f"{IMAGES}/highlights_and_shadows.jpg", OUTPUTS).run()

def run_q5():
    separator(5)
    GaussianFiltering(f"{IMAGES}/brain_proton_density_slice.png", OUTPUTS).run()

def run_q6():
    separator(6)
    DerivativeOfGaussian(f"{IMAGES}/einstein.png", OUTPUTS).run()

def run_q7():
    separator(7)
    ImageZooming(OUTPUTS).run(images_dir=IMAGES)

def run_q8():
    separator(8)
    NoiseRemoval(f"{IMAGES}/emma.jpg", OUTPUTS).run()

def run_q9():
    separator(9)
    ImageSharpening(f"{IMAGES}/daisy.jpg", OUTPUTS).run()

def run_q10():
    separator(10)
    BilateralFiltering(f"{IMAGES}/einstein.png", OUTPUTS).run()

def run_q11():
    separator(11)
    SpatialFrequencyResponse(OUTPUTS).run()

def run_q12():
    separator(12)
    HomomorphicFiltering(f"{IMAGES}/runway.png", OUTPUTS).run()


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  IT5437 / IN4640 — Assignment 1                     ║")
    print("║  Intensity Transformations & Neighborhood Filtering  ║")
    print("╚══════════════════════════════════════════════════════╝")

    run_q1()
    run_q2()
    run_q3()
    run_q4()
    run_q5()
    run_q6()
    run_q7()
    run_q8()
    run_q9()
    run_q10()
    run_q11()
    run_q12()

    print("\n✓ All questions complete. Outputs saved to ./outputs/")
