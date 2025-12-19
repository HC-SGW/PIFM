import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.getenv("PIFM_OUTPUT_ROOT", os.path.join(SCRIPT_DIR, "outputs"))
OUTPUT_INTER_DIR = os.path.join(OUTPUT_ROOT, "output_inter")
DIFFERENCE_DIR = os.path.join(OUTPUT_INTER_DIR, "difference_inter")
VAL_MMSE_DIR = os.path.join(OUTPUT_INTER_DIR, "val_mmse")
LOSS_CURVE_DIR = os.path.join(OUTPUT_ROOT, "loss_curve")
MMSE_RAW_DIR = os.path.join(OUTPUT_INTER_DIR, "MMSE_raw")
KGRID_RAW_DIR = os.path.join(OUTPUT_INTER_DIR, "Kgrid_raw")
MMSE_FAKE_DIR = os.path.join(OUTPUT_INTER_DIR, "MMSE_fake")
MODELS_DIR = os.path.join(OUTPUT_ROOT, "models")


__all__ = [
    "SCRIPT_DIR",
    "OUTPUT_ROOT",
    "OUTPUT_INTER_DIR",
    "DIFFERENCE_DIR",
    "VAL_MMSE_DIR",
    "LOSS_CURVE_DIR",
    "MMSE_RAW_DIR",
    "KGRID_RAW_DIR",
    "MMSE_FAKE_DIR",
    "MODELS_DIR",
]
