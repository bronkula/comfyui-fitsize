from .nodes import *
from .startup_utils import symlink_web_dir

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FS: Fit Size From Int": FitSize,
    "FS: Fit Size From Image": FitSizeFromImage,
    "FS: Fit Image And Resize": FitResizeLatent,
    "FS: Load Image And Resize To Fit": LoadToFitResizeLatent,
    "FS: Pick Image From Batch": RandomImageFromBatch,
    "FS: Pick Image From Batches": RandomImageFromBatches,
    "FS: Pick Image From List": RandomImageFromList,
    "FS: Crop Image Into Even Pieces": CropImageIntoEvenPieces,
    "FS: Image Region To Mask": ImageRegionMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FS: Fit Size From Int": "Fit Size From Int (FS)",
    "FS: Fit Size From Image": "Fit Size From Image (FS)",
    "FS: Fit Image And Resize": "Fit Image And Resize (FS)",
    "FS: Load Image And Resize To Fit": "Load Image And Resize To Fit (FS)",
    "FS: Pick Image From Batch": "Pick Image From Batch (FS)",
    "FS: Pick Image From Batches": "Pick Image From Batches (FS)",
    "FS: Pick Image From List": "Pick Image From List (FS)",
    "FS: Crop Image Into Even Pieces": "Crop Image Into Even Pieces (FS)",
    "FS: Image Region To Mask": "Image Region To Mask (FS)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

EXTENSION_NAME = "Fitsize"
WEB_DIRECTORY = "./js"
