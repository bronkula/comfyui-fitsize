from .nodes import FitSize, FitSizeFromImage, FitResizeImage, FitResizeLatent, LoadToFitResizeLatent

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FitSizeFromInt": FitSize,
    "FitSizeFromImage": FitSizeFromImage,
    "FitSizeResizeImage": FitResizeImage,
    "FitSizeResizeLatent": FitResizeLatent,
    "LoadToFitResizeLatent": LoadToFitResizeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FitSizeFromInt": "Fit Size From Int",
    "FitSizeFromImage": "Fit Size From Image",
    "FitSizeResizeImage": "Fit Resize Image",
    "FitSizeResizeLatent": "Fit Resize Latent",
    "LoadToFitResizeLatent": "Load Image To Fit Resize Latent",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']