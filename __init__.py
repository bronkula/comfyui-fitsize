from .nodes import FitSize, FitSizeFromImage

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FitSizeByMax": FitSize,
    "FitSizeFromImage": FitSizeFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FitSizeByMax": "Fit Size",
    "FitSizeFromImage": "Fit Size From Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']