from .nodes import FitSize, FitSizeFromImage, FitResizeImage, FitResizeLatent, LoadToFitResizeLatent

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Fit Size From Int": FitSize,
    "Fit Size From Image": FitSizeFromImage,
    "Fit Image And Resize": FitResizeLatent,
    "Load Image And Resize To Fit": LoadToFitResizeLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Fit Size From Int": "Fit Size From Int",
    "Fit Size From Image": "Fit Size From Image",
    "Fit Image And Resize": "Fit Image And Resize",
    "Load Image And Resize To Fit": "Load Image And Resize To Fit",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']