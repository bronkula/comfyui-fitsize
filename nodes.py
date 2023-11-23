from PIL import Image
import numpy as np
import torch

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_max_size (width, height, max):
    aspect_ratio = width / height

    fit_width = max
    fit_height = max

    if aspect_ratio > 1:
        fit_height = int(max / aspect_ratio)
    else:
        fit_width = int(max * aspect_ratio)

    return (fit_width, fit_height, aspect_ratio)

def get_image_size(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    return size

class FitSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_width": ("INT", {}),
                "original_height": ("INT", {}),
                "max_size": ("INT", {"default": 768, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_to_size"

    CATEGORY = "Fitsize"

    def fit_to_size (self, original_width, original_height, max_size):
        values = get_max_size(original_width, original_height, max_size)
        return values
    
class FitSizeFromImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 768, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_to_size_from_image"

    CATEGORY = "Fitsize"

    def fit_to_size_from_image (self, image, max_size):
        size = get_image_size(image)
        values = get_max_size(size[0], size[1], max_size)
        return values
    
class FitResizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_size": ("INT", {"default": 768, "step": 8}),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT","FLOAT")
    RETURN_NAMES = ("Image","Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_resize_image"

    CATEGORY = "Fitsize"

    def fit_resize_image (self, image, max_size=768, resampling="bicubic"):
        size = get_image_size(image)
        img = tensor2pil(image)

        octalwidth = size[0] if size[0] % 8 == 0 else size[0] + (8 - size[0] % 8)
        octalheight = size[1] if size[1] % 8 == 0 else size[1] + (8 - size[1] % 8)

        new_width, new_height, aspect_ratio = get_max_size(octalwidth, octalheight, max_size)

        print("new values",new_width, new_height, aspect_ratio)

        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1
        }

        print("Resampling:", resampling, resample_filters[resampling])

        resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))

        return (pil2tensor(resized_image),new_width,new_height,aspect_ratio)