



def get_max_size (width, height, max):
    aspect_ratio = width / height

    fit_width = max
    fit_height = max

    print(aspect_ratio, fit_width, fit_height, max, width, height)

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