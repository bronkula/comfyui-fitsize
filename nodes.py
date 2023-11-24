from PIL import Image, ImageOps
import torch
import os
import hashlib
import folder_paths
import numpy as np

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_max_size (width, height, max, upscale="false"):
    aspect_ratio = width / height

    fit_width = max
    fit_height = max

    if upscale == "false" and width <= max and height <= max:
        return (width, height, aspect_ratio)
    
    if aspect_ratio > 1:
        fit_height = int(max / aspect_ratio)
    else:
        fit_width = int(max * aspect_ratio)

    new_width, new_height = octal_sizes(fit_width, fit_height)

    return (new_width, new_height, aspect_ratio)

def get_image_size(IMAGE) -> tuple[int, int]:
    samples = IMAGE.movedim(-1, 1)
    size = samples.shape[3], samples.shape[2]
    return size

def octal_sizes (width, height):
    octalwidth = width if width % 8 == 0 else width + (8 - width % 8)
    octalheight = height if height % 8 == 0 else height + (8 - height % 8)
    return (octalwidth, octalheight)

resample_filters = {
    'nearest': 0,
    'lanczos': 1,
    'bilinear': 2,
    'bicubic': 3,
}

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
                "upscale": (["false", "true"],)
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_to_size"

    CATEGORY = "Fitsize"

    def fit_to_size (self, original_width, original_height, max_size, upscale="false"):
        values = get_max_size(original_width, original_height, max_size, upscale)
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
                "upscale": (["false", "true"],)
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_to_size_from_image"

    CATEGORY = "Fitsize"

    def fit_to_size_from_image (self, image, max_size, upscale="false"):
        size = get_image_size(image)
        values = get_max_size(size[0], size[1], max_size, upscale)
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
                "upscale": (["false", "true"],)
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT","FLOAT")
    RETURN_NAMES = ("Image","Fit Width", "Fit Height", "Aspect Ratio")
    FUNCTION = "fit_resize_image"

    CATEGORY = "Fitsize"

    def fit_resize_image (self, image, max_size=768, resampling="bicubic", upscale="false", latent=False):
        size = get_image_size(image)
        img = tensor2pil(image)

        new_width, new_height, aspect_ratio = get_max_size(size[0], size[1], max_size, upscale)

        resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))

        return (pil2tensor(resized_image),new_width,new_height,aspect_ratio)


    
class FitResizeLatent():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "max_size": ("INT", {"default": 768, "step": 8}),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "upscale": (["false", "true"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "IMAGE",
        "INT",
        "INT",
        "FLOAT",
        )
    RETURN_NAMES = (
        "Latent",
        "Image",
        "Fit Width",
        "Fit Height",
        "Aspect Ratio",
        )
    FUNCTION = "fit_resize_latent"

    CATEGORY = "Fitsize"

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def fit_resize_latent (self, image, vae, max_size=768, resampling="bicubic", upscale="false", batch_size=1):

        size = get_image_size(image)
        img = tensor2pil(image)

        new_width, new_height, aspect_ratio = get_max_size(size[0], size[1], max_size, upscale)
        
        resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))
        tensor_img = pil2tensor(resized_image)

        # vae encode the image
        pixels = self.vae_encode_crop_pixels(tensor_img)
        t = vae.encode(pixels[:,:,:,:3])

        # batch the latent vectors
        batched = t.repeat((batch_size, 1,1,1))

        return (
            {"samples":batched},
            tensor_img,
            new_width,
            new_height,
            aspect_ratio,
            )
    
class LoadToFitResizeLatent():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "vae": ("VAE",),
                "image": (sorted(files), {"image_upload": True}),
                "max_size": ("INT", {"default": 768, "step": 8}),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "upscale": (["false", "true"],),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "IMAGE",
        "INT",
        "INT",
        "FLOAT",
        )
    RETURN_NAMES = (
        "Latent",
        "Image",
        "Fit Width",
        "Fit Height",
        "Aspect Ratio",
        )
    FUNCTION = "fit_resize_latent"

    CATEGORY = "Fitsize"

    @staticmethod
    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels
    
    @staticmethod
    def load_image(image):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

    @classmethod
    def IS_CHANGED(s, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True

    def fit_resize_latent (self, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1):

        got_image,mask = self.load_image(image)

        size = get_image_size(got_image)
        img = tensor2pil(got_image)

        new_width, new_height, aspect_ratio = get_max_size(size[0], size[1], max_size, upscale)
        
        resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))
        tensor_img = pil2tensor(resized_image)

        # vae encode the image
        pixels = self.vae_encode_crop_pixels(tensor_img)
        t = vae.encode(pixels[:,:,:,:3])

        # batch the latent vectors
        batched = t.repeat((batch_size, 1,1,1))

        return (
            {"samples":batched},
            tensor_img,
            new_width,
            new_height,
            aspect_ratio,
            )