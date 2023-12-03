import random
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


def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels


def blend_latents(latent, noised_latent, alpha):
    return latent * alpha + noised_latent * (1. - alpha)

def fit_and_resize_image (image, vae, max_size=768, resampling="bicubic", upscale="false", batch_size=1, add_noise=0.0):
    size = get_image_size(image)
    new_width, new_height, aspect_ratio = get_max_size(size[0], size[1], max_size, upscale)

    img = tensor2pil(image)
    resized_image = img.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resampling]))
    tensor_img = pil2tensor(resized_image)

    pixels = vae_encode_crop_pixels(tensor_img)

    if add_noise > 0.0:
        noise = torch.randn_like(vae.encode(pixels[:,:,:,:3]))
        noised_latent = blend_latents(noise, vae.encode(pixels[:,:,:,:3]), add_noise)
        noised_latent = noised_latent.repeat((batch_size, 1,1,1))

    # vae encode the image
    t = vae.encode(pixels[:,:,:,:3])

    # batch the latent vectors
    batched = t.repeat((batch_size, 1,1,1))

    return (
        {"samples": noised_latent if add_noise > 0.0 else batched},
        tensor_img,
        new_width,
        new_height,
        aspect_ratio,
        )

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

    CATEGORY = "Fitsize/Numbers"

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

    CATEGORY = "Fitsize/Numbers"

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

    CATEGORY = "Fitsize/Image"

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
                "add_noise": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
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

    CATEGORY = "Fitsize/Image"

    def fit_resize_latent (self, image, vae, max_size=768, resampling="bicubic", upscale="false", batch_size=1, add_noise=0.0):

        return fit_and_resize_image(image, vae, max_size, resampling, upscale, batch_size, add_noise)
    
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
                "add_noise": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
            }
        }

    RETURN_TYPES = (
        "LATENT",
        "IMAGE",
        "INT",
        "INT",
        "FLOAT",
        "MASK",
        )
    RETURN_NAMES = (
        "Latent",
        "Image",
        "Width",
        "Height",
        "Aspect Ratio",
        "Mask",
        )
    FUNCTION = "fit_resize_latent"

    CATEGORY = "Fitsize/Image"
    
    @staticmethod
    def load_image(image):
        if (type(image) == str):

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
    def IS_CHANGED(s, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1, add_noise=0.0):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1, add_noise=0.0):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

    def fit_resize_latent (self, vae, image, max_size=768, resampling="bicubic", upscale="false", batch_size=1, add_noise=0.0):

        got_image,mask = self.load_image(image)

        latent,img,new_width,new_height,aspect_ratio = fit_and_resize_image(got_image, vae, max_size, resampling, upscale, batch_size, add_noise)

        return (
            latent,
            img,
            new_width,
            new_height,
            aspect_ratio,
            mask,
            )
    

class CropImageIntoEvenPieces:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1,}),
                "columns": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "Fitsize/Image"

    def run(self, image, rows, columns):

        if rows < 1:
            rows = 1
        if columns < 1:
            columns = 1

        w = image.shape[2] # width
        h = image.shape[1] # height

        crop_width = int(w / columns)
        crop_height = int(h / rows)

        image = image.numpy()

        pieces = []
        for i in range(rows):
            for j in range(columns):
                y = i * crop_height
                x = j * crop_width

                crop = image[: , y : y + crop_height , x : x + crop_width , :]
                pieces.append(torch.from_numpy(crop))

        return (torch.cat(pieces, dim=0), )
    
class ImageRegionMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1,}),
                "columns": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1,}),
                "chosen_row": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,}),
                "chosen_column": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1,}),
            },
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "run"

    CATEGORY = "Fitsize/Mask"

    def run(self, image, rows, columns, chosen_row, chosen_column):

        if rows < 1:
            rows = 1
        if columns < 1:
            columns = 1

        w = image.shape[2] # width
        h = image.shape[1] # height

        crop_width = int(w / columns)
        crop_height = int(h / rows)
        
        mask = torch.zeros((h, w))

        min_y = crop_height * chosen_row
        max_y = min_y + crop_height
        min_x = crop_width * chosen_column
        max_x = min_x + crop_width

        mask[int(min_y):int(max_y), int(min_x):int(max_x)] = 1

        return (mask.unsqueeze(0), )
    

    
class RandomImageFromBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "seed": ("INT", {"default": 0}),
                "start_index": ("INT", {"default": -1, "min": -1, "max": 32, "step": 1,}),
                "select_amount": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "Fitsize/Image"

    def run(self, images, seed, start_index, select_amount):

        # if type(images) == torch.Tensor:
        #     images = images.numpy()

        if start_index == -1:
            start_index = np.random.randint(0, images.shape[0])
        if start_index >= images.shape[0]:
            start_index = images.shape[0]-1

        if select_amount > images.shape[0]:
            select_amount = images.shape[0]
        if select_amount < 1:
            select_amount = 1

        selected = images[start_index:start_index + select_amount]

        return (selected, )
    
class RandomImageFromList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": ("IMAGE", ),
                "seed": ("INT", {"default": 0}),
                "start_index": ("INT", {"default": -1, "min": -1, "max": 32, "step": 1,}),
                "select_amount": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "Fitsize/Image"

    def run(self, list, seed, start_index, select_amount):
        print(f'type of list: {type(list)}, length: {len(list)}')

        list_length = len(list)

        if start_index == -1:
            start_index = np.random.randint(0, list_length)
            # return random.choice(list, select_amount)
        if start_index >= list_length:
            start_index = list_length-1

        if select_amount > list_length:
            select_amount = list_length
        if select_amount < 1:
            select_amount = 1

        selected = list[start_index:start_index + select_amount]

        print(f'selected: {start_index} to {start_index + select_amount} found {len(selected)}')

        return (selected, )


    
class RandomImageFromBatches:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT",{"default": 0}),
                "start_index": ("INT", {"default": -1, "min": -1, "max": 32, "step": 1,}),
                "select_amount": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1,}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "run"

    CATEGORY = "Fitsize/Image"

    def run(self, seed=0, start_index=0, select_amount=1, **kwargs):

        batches = kwargs.values()

        selected = []

        print(f'len(batches): {len(batches)}')



        for img in batches:

        # if type(images) == torch.Tensor:
        #     images = images.numpy()

            if start_index == -1:
                start_index = np.random.randint(0, img.shape[0])
            if start_index >= img.shape[0]:
                start_index = img.shape[0]-1

            if select_amount > img.shape[0]:
                select_amount = img.shape[0]
            if select_amount < 1:
                select_amount = 1

            # add images to selected
            selected.append(img[start_index:start_index + select_amount])
        
        # try to return a tensor of images if all widths and heights match
        try:
            selected = torch.cat(selected, dim=0)
        except:
            pass

        return (selected, )
