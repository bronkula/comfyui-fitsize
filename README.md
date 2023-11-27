# Comfy Fit Size

A simple set of nodes for making an image fit within a bounding box

## Installation

To install this node, open your `custom_nodes` folder in a terminal and clone this project into it.

`git clone https://github.com/bronkula/comfyui-fitsize.git`

## Nodes

There are currently a few different nodes that are all based around fitting content within a bounding box. The reasoning behind this is that I like to copy screengrab content into comfyui, and I don't want to worry about what size or aspect ratio it is. I want it to automatically be fit to something appropriate.

These nodes will automatically fudge the sizes to fit within increments of 8, so that the output image sizes will be more palatable to Stable Diffusion. They should all be able to upscale or downscale whatever input they're given.

### Fit Size From Int

This node is the most basic. It's the basic math of fitting some content within a bounding box. Give it a width, height, and max size, and it will scale the sizes to fit in the max size.

![Fit Size From Int](assets/fitsizefromint.png)

Obviously you can also change the inputs to accept values from outside.

![Fit Size From Int](assets/fitsizefromintb.png)

### Fit Size From Image

This node accepts any image input and will extract the width and height automatically. All of these nodes can be told to upscale or not. The normal use of these nodes is to reduce a size down to something reasonable, but if upscale is true than it will also try to increase the size to the max_size.

![Fit Size From Image](assets/fitsizefromimage.png)

### Fit And Resize Image

Now this is where things get interesting. This node accepts a vae so that we can skip right to outputting a rescaled image. It will output both an image and a latent batch. This makes it a very useful tool for img2img workflows. 

![Fit Resize Image](assets/fitresizeimage.png)

Because the node now outputs a latent batch based on the original image, img2img workflows are much easier. Simply reduce the denoise in the ksampler to somewhere around `0.5`. Anything less will be more like the original image, and anything more will start to deviate wildly.

![Fit Resize Image](assets/fitresizeimagec.png)

### Load Image And Resize To Fit

But I decided that I wanted to just add in the image handling completely into one node, so that's what this one is. It has built in image handling compeletely. You can copy and paste image data directly into it, just like the default comfyui node. You don't have to save an image, just paste it in.

This workflow uses a number of other custom node sets to showcase that this node works great in the middle of a workflow.

![Fit Resize Image](assets/loadtofitresizeimage.png)

Here is a simpler workflow example using the Efficiency nodes. The Fit nodes should fit right into the middle of a lot of other workflows.

![Fit Resize Image](assets/loadtofitresizeimageb.png)

## Author

Hamilton Cline https://hdraws.com

## Checkpoints

- [Sugar Donut](https://civitai.com/models/161043/sugar-donut)
- [Jelly Donut](https://civitai.com/models/156381/jelly-donut)