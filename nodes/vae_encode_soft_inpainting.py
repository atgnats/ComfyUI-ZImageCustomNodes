"""
File    : vae_encode_soft_inpainting.py
Purpose : 
Author  : Martin Rizzo | <martinrizzo@gmail.com>
Date    : Feb 25, 2026
Repo    : https://github.com/martin-rizzo/ComfyUI-ZImagePowerNodes
License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          ComfyUI-ZImagePowerNodes
         ComfyUI nodes designed specifically for the "Z-Image" model.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

  ComfyUI V3 schema documentation can be found here:
  - https://docs.comfy.org/custom-nodes/v3_migration

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import torch
from torchvision.transforms.functional import gaussian_blur
from comfy_api.latest import io



class VAEEncodeSoftInpainting(io.ComfyNode):
    xTITLE         = "VAE Encode (for Soft Inpainting)"
    xCATEGORY      = ""
    xCOMFY_NODE_ID = ""
    xDEPRECATED    = False

    #__ INPUT / OUTPUT ____________________________________
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            display_name  = cls.xTITLE,
            category      = cls.xCATEGORY,
            node_id       = cls.xCOMFY_NODE_ID,
            is_deprecated = cls.xDEPRECATED,
            description   = (
                'Encodes the input image into a latent space using VAE encoding. '
            ),
            inputs=[
                io.Image.Input("pixels",
                               tooltip="??",
                              ),
                io.Vae.Input  ("vae",
                               tooltip="??",
                              ),
                io.Mask.Input ("mask",
                               tooltip="??",
                              ),
                io.Float.Input("softness_width",
                               tooltip="Sets the width of the softened area in pixels. "
                                       "Higher values expand the mask further with a smoother transition",
                              ),
            ],
            outputs=[
                io.Latent.Output(display_name="LATENT",
                                 tooltip="??",
                                ),
            ],
        )

    #__ FUNCTION __________________________________________
    @classmethod
    def execute(cls, vae, pixels, mask, softness_width=6):
        # `mask` must have shape [B, C, H, W] or [C, H, W]

        gray_out_masked_pixels = False
        return_binary_mask     = False

        # if required, smooth the mask using a Gaussian blur
        if softness_width > 0:
            kernel_size = int(2 * softness_width + 1)
            if  kernel_size % 2 == 0:
                kernel_size += 1
            sigma = float(softness_width) / 2.0
            smoothed_mask = gaussian_blur(mask, [kernel_size, kernel_size], [sigma, sigma])
            mask = torch.clamp(smoothed_mask , 0.0, 1.0)

        # resize the mask to the size of the image
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        # calculate which resolution would be compatible with VAE
        vae_downscale_ratio = vae.spacial_compression_encode()
        vae_width  = (pixels.shape[1] // vae_downscale_ratio) * vae_downscale_ratio
        vae_height = (pixels.shape[2] // vae_downscale_ratio) * vae_downscale_ratio

        # crop the image and the mask to the VAE compatible resolution
        pixels = pixels.clone()
        if pixels.shape[1] != vae_width or pixels.shape[2] != vae_height:
            x_offset = (pixels.shape[1] % vae_downscale_ratio) // 2
            y_offset = (pixels.shape[2] % vae_downscale_ratio) // 2
            pixels = pixels[:  , x_offset:(x_offset+vae_width), y_offset:(y_offset+vae_height), :]
            mask   = mask  [:,:, x_offset:(x_offset+vae_width), y_offset:(y_offset+vae_height)   ]

        # if required, gray out the image where the mask is set
        # (this operation is binary, the pixel is grayed out if the mask value is > 0.5)
        if gray_out_masked_pixels:
            mask_binary = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:,:,:,i] = (pixels[:,:,:,i] - 0.5) * mask_binary + 0.5

        # if required, convert the mask values to binary values (0 or 1)
        if return_binary_mask:
            mask = mask.round()

        samples = vae.encode(pixels)
        return ({"samples":samples, "noise_mask": mask}, )

