# Z-Sampler Turbo

![text](/docs/zsampler_turbo.jpg)

The distinctive feature of this sampler is that it divides the number of steps into three stages: composition, details, and refinement. The sigmas are calculated to keep the image stable from 4 to 9 steps. Naturally, with fewer steps, the final quality decreases, but the resulting images remain quite similar.

Between 7 and 9 steps, the image achieves sufficient quality and detail, making further refining or post-processing unnecessary.

This sampler also enhances prompt adherence, improves overall image coherence, and eliminates the need for a "ModelSamplingAuraFlow" node with its 'shift' parameter adjustment.


## Inputs

### model
Any checkpoint from the "Z-Image Turbo" model. This sampler hasn't been extensively tested with LoRAs applied, nor has it been determined which types of LoRA training might benefit from this three-stage sampling process.

### positive
The positive conditioning input, typically the prompt embedding. There is no negative conditioning because this sampler always operates at CFG 1.0. In the future, it may be worth testing a similar sampling approach with a slightly higher CFG value, but I wanted to avoid adding another variable during testing.

### latent_input
The initial latent image to be denoised, typically an 'Empty Latent' for text-to-image or an encoded image for img2img processing.

### seed
The random number generator seed, ensuring reproducibility with the same value.

### steps
Number of sampling iterations, ranging from 4 to 9.

### denoise
Amount of denoising applied:
 - For standard text-to-image processes, keep at 1.0.
 - For inpainting tasks, values between 0.75 and 0.90 often work well, though smaller values may also be used.
 - For minor adjustments across the entire image without losing composition, small values like 0.20 or 0.10 can be effective.

### initial_noise_calibration
Adjusts the initial noise level to align with model expectations.  
Typically enhances image contrast and saturation, with higher percentages increasing these effects more strongly.
A value of "100%" is usually optimal, but lower percentages like "50%" or even complete disablement might be necessary if less contrast or saturation is desired.  
(More information on this parameter below).

### lowres_bias
A hack to accelerate initial noise calibration by calculating it on a 256x256 image instead of the actual size. Generally, keep this disabled as it may reduce final image quality.


## Outputs

### latent_output
The resulting denoised latent image, ready for VAE decoding or further processing in another sampler node.

