"""
    modeled after the textual_inversion.py / train_dreambooth.py and the work
    of justinpinkney here: https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
"""
import inspect
import warnings
from typing import List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import torch.fft as fft

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging

from models.mutual_information.mutualinformation import *

# LORA Imports
from typing import Dict
from diffusers.loaders import (
    LoraLoaderMixin,
    text_encoder_lora_state_dict,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class ImagicStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Pipeline for imagic image editing.
    See paper here: https://arxiv.org/pdf/2210.09276.pdf

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offsensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
        r"""
        Returns:
            a state dict containing just the attention processor parameters.
        """
        attn_processors = unet.attn_processors
        attn_processors = attn_processors

        attn_processors_state_dict = {}

        for attn_processor_key, attn_processor in attn_processors.items():
            for parameter_key, parameter in attn_processor.state_dict().items():
                attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

        return attn_processors_state_dict


    def train(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        diffusion_model_learning_rate: float = 2e-4,
        text_embedding_optimization_steps: int = 500,
        model_fine_tuning_optimization_steps: int = 1000,
        # image_hom: Union[torch.FloatTensor, PIL.Image.Image] = None,
        **kwargs,
    ):
        
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="fp16",
        )

        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        if accelerator.is_main_process:
            accelerator.init_trackers(
                "imagic",
                config={
                    "embedding_learning_rate": embedding_learning_rate,
                    "text_embedding_optimization_steps": text_embedding_optimization_steps,
                },
            )

        
        prompt_modified = 'aerial view, ' + prompt
        text_input_modified = self.tokenizer(
            prompt_modified,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings_aerial = torch.nn.Parameter(
            self.text_encoder(text_input_modified.input_ids.to(self.device))[0]
        )
        text_embeddings_aerial = text_embeddings_aerial.detach()
        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0]
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_(True)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            [text_embeddings],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)
        latents_dtype = text_embeddings.dtype
        image = image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(image).latent_dist
        image_latents = init_latent_image_dist.sample(generator=generator)
        image_latents = 0.18215 * image_latents
        self.image_latents = image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        global_step = 0

        #Text embedding optimization 
        
        logger.info("First optimizing the text embedding to better reconstruct the init image")
        for _ in range(text_embedding_optimization_steps):
            with accelerator.accumulate(text_embeddings):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()
        
        text_embeddings.requires_grad_(False)
        
        # Now we fine tune the unet to better reconstruct the image        
        # now we will add new LoRA weights to the attention layers
        device = "cuda"
        unet_lora_attn_procs = {}
        unet_lora_parameters = []
        for name, attn_processor in self.unet.attn_processors.items():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
                lora_attn_processor_class = LoRAAttnAddedKVProcessor
            else:
                lora_attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )

            module = lora_attn_processor_class(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4
            )
            module.to(device)
            unet_lora_attn_procs[name] = module
            unet_lora_parameters.extend(module.parameters())

        self.unet.set_attn_processor(unet_lora_attn_procs)
        #LORA Addition Done

        # Now optimize the UNet 
        
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps), disable=not accelerator.is_local_main_process)
        
        logger.info("Next fine tuning the entire model to better reconstruct the init image")
        for _ in range(model_fine_tuning_optimization_steps):
            with accelerator.accumulate(self.unet.parameters()):
                # Sample noise that we'll add to the latents
                noise = torch.randn(image_latents.shape).to(image_latents.device)
                timesteps = torch.randint(1000, (1,), device=image_latents.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps)
                
                # Predict the noise residual
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        
        accelerator.wait_for_everyone()
        
        # Now optimize the network on the homography image 

        # Initialize the optimizer
        text_embeddings_hom = text_embeddings.clone()
        text_embeddings_hom.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [text_embeddings_hom],  # only optimize the embeddings
            lr=embedding_learning_rate,
        )

        progress_bar = tqdm(range(text_embedding_optimization_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        #Text embedding optimization 

        optimizer = torch.optim.Adam(
            self.unet.parameters(),  # only optimize unet
            lr=diffusion_model_learning_rate,
        )
        
        progress_bar = tqdm(range(model_fine_tuning_optimization_steps), disable=not accelerator.is_local_main_process)
        
        # Saving text embeddings variables for inferencing
        self.text_embeddings_front_opt = text_embeddings
        self.text_embeddings_aerial = text_embeddings_aerial
    
    @torch.no_grad()
    def __call__(
	    self,
	    alpha: float = 1.2,
	    height: Optional[int] = 512,
	    width: Optional[int] = 512,
	    num_inference_steps: Optional[int] = 50,
	    generator: Optional[torch.Generator] = None,
	    output_type: Optional[str] = "pil",
	    return_dict: bool = True,
	    guidance_scale: float = 7.5,
	    eta: float = 0.0,
	    mi_lr: float = 1e-5, 
	    eval_prompt: Union[str, List[str]] = None,
        image_hom: PIL.Image.Image = None,
	    **kwargs,
	):
        prompt_modified = eval_prompt
        text_input_modified = self.tokenizer(
            prompt_modified,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings_aerial = torch.nn.Parameter(
            self.text_encoder(text_input_modified.input_ids.to(self.device))[0]
        )
        text_embeddings_aerial = text_embeddings_aerial.detach()

        if isinstance(image_hom, PIL.Image.Image):
            self.image_hom = preprocess(image_hom).to(self.device)
        init_latent_image_dist_hom = self.vae.encode(self.image_hom).latent_dist
        image_latents_hom = init_latent_image_dist_hom.sample(generator=generator)
        noise = torch.randn(image_latents_hom.shape, device=self.device)
        image_latents_hom_noisy = image_latents_hom + 0.1 * noise  

        latents = image_latents_hom_noisy

        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings_aerial.dtype
        if self.device.type == "mps":
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                self.device
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)


        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        MI = MutualInformation(num_bins=256, sigma=0.1, normalize=True).to(self.device)
        for i, t in enumerate(tqdm(range(num_inference_steps))):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            text_embeddings = text_embeddings_aerial

            print(f"initial latent shape: {latent_model_input.shape}, latents shape: {latents.shape}")
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)





