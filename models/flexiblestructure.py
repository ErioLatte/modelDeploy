from typing import Optional, Tuple, Union, Dict, Any
import torch.nn as nn
import torch
from diffusers import (
    ControlNetModel,
)
from diffusers.configuration_utils import register_to_config
from models.extrablocks import INITIAL_STRUCTURE, TARGET_CHANNELS, DownsampleBlock2D, DownsampleLoCon2D, DownsampleLoConBlock2D
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.controlnet import ControlNetOutput, ControlNetConditioningEmbedding
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)

class FlexibleStructureControlNetModel(ControlNetModel):
  @register_to_config
  def __init__(
      self,
      target_block_out_channels=TARGET_CHANNELS,
      structure=None,
      zero_conv_mapping=list(range(len(TARGET_CHANNELS))),
      in_channels: int = 4,
      conditioning_channels: int = 3,
      flip_sin_to_cos: bool = True,
      freq_shift: int = 0,
      down_block_types: Tuple[str, ...] = (
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "DownBlock2D",
      ),
      mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
      only_cross_attention: Union[bool, Tuple[bool]] = False,
      downsample_padding: int = 1,
      mid_block_scale_factor: float = 1,
      act_fn: str = "silu",
      norm_num_groups: Optional[int] = 32,
      norm_eps: float = 1e-5,
      cross_attention_dim: int = 1280,
      transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
      encoder_hid_dim: Optional[int] = None,
      encoder_hid_dim_type: Optional[str] = None,
      use_linear_projection: bool = False,
      class_embed_type: Optional[str] = None,
      addition_embed_type: Optional[str] = None,
      addition_time_embed_dim: Optional[int] = None,
      num_class_embeds: Optional[int] = None,
      upcast_attention: bool = False,
      resnet_time_scale_shift: str = "default",
      projection_class_embeddings_input_dim: Optional[int] = None,
      controlnet_conditioning_channel_order: str = "rgb",
      conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
      global_pool_conditions: bool = False,
      addition_embed_type_num_heads: int = 64,
      **kwargs):
    nn.Module.__init__(self)

    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(structure)

    # input
    conv_in_kernel = 3
    conv_in_padding = (conv_in_kernel - 1) // 2
    self.conv_in = nn.Conv2d(
        in_channels, structure[0][1], kernel_size=conv_in_kernel, padding=conv_in_padding
    )

    # time
    time_embed_dim = structure[0][1] * 4
    self.time_proj = Timesteps(structure[0][1], flip_sin_to_cos, freq_shift)
    timestep_input_dim = structure[0][1]
    self.time_embedding = TimestepEmbedding(
        timestep_input_dim,
        time_embed_dim,
        act_fn=act_fn,
    )

    if encoder_hid_dim_type is None and encoder_hid_dim is not None:
        encoder_hid_dim_type = "text_proj"
        self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
        #logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

    if encoder_hid_dim is None and encoder_hid_dim_type is not None:
        raise ValueError(
            f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
        )

    if encoder_hid_dim_type == "text_proj":
        self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
    elif encoder_hid_dim_type == "text_image_proj":
        self.encoder_hid_proj = TextImageProjection(
            text_embed_dim=encoder_hid_dim,
            image_embed_dim=cross_attention_dim,
            cross_attention_dim=cross_attention_dim,
        )

    elif encoder_hid_dim_type is not None:
        raise ValueError(
            f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
        )
    else:
        self.encoder_hid_proj = None

    # class embedding
    if class_embed_type is None and num_class_embeds is not None:
        self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
    elif class_embed_type == "timestep":
        self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
    elif class_embed_type == "identity":
        self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
    elif class_embed_type == "projection":
        if projection_class_embeddings_input_dim is None:
            raise ValueError(
                "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
            )
        self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
    else:
        self.class_embedding = None

    if addition_embed_type == "text":
        if encoder_hid_dim is not None:
            text_time_embedding_from_dim = encoder_hid_dim
        else:
            text_time_embedding_from_dim = cross_attention_dim

        self.add_embedding = TextTimeEmbedding(
            text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
        )
    elif addition_embed_type == "text_image":
        self.add_embedding = TextImageTimeEmbedding(
            text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
        )
    elif addition_embed_type == "text_time":
        self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

    elif addition_embed_type is not None:
        raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

    # control net conditioning embedding
    self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
        conditioning_embedding_channels=structure[0][1],
        block_out_channels=conditioning_embedding_out_channels,
        conditioning_channels=conditioning_channels,
    )

    self.down_blocks = nn.ModuleList([])
    self.controlnet_down_blocks = nn.ModuleList([])

    if isinstance(only_cross_attention, bool):
        only_cross_attention = [only_cross_attention] * len(structure)


    # down
    structure_name_dict = {
        'ResnetBlock2D+Transformer2DModel': "CrossAttnDownBlock2D",
        'ResnetBlock2D': "DownBlock2D",
        'Downsample2D': "DownBlock2D",
        'DownsampleLoCon2D': "DownLoConBlock2D"
    }
    output_channel = structure[0][1]

    controlnet_block = nn.Conv2d(output_channel, target_block_out_channels[0], kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_down_blocks.append(controlnet_block)

    for i, (down_block_type, new_output_channel, num_heads) in enumerate(structure):
        input_channel = output_channel
        output_channel = new_output_channel

        if down_block_type == 'Downsample2D':
          down_block = DownsampleBlock2D(input_channel, output_channel)
        elif down_block_type == "DownsampleLoCon2D":
          down_block = DownsampleLoConBlock2D(input_channel, output_channel)
        else:
          down_block = get_down_block(
              structure_name_dict[down_block_type],
              num_layers= 1,
              transformer_layers_per_block=transformer_layers_per_block[i],
              in_channels=input_channel,
              out_channels=output_channel,
              temb_channels=time_embed_dim,
              add_downsample= False,
              resnet_eps=norm_eps,
              resnet_act_fn=act_fn,
              resnet_groups=norm_num_groups,
              cross_attention_dim=cross_attention_dim,
              num_attention_heads=num_heads,
              attention_head_dim=8,
              downsample_padding=downsample_padding,
              use_linear_projection=use_linear_projection,
              only_cross_attention=only_cross_attention[i],
              upcast_attention=upcast_attention,
              resnet_time_scale_shift=resnet_time_scale_shift,
          )
        self.down_blocks.append(down_block)

    self.zero_conv_mapping = zero_conv_mapping
    for i, out_channel_dim in enumerate(target_block_out_channels):

      layer_idx = zero_conv_mapping[i]
      in_channel_dim = structure[layer_idx][1]
      controlnet_block = nn.Conv2d(in_channel_dim, out_channel_dim, kernel_size=1)
      controlnet_block = zero_module(controlnet_block)
      self.controlnet_down_blocks.append(controlnet_block)

    # mid
    mid_block_channel = structure[-1][1]

    controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_mid_block = controlnet_block

    if mid_block_type == "UNetMidBlock2DCrossAttn":
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_heads,
            resnet_groups=norm_num_groups,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
        )
    elif mid_block_type == "UNetMidBlock2D":
        self.mid_block = UNetMidBlock2D(
            in_channels=structure[-1][1],
            temb_channels=time_embed_dim,
            num_layers=0,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=False,
        )
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")

  def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:

            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()
        shifted_zero_conv_mapping = [0] + [x+1 for x in self.zero_conv_mapping]

        for i, controlnet_block in enumerate(self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_samples[shifted_zero_conv_mapping[i]])
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )

def zero_module(module):
  for p in module.parameters():
      nn.init.zeros_(p)
  return module
