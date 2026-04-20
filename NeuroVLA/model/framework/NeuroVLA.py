
from __future__ import annotations
from typing import Union, List, Dict, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from NeuroVLA.model.framework.base_framework import baseframework
from NeuroVLA.model.modules.vlm import get_vlm_model
from NeuroVLA.training.trainer_utils.trainer_tools import resize_images
from NeuroVLA.model.tools import FRAMEWORK_REGISTRY
from NeuroVLA.model.modules.projector.QFormer import get_layerwise_qformer
from NeuroVLA.model.modules.action_model.spike_action_model_multitimestep import (
    get_action_model,
    get_gruedit_model
)





@FRAMEWORK_REGISTRY.register("NeuroVLA")
class NeuroVLA(baseframework):
    """
    NeuroVLA: Vision-Language-Action model for robotic manipulation.

    This model combines a vision-language model (Qwen-VL) with action prediction
    to generate robot actions from visual observations and language instructions.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # Vision-language model for processing images and instructions
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Q-Former for extracting action-relevant features from VLM hidden states
        self.layer_qformer = get_layerwise_qformer(config=self.config)

        # Read dimensions from config (with single-arm defaults for backward-compatibility)
        _am_cfg = config.framework.action_model if (config and hasattr(config.framework, "action_model")) else None
        _qf_cfg = config.framework.layer_qformer if (config and hasattr(config.framework, "layer_qformer")) else None
        qformer_out_dim = getattr(_qf_cfg, "ouptput_dim", 768) if _qf_cfg else 768
        action_dim     = getattr(_am_cfg, "action_dim", 7)     if _am_cfg else 7
        state_dim      = getattr(_am_cfg, "state_dim", 8)      if _am_cfg else 8
        hidden_dim     = getattr(_am_cfg, "hidden_dim", 768*2) if _am_cfg else 768*2

        # SNN action prediction model
        self.action_model = get_action_model(
            input_dim=qformer_out_dim, hidden_dim=hidden_dim, action_dim=action_dim
        )

        # GRU-gated FiLM edit model for robot-state conditioning
        self.edit_model = get_gruedit_model(
            input_dim=qformer_out_dim, hidden_dim=256, robot_state_dim=state_dim
        )

        self.L1_loss = nn.L1Loss()
        self.norm_stats = norm_stats




    def forward(
        self,
        examples: List[dict] = None,
        repeated_diffusion_steps: int = 4,
        **kwargs,
    ) -> Tuple:
        """
        Run a forward pass through the VLM, returning loss for training.

        Args:
            examples: List of training examples, each containing:
                - "image": Input images
                - "lang": Language instructions
                - "action": Ground truth actions [B, T, action_dim]
                - "state": Robot states [B, history_len, state_dim]
                - "solution" (optional): Chain-of-thought solutions

        Returns:
            Dictionary containing action_loss
        """
        # Extract data from examples
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        states = [example["state"] for example in examples]

        if "solution" in examples[0]:
            solutions = [example["solution"] for example in examples]
        else:
            solutions = None

        # Build inputs for vision-language model
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images, instructions=instructions, solutions=solutions
        )

        # Forward pass through VLM to get hidden states
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

        vlm_cot_loss = qwenvl_outputs.loss
        if vlm_cot_loss is None or torch.isnan(vlm_cot_loss):
            vlm_cot_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        # Single-pass action prediction via SNN
        # The SNN processes QFormer tokens as a temporal sequence; num_query_tokens
        # must equal the action prediction horizon so that shapes align.
        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer   = self.config.framework.layer_qformer.qformer_end_layer   if self.config else -1
            # action_latent_feature: [B, num_query_tokens, qformer_out_dim]
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32,
                                         device=action_latent_feature.device)

            # GRU-gated FiLM conditioning on robot state history
            edit_action_feature = self.edit_model(action_latent_feature, states_tensor)

            # SNN temporal prediction: [B, num_query_tokens, action_dim]
            predicted_actions = self.action_model.predict_action(edit_action_feature)

            # Ground truth: [B, T_action, action_dim]  (T_action == num_query_tokens)
            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32,
                                         device=predicted_actions.device)
            action_loss = self.L1_loss(predicted_actions, action_tensor)

        return {"action_loss": action_loss}


    def predict_action(
        self,
        batch_images: Union[Image, List[Image]],
        instructions: List[str],
        states: Optional[List[Sequence[float]]] = None,
        solutions: Union[Dict, List[Dict]] = None,
        unnorm_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Predict action from images and instructions.

        Args:
            batch_images: Input images (PIL Image or list of PIL Images)
            instructions: Task instructions (list of strings)
            states: Robot states history [B, T, 8], where last dim is [x,y,z,roll,pitch,yaw,gripper,pad]
            solutions: Optional solution dict for chain-of-thought
            unnorm_key: Key for unnormalization (if using norm_stats)
            cfg_scale: Classifier-free guidance scale (>1.0 enables CFG)
            use_ddim: Whether to use DDIM sampling
            num_ddim_steps: Number of DDIM steps

        Returns:
            Dictionary containing "normalized_actions" [B, T, 7]
        """
        # Resize images to model input size
        batch_images = resize_images(batch_images, target_size=(224, 224))

        # Build VLM inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        # Generate cognition features through VLM
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs.input_ids,
                attention_mask=qwen_inputs.attention_mask,
                pixel_values=qwen_inputs.pixel_values,
                image_grid_thw=qwen_inputs.image_grid_thw,
                labels=qwen_inputs.input_ids.clone(),
                output_hidden_states=True,
                return_dict=True,
            )

        # Single-pass action prediction via SNN
        with torch.autocast("cuda", dtype=torch.float32):
            start_layer = self.config.framework.layer_qformer.qformer_start_layer if self.config else -6
            end_layer   = self.config.framework.layer_qformer.qformer_end_layer   if self.config else -1
            action_latent_feature = self.layer_qformer(qwenvl_outputs.hidden_states[start_layer:end_layer])

            states_tensor = torch.tensor(
                np.array(states, dtype=np.float32),
                dtype=torch.float32,
                device=action_latent_feature.device
            )

            edit_action_feature = self.edit_model(action_latent_feature, states_tensor)
            samples = self.action_model.predict_action(edit_action_feature)

        normalized_actions = samples.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


def build_model_framework(config: dict = {}) -> NeuroVLA:
    """Build NeuroVLA model from config."""
    model = NeuroVLA(config=config)
    return model


if __name__ == "__main__":
    """
    Example usage for testing the model.

    This demonstrates how to:
    1. Load a pretrained model
    2. Prepare input data
    3. Run inference to predict actions
    """
    import pickle
    from omegaconf import OmegaConf

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Option 1: Load from pretrained checkpoint
    # model = NeuroVLA.from_pretrained("path/to/checkpoint.pt").to(device)
    # model = NeuroVLA.from_pretrained("/workspace/nature_submit/NeuroVLA/playground/Checkpoints/1104_spikevla_gru_xiaonao_goal_dualimage_nospike_ac8_768*2_yibu/checkpoints/steps_10000_pytorch_model.pt").to(device)
    # Option 2: Build from config
    # config = OmegaConf.load("path/to/config.yaml")
    # model = NeuroVLA(config).to(device)

    # Prepare sample data
    # Each sample should contain:
    # - "image": List of PIL Images
    # - "lang": Language instruction (string)
    # - "state": Robot state history [T, 8]
    # - "action": Ground truth actions [T, 7] (for training only)

    # Example data structure:
    samples = [
        {
            "image": [],  # List of PIL Images
            "lang": "pick up the red block",
            "state": np.zeros((16, 8)),  # [T, 8] state history
            "action": np.zeros((8, 7)),  # [T, 7] action sequence
        }
    ]
    # import pickle
    # from omegaconf import OmegaConf
    # with open("/workspace/samples_states.pkl", "rb") as f:
    #     samples = pickle.load(f)
    # device = torch.device("cuda:0")

    # Extract data for inference
    images = [sample["image"] for sample in samples]
    instructions = [sample["lang"] for sample in samples]
    states = [sample["state"] for sample in samples]

    # Run inference
    # with torch.inference_mode():
    #     result = model.predict_action(
    #         batch_images=images,
    #         instructions=instructions,
    #         states=states,
    #     )
    #     normalized_actions = result["normalized_actions"]
    #     print(f"Predicted actions shape: {normalized_actions.shape}")

    print("Test example ready. Uncomment the code above to run inference.")

