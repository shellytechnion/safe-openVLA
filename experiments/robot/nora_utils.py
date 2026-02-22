"""Utils for evaluating Nora VLA policies."""

from __future__ import annotations

import numpy as np
from PIL import Image

from experiments.robot.libero.inference.modelling_expert import VLAWithExpert


def get_nora(cfg):
    model_path = cfg.pretrained_checkpoint or "declare-lab/nora-1.5-libero"
    print(f"[*] Instantiating Nora model from {model_path}")
    model = VLAWithExpert.from_pretrained(model_path)
    model.unnorm_key = getattr(cfg, "unnorm_key", None)
    return model


def get_nora_action(model: VLAWithExpert, obs, task_label: str, num_steps: int = 10):
    image = obs["full_image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image)).convert("RGB")
    action = model.sample_actions(image, task_label, num_steps=num_steps)
    return np.asarray(action, dtype=np.float32)

