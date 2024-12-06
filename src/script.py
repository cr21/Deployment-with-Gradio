import os
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig
import torch
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint if specified
    if cfg.get("ckpt_path"):
        log.info(f"Loading checkpoint: {cfg.ckpt_path} model.__class__ {model.__class__}")
        model = model.__class__.load_from_checkpoint(cfg.ckpt_path)
    
    # Set model to eval mode and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model = model.to(device)
    model.eval()
    img_w = cfg.IMG_W or 224
    img_h = cfg.IMG_H or 224
    # Create example input
    example_input = torch.randn(1, 3, img_h, img_w).to(device)  # Move input to same device as model
    
    # Trace the model
    log.info(f"Tracing model on device: {device}")
    traced_model = model.to_torchscript(method="trace", example_inputs=example_input)
    
    # Move traced model to CPU before saving
    traced_model = traced_model.cpu()
    
    # Create output directory if it doesn't exist
    output_dir = Path(f"traced_models/{cfg.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the traced model
    output_path = output_dir / "model.pt"
    torch.jit.save(traced_model, output_path)
    log.info(f"Traced model saved to: {output_path}")

if __name__ == "__main__":
    main()
