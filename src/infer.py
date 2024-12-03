import os
from pathlib import Path
import random
from typing import List, Tuple
import rootutils
import torch
# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import logging
from src.utils.logging_utils import setup_logger
from src.utils.s3_utility import download_model_from_s3, remove_files, upload_file_to_s3

# Set up logging
log = logging.getLogger(__name__)

def pred_and_plot_image(
    cfg:DictConfig,
    model: pl.LightningModule,
    image_path: str,
    results_dir: str,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform=None
):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = Image.open(image_path).convert("RGB")
    if transform:
        target_image = transform(img)
    else:
        target_image = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),])(img)
    model.to(device)
    model.eval()
    with torch.no_grad():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    log.info(target_image_pred_label, target_image_pred_probs)
    log.info(f"class label actual {Path(image_path).parent.name}")
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    title = f"Actual: {Path(image_path).parent.name}, Pred: {class_names[target_image_pred_label.item()]} | Prob: {target_image_pred_probs.max().item():.3f}"
    plt.title(title)
    plt.axis(False)
    result_path = Path(results_dir) / f"{Path(image_path).stem}_{class_names[target_image_pred_label.item()]}.png"
    plt.savefig(result_path)
    try:
        upload_file_to_s3(result_path, cfg.inference.s3_prediction_bucket_location)
    except Exception as exp:
        print(f"Trying to store prediction image to s3 location result_path {result_path} bucket location {cfg.inference.s3_prediction_bucket_location}")
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    log_dir = Path(cfg.paths.log_dir)
    # set up logger
    setup_logger(log_dir/"inference.log")
    print(f"checkpoint path ", cfg.ckpt_path)
    # Set up paths
    ckpt_path = Path(cfg.ckpt_path)
    save_dir = Path(cfg.save_dir)
    data_dir = Path(cfg.data_dir)
    results_dir = save_dir 
    results_dir.mkdir(parents=True, exist_ok=True)
    remove_files(os.path.dirname(cfg.ckpt_path),pattern="*.ckpt")
    download_model_from_s3(cfg.ckpt_path, cfg.inference.s3_model_bucket_location, cfg.inference.s3_model_bucket_folder_location)
    print(f"ckptpath {ckpt_path}")
    # Load model
    log.info(f"Loading model from {ckpt_path}")
    model = hydra.utils.instantiate(cfg.model)
    model = model.__class__.load_from_checkpoint(ckpt_path)
    # model.to('auto')

    # Set up data module
    log.info("Setting up data module")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    data_module.setup("test")

    # Get class names
    class_names = data_module.class_names
    print(class_names)
    # Get inference image paths
    inference_image_path_list = list(data_dir.glob(cfg.inference.inference_glob_pattern))
    test_image_path_samples = random.sample(inference_image_path_list, k=cfg.num_samples)
    print("Total Images: ", len(inference_image_path_list))
    # Perform inference
    log.info(f"Performing inference on {cfg.num_samples} samples")
    remove_files(os.path.dirname(cfg.save_dir),pattern="*.png")
    for image_path in test_image_path_samples:
        pred_and_plot_image(
            cfg=cfg,
            model=model,
            image_path=str(image_path),
            results_dir=results_dir,
            class_names=class_names,
            transform=data_module.test_transform,
            image_size=(cfg.img_size, cfg.img_size)
        )

    log.info(f"Inference complete. Results saved in {results_dir}")

if __name__ == '__main__':
    main()