import torch
import timm
from PIL import Image
import io
import base64
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import gradio as gr
from src.utils.s3_utility import download_model_from_s3, read_s3_file
from torchvision import transforms
from PIL import Image


class ImageClassifier:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create checkpoints directory if it doesn't exist
        os.makedirs(f"checkpoints/{self.cfg.gradio_deployment.name}", exist_ok=True)
        os.makedirs(f"labels/{self.cfg.gradio_deployment.name}", exist_ok=True)

        # Only download if model doesn't exist
        model_path = f"checkpoints/{self.cfg.gradio_deployment.name}/{self.cfg.ckpt_path.split(os.path.sep)[-1]}"
        if not os.path.exists(model_path):
            download_model_from_s3(
                local_file_name=self.cfg.ckpt_path.split(os.path.sep)[-1],
                bucket_name=self.cfg.gradio_deployment.s3_model_bucket_location,
                s3_folder=self.cfg.gradio_deployment.s3_model_bucket_folder_location,
                output_location=f"checkpoints/{self.cfg.gradio_deployment.name}"
            )
        else:
            print(f"Model already exists locally at {model_path}")

        # Only download labels if they don't exist
        labels_path = f"labels/{self.cfg.gradio_deployment.name}/{self.cfg.gradio_deployment.s3_labels_file_name}"
        if not os.path.exists(labels_path):
            self.labels = read_s3_file(
                file_name=self.cfg.gradio_deployment.s3_labels_file_name,
                bucket_name=self.cfg.gradio_deployment.s3_labels_bucket_location,
                s3_folder=self.cfg.gradio_deployment.s3_labels_bucket_folder_location
            ).strip().split('\n')
            # Save labels locally
            with open(labels_path, 'w') as f:
                f.write('\n'.join(self.labels))
        else:
            print(f"Labels already exist locally at {labels_path}")
            # Read labels from local file
            with open(labels_path, 'r') as f:
                self.labels = f.read().strip().split('\n')

        # Load checkpoint to get stored parameters
        checkpoint = torch.load(self.cfg.ckpt_path, map_location=self.device)
        # Create model using base_model from config and checkpoint parameters
        model_name = checkpoint['hyper_parameters']['base_model'] or self.cfg.batch_deployment.name
        num_classes = checkpoint['hyper_parameters']['num_classes'] or len(self.labels)
        self.model = timm.create_model(
            model_name=model_name,  
            num_classes=num_classes,
            pretrained=checkpoint['hyper_parameters']['pretrained']
        )
        # Remove 'model.' prefix from state dict keys
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('model.', '')
            new_state_dict[new_key] = state_dict[key]
        
        # Load the modified state dict
        self.model.load_state_dict(new_state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()

        # Replace the timm transforms with your test transforms
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Set up Done!")

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None
        
        # Preprocess image
        img = Image.fromarray(image).convert('RGB')
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        return {
            self.labels[idx.item()]: float(prob)
            for prob, idx in zip(top5_prob, top5_catid)
        }



@hydra.main(version_base=None, config_path="../configs", config_name="gradio_deployment")
def main(cfg: DictConfig):
    # Create classifier instance
    classifier = ImageClassifier(cfg)

    # Create Gradio interface
    demo = gr.Interface(
        fn=classifier.predict,
        inputs=gr.Image(),
        outputs=gr.Label(num_top_classes=5),
        title=f"Basic  Food Image Classification with {cfg.gradio_deployment.name}",
        description=f"Upload an food image to classify it using the {cfg.gradio_deployment.base_model} model",
        examples=[
            ["sample_data/153195.jpg"],
            ["sample_data/2395631.jpg"]
        ]
    )
    demo.launch() 


if __name__ == "__main__":
    main()    
