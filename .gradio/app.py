import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path

class FoodImageClassifier:
    def __init__(self, model_dir="traced_models/food_101_vit_tiny",
                     model_file_name="model.pt",
                     labels_path='food_101_classes.txt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the traced model
        model_full_path = Path(model_dir,model_file_name)
        self.model = torch.jit.load(model_full_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the same transforms used during training/testing
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Load labels from file
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Create prediction dictionary
        return {
            self.labels[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }

# Create classifier instance
classifier = FoodImageClassifier()

# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Food classifier",
    description="Upload an image to classify Food Images",
    examples=[
        ["sample_data/apple_pie.jpg"],
        ["sample_data/pizza.jpg"]
    ]
)


if __name__ == "__main__":
    demo.launch()