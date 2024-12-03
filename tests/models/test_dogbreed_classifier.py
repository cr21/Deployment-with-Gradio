import pytest
import sys
import rootutils
import torch
from hydra import initialize, compose
from hydra.utils import instantiate


# Setup the root directory and add it to the Python path
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# sys.path.append(str(root))
from src.model_builder.timm_classifier import TimmClassifier
print(f"Project root: {root}")

@pytest.fixture(scope="module")
def dogbreed_classifier_config():
    with initialize(version_base=None, config_path=str("../../configs")):
        cfg = compose(config_name="train", overrides=["model=timm_classifier"])
    return cfg.model

def test_dogbreed_classifier_instantiation(dogbreed_classifier_config):
    model = instantiate(dogbreed_classifier_config)
    assert isinstance(model, TimmClassifier)

def test_dogbreed_classifier_forward(dogbreed_classifier_config):
    model = instantiate(dogbreed_classifier_config)
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, dogbreed_classifier_config.num_classes)

# Add more tests similar to test_bird_classifier.py
