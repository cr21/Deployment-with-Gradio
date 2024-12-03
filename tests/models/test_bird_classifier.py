import pytest
import sys
import rootutils
import torch
from hydra import initialize, compose
from hydra.utils import instantiate


# Setup the root directory and add it to the Python path
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(str(root))
from src.model_builder.timm_classifier import TimmClassifier
@pytest.fixture(scope="module")
def bird_classifier_config():
    with initialize(version_base=None, config_path=str("../../configs")):
        cfg = compose(config_name="train", overrides=["model=timm_classifier"])
    return cfg.model

def test_bird_classifier_instantiation(bird_classifier_config):
    model = instantiate(bird_classifier_config)
    assert isinstance(model, TimmClassifier)

def test_bird_classifier_forward(bird_classifier_config):
    model = instantiate(bird_classifier_config)
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, bird_classifier_config.num_classes)

def test_bird_classifier_training_step(bird_classifier_config):
    model = instantiate(bird_classifier_config)
    batch_size = 4
    channels = 3
    height, width = 224, 224
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, bird_classifier_config.num_classes, (batch_size,))
    batch = (x, y)
    loss = model.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

# Add more tests for validation_step, test_step, configure_optimizers, etc.
