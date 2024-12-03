import pytest
import sys
from pathlib import Path
import rootutils

# Setup the root directory and add it to the Python path
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(str(root))

from hydra import initialize, compose
from hydra.utils import instantiate
from src.datamodules.data_module import GenericDataModule

print(f"Project root: {root}")
print(f"Python path: {sys.path}")

@pytest.fixture(scope="module")
def bird_datamodule_config():
    with initialize(version_base=None, config_path=str("../../configs")):
        cfg = compose(config_name="train", overrides=["data=birddata"])
    return cfg.data

def test_bird_datamodule_instantiation(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    assert isinstance(datamodule, GenericDataModule)

def test_bird_datamodule_setup(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    datamodule.setup()
    assert hasattr(datamodule, 'train_dataset')
    assert hasattr(datamodule, 'val_dataset')
    assert hasattr(datamodule, 'test_dataset')

def test_bird_datamodule_train_dataloader(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    assert len(train_dataloader) > 0

def test_bird_datamodule_val_dataloader(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    datamodule.setup()
    val_dataloader = datamodule.val_dataloader()
    assert len(val_dataloader) > 0

def test_bird_datamodule_test_dataloader(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    assert len(test_dataloader) > 0

def test_bird_datamodule_class_names(bird_datamodule_config):
    datamodule = instantiate(bird_datamodule_config)
    datamodule.setup('train')
    assert datamodule.class_names is not None
