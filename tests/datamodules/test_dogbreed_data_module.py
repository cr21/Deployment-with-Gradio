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
def dogbreed_datamodule_config():
    with initialize(version_base=None, config_path=str("../../configs")):
        cfg = compose(config_name="train", overrides=["data=dogbreed"])
    return cfg.data

def test_dogbreed_datamodule_instantiation(dogbreed_datamodule_config):
    datamodule = instantiate(dogbreed_datamodule_config)
    assert isinstance(datamodule, GenericDataModule)

def test_dogbreed_datamodule_setup(dogbreed_datamodule_config):
    datamodule = instantiate(dogbreed_datamodule_config)
    datamodule.setup()
    assert hasattr(datamodule, 'train_dataset')
    assert hasattr(datamodule, 'val_dataset')
    assert hasattr(datamodule, 'test_dataset')

def test_dogbreed_datamodule_train_dataloader(dogbreed_datamodule_config):
    datamodule = instantiate(dogbreed_datamodule_config)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    assert len(train_dataloader) > 0

# Add more tests for val_dataloader, test_dataloader, etc.
