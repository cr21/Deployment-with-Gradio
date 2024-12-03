import pytest
from unittest.mock import Mock, patch
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.train import test

@pytest.fixture
def mock_cfg():
    return OmegaConf.create({
        "trainer": {"max_epochs": 1},
        "model": {"_target_": "pytorch_lightning.LightningModule"},
        "data": {"_target_": "pytorch_lightning.LightningDataModule"},
        "test": True,
    })

@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.test.return_value = [{"test_accuracy": 0.95}]
    trainer.checkpoint_callback = Mock()
    trainer.checkpoint_callback.best_model_path = "path/to/best/model.ckpt"
    return trainer

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def mock_datamodule():
    return Mock()

# @pytest.mark.parametrize("config_name", ["test_test_cfg"])
# def test_test_function(config_name, mock_cfg, mock_trainer, mock_model, mock_datamodule):
#     with hydra.initialize(version_base=None, config_path="../configs"):
#         cfg = hydra.compose(config_name=config_name)
#         cfg = OmegaConf.merge(mock_cfg, cfg)
        
#         # Call the test function
#         with patch('src.train.log'):  # Mock the logger to avoid actual logging
#             metrics = test(cfg, mock_trainer, mock_model, mock_datamodule)

#         # Assert that the trainer's test method was called with the correct arguments
#         mock_trainer.test.assert_called_once_with(mock_model, mock_datamodule, ckpt_path="path/to/best/model.ckpt")

#         # Assert that the returned metrics match the expected test metrics
#         assert metrics == [{"test_accuracy": 0.95}]

@pytest.mark.parametrize("config_name", ["test_test_cfg"])
def test_test_function_no_checkpoint(config_name, mock_cfg, mock_trainer, mock_model, mock_datamodule):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
        cfg = OmegaConf.merge(mock_cfg, cfg)
        
        # Remove the checkpoint callback to test the case where no checkpoint is found
        mock_trainer.checkpoint_callback = None

        # Call the test function
        with patch('src.train.log'):  # Mock the logger to avoid actual logging
            metrics = test(cfg, mock_trainer, mock_model, mock_datamodule)

        # Assert that the trainer's test method was called without a checkpoint path
        mock_trainer.test.assert_called_once_with(mock_model, mock_datamodule)

        # Assert that the returned metrics match the expected test metrics
        assert metrics == [{"test_accuracy": 0.95}]

if __name__ == "__main__":
    pytest.main([__file__])
