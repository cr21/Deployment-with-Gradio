import pytest
from unittest.mock import Mock, patch
import rootutils
import hydra
from omegaconf import DictConfig, OmegaConf

# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

from src.train import train, test

@pytest.fixture
def mock_cfg():
    return OmegaConf.create({
        "trainer": {"max_epochs": 1},
        "model": {"_target_": "pytorch_lightning.LightningModule"},
        "data": {"_target_": "pytorch_lightning.LightningDataModule"},
        "test": True,  # Add this to enable testing
    })

@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.fit.return_value = None
    trainer.test.return_value = [{"test_accuracy": 0.95}]
    trainer.callback_metrics = {"loss": 0.5}
    return trainer

@pytest.fixture
def mock_model():
    return Mock()

@pytest.fixture
def mock_datamodule():
    return Mock()

@pytest.mark.parametrize("config_name", ["test_configs"])
def test_test_fn(config_name, mock_cfg, mock_trainer, mock_model, mock_datamodule):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
        cfg = OmegaConf.merge(mock_cfg, cfg)
        
        # Set up the mock trainer with a checkpoint callback
        mock_trainer.checkpoint_callback = Mock()
        mock_trainer.checkpoint_callback.best_model_path = "path/to/best/model.ckpt"

        # Call the test function
        with patch('src.train.log'):  # Mock the logger to avoid actual logging
            metrics = test(cfg, mock_trainer, mock_model, mock_datamodule)

        # Assert that the trainer's test method was called with the correct arguments
        mock_trainer.test.assert_called_once_with(mock_model, mock_datamodule, ckpt_path="path/to/best/model.ckpt")

        # Assert that the returned metrics match the expected test metrics
        assert metrics == [{"test_accuracy": 0.95}]

@pytest.mark.parametrize("config_name", ["test_configs"])
def test_train(config_name, mock_cfg, mock_trainer, mock_model, mock_datamodule):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name=config_name)
        cfg = OmegaConf.merge(mock_cfg, cfg)
        
        # Call the train function
        with patch('src.train.log'):  # Mock the logger to avoid actual logging
            metrics = train(cfg, mock_trainer, mock_model, mock_datamodule)

        # Assert that the trainer's fit method was called
        mock_trainer.fit.assert_called_once_with(mock_model, mock_datamodule)

        # Assert that the returned metrics match the trainer's callback_metrics
        assert metrics == mock_trainer.callback_metrics

if __name__ == "__main__":
    pytest.main([__file__])
