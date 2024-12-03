import pytest
import os
from hydra import initialize, compose
import rootutils
# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

@pytest.mark.parametrize("config_name", ["train", "eval", "infer"])
def test_config_composition(config_name):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=config_name)
    assert cfg is not None

def test_train_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train")
    assert "model" in cfg
    assert "data" in cfg
    assert "trainer" in cfg

def test_eval_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="eval")
    assert "model" in cfg
    assert "data" in cfg
    assert "ckpt_path" in cfg

def test_infer_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer")
    assert "model" in cfg
    assert "data" in cfg
    assert "ckpt_path" in cfg
    assert "num_samples" in cfg
