import pytest
import rootutils
from hydra import initialize, compose
from hydra.utils import instantiate


# Setup the root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")
from src.infer import main as infer_main

@pytest.fixture(scope="module")
def infer_config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer")
    return cfg

def test_infer_main(infer_config):
    # Instantiate the necessary components
    model = instantiate(infer_config.model)
    data_module = instantiate(infer_config.data)

    # Call the main function
    infer_main(infer_config)

    # Assert that the necessary attributes exist
    assert hasattr(model, 'load_from_checkpoint')
    assert hasattr(data_module, 'setup')
