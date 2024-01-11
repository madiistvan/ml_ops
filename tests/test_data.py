import os

import pytest
import torch
from tests import _PATH_DATA, _PROJECT_ROOT
import hydra
from torch.utils.data import DataLoader

config_path = os.path.join(_PROJECT_ROOT, 'config')

hydra.initialize(config_path=config_path, version_base=None)
hparams = hydra.compose(config_name="train_config")
dataset_path = hparams.data_path
batch_size = hparams.batch_size

train = torch.load(f'{dataset_path}/train.pt')
val = torch.load(f'{dataset_path}/val.pt')
# Create dataloaders
train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True)
test_ldr = DataLoader(val, batch_size=batch_size, shuffle=True)


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    assert len(train_ldr) > 0, "Dataset did not have the correct number of samples"
    assert len(test_ldr) > 0, "Dataset did not have the correct number of samples"
    assert len(train_ldr) > len(
        test_ldr), "Train set should be larger than test set"
    assert train_ldr.dataset[0][0].shape == torch.Size(
        [3, 200, 200]), "Input data shape is incorrect"
