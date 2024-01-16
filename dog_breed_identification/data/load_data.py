from google.cloud import storage
import hydra
from hydra.core.global_hydra import GlobalHydra
import os
import io
import torch


class LoadData:
    config_path = os.path.join("/", 'config')
    if not GlobalHydra().is_initialized():
        hydra.initialize(config_path=config_path, version_base=None)
    hparams = hydra.compose(config_name="train_config")
    storage_client = storage.Client()
    bucket_name = hparams.data_bucket_name
    bucket = storage_client.bucket(bucket_name)

    def load(blob_name):
        blob = LoadData.bucket.blob(blob_name)
        # rb important otherwise it will be read as string and fails with some character encoding error
        with blob.open("rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data))
