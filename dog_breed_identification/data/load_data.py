from google.cloud import storage
from tests import _PATH_DATA, _PROJECT_ROOT
import hydra
from hydra.core.global_hydra import GlobalHydra
import os
import io
import torch
class LoadData:
    config_path = os.path.join(_PROJECT_ROOT, 'config')
    if not GlobalHydra().is_initialized():
        hydra.initialize(config_path=config_path, version_base=None)
    hparams = hydra.compose(config_name="train_config")
    storage_client = storage.Client()
    bucket_name = hparams.data_bucket_name
    bucket = storage_client.bucket(bucket_name)
    
    def load(blob_name):
        blob = LoadData.bucket.blob(blob_name)
        with blob.open("rb") as f: # rb important otherwise it will be read as string and fails with some character encoding error
            data = f.read()
        return torch.load(io.BytesIO(data))
        
