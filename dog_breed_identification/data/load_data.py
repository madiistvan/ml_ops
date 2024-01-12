from google.cloud import storage
from tests import _PATH_DATA, _PROJECT_ROOT
import hydra
import os

class LoadData:
    config_path = os.path.join(_PROJECT_ROOT, 'config')
    hydra.initialize(config_path=config_path, version_base=None)
    hparams = hydra.compose(config_name="train_config")
    storage_client = storage.Client()
    bucket_name = hparams.bucket_name
    bucket = storage_client.create_bucket(bucket_name)
    
    def load():
        blob = LoadData.bucket.blob()
        with blob.open("r") as f:
            data = f.read()
        return data
        
