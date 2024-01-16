from google.cloud import storage
import hydra
import os
import io
import torch
from dog_breed_identification import train_config

class LoadData:
    storage_client = storage.Client()
    bucket_name = train_config.data_bucket_name
    bucket = storage_client.bucket(bucket_name)

    def load(blob_name):
        blob = LoadData.bucket.blob(blob_name)
        # rb important otherwise it will be read as string and fails with some character encoding error
        with blob.open("rb") as f:
            data = f.read()
        return torch.load(io.BytesIO(data))

if __name__ == "__main__":
    print(LoadData())