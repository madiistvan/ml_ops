from fastapi import FastAPI, UploadFile
import timm
import torch
from google.cloud import storage
import torchvision
import pandas as pd
from typing import List

def get_params():
    BUCKET_NAME = "dog-breed-identification-model" 
    MODEL_FILE = "exp2.pth"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)

    with open(MODEL_FILE, 'wb') as file_obj:
        blob.download_to_file(file_obj)
    params = torch.load(MODEL_FILE, map_location=torch.device('cpu'))

    return params


def get_model():
    params = get_params()
    model = timm.create_model('mobilenetv3_large_100', num_classes=120)
    model.load_state_dict(params)
    model.eval()

    return model

def predict_one(imagefile):
    try:
        contents = imagefile.file.read()
        with open("image.png", "wb") as f:
            f.write(contents)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((200, 200))
        ])

        image = torchvision.io.read_image("image.png")
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.float()
        output = model(image)
        output = torch.softmax(output, dim=1)
        probability, breed_idx = torch.max(output, dim=1)
        predicted_breed = breeds[breeds["id"] == breed_idx.item()]["breed"].values[0]

        return {
            "filename": imagefile.filename,
            "probability": probability.item(),
            "breed_idx": breed_idx.item(),
            "breed": predicted_breed,
            "status": "success",
            "message": "Cute puppy detected"
        }
    except Exception as e:
        print(e)
        return {
            "filename": imagefile.filename,
            "probability": -1,
            "breed_idx": -1,
            "breed": "-",
            "status": "error",
            "message": "There was an error predicting for this image. Perhaps the file is not an image or the format of the image is corrupted."
        }
    finally:
        imagefile.file.close()

app = FastAPI()
model = get_model()
breeds = pd.read_csv('src/breeds.csv', names=["id", "breed"])

@app.get("/")
async def root():
    return {
        "message": "Hi"
    }


@app.post("/predict")
def predict(imagefiles: List[UploadFile]):
    predictions = []
    
    for imagefile in imagefiles:
        predictions.append(predict_one(imagefile))
    
    return predictions
