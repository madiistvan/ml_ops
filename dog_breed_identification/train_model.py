import torch
from torch.utils.data import DataLoader
import timm
import hydra
import wandb
import pandas as pd
from dog_breed_identification.models.model import Model

# Init wandb
wandb.init(project="dog-breed-identification")

# Get config
hydra.initialize(config_path="config", version_base=None)
hparams = hydra.compose(config_name="train_config")

# Hyperparameters
num_epochs = hparams.epochs
learnig_rate = hparams.lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = hparams.batch_size
dataset_path = hparams.data_path

with open(f'{dataset_path}/train.pt', 'r') as f:
    train = torch.load(f)
with open(f'{dataset_path}/val.pt', 'r') as f:
    val = torch.load(f)

# Create dataloaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

# Load model
model = Model()
model.to(device)
model.train()

# Loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate)


def train():
    for epoch in range(num_epochs):
        overall_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            preds = model(x)
            loss = loss_fn(preds, y)
            overall_loss += loss.item()

            if batch_idx % 1 == 0:
                print(
                    f"Batch Index: {batch_idx} Loss: {loss.item() / batch_size}")
                wandb.log({"Train loss:": loss.item() / batch_size})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "complete!", "\tAverage Loss: ",
              overall_loss / ((batch_idx + 1) * batch_size),
              )

    print("Training complete, saving model...")
    torch.save(model.state_dict(), f'models/{hparams.name}.pth')


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            preds = model(x)
            total += y.size(0)

            predicted_index = torch.argmax(preds.data, 1)
            correct_index = torch.argmax(y, 1)
            correct += (predicted_index == correct_index).sum().item()
        print('Accuracy of the model on the validation set: {} %'.format(
            100 * correct / total))

    labels = pd.read_csv('data/processed/breeds.csv', names=["id", "breed"])
    for i in range(5):
        correct_breed = labels[labels["id"] ==
                               correct_index[i].item()]["breed"].values[0]
        predicted_breed = labels[labels["id"] ==
                                 predicted_index[i].item()]["breed"].values[0]

        wandb.log({
            "Examples": wandb.Image(x[i], caption=f"Predicted: {predicted_breed}, Correct: {correct_breed}"),
        })


if __name__ == '__main__':
    train()
    evaluate()
