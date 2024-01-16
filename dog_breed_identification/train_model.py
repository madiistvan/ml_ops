import click
import torch
from torch.utils.data import DataLoader
import timm
import hydra
import wandb
import pandas as pd
from dog_breed_identification.models.model import Model
from dog_breed_identification.data.load_data import LoadData

# Init wandb
wandb.init(project="dog-breed-identification")

# Get config
hydra.initialize(config_path="config", version_base=None)

# Hyperparameters
hparams = hydra.compose(config_name="train_config")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = hparams.data_path

train = LoadData.load(f'{dataset_path}/train.pt')
val = LoadData.load(f'{dataset_path}/val.pt')

# Create dataloaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)


# Load model
model = Model()
model.to(device)
model.train()


def train(num_epochs, learning_rate, batch_size, model_name):
    # Load data
    train = torch.load(f'{dataset_path}/train.pt')
    # Create dataloaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    torch.save(model.state_dict(), f'models/{model_name}.pth')


def evaluate(batch_size):
    val = torch.load(f'{dataset_path}/val.pt')
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

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

@click.command()
@click.option('--num_epochs', default=hparams.epochs, type=int, help='Set the num_epochs')
@click.option('--learning_rate', default=hparams.lr, type=float, help='Set the learning_rate')
@click.option('--batch_size', default=hparams.batch_size, type=int, help='Set the batch_size')
@click.option('--model_name', default=hparams.name, type=str, help='Set the model file name')
def main(num_epochs, learning_rate, batch_size, model_name):
    train(num_epochs, learning_rate, batch_size, model_name)
    evaluate(batch_size)


if __name__ == '__main__':
    main()
