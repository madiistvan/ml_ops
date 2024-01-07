import torch
from torch.utils.data import DataLoader
import timm

# Hyperparameters
num_epochs = 1
learnig_rate = 0.003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
dataset_path = 'data/processed'

# Load data
train = torch.load(f'{dataset_path}/train.pt')
val = torch.load(f'{dataset_path}/val.pt')

# Create dataloaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

# Load model
model = timm.create_model('mobilenetv3_large_100',
                          pretrained=True, num_classes=120)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch", epoch + 1, "complete!", "\tAverage Loss: ",
              overall_loss / ((batch_idx + 1) * batch_size),
              )

    print("Training complete, saving model...")
    torch.save(model.state_dict(), 'models/model.pth')


def evaluate():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            preds = model(x)
            _, predicted = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print('Accuracy of the model on the validation set: {} %'.format(
            100 * correct / total))


if __name__ == '__main__':
    train()
    evaluate()
