# Step 1: Download the data from Kaggle (https://www.kaggle.com/competitions/dog-breed-identification/data)
# Step 2: Put it in the data/raw folder and unzip it
import torch
import torchvision
import pandas as pd
import os


def list_files(path):
    return os.listdir(path)


def get_labels(files, labels):
    return [labels[labels['id'] == file.split('.')[0]]['breed'] for file in files]


def process_data():
    # Load the data
    data_dir = "data/raw/dog-breed-identification"

    labels = pd.read_csv(f'{data_dir}/labels.csv')

    # Split the data into train and test

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    train_files = list_files(f'{data_dir}/train')
    test_files = list_files(f'{data_dir}/test')

    train_labels = get_labels(train_files, labels)
    test_labels = get_labels(test_files, labels)

    image = torchvision.io.read_image(
        "data/raw/dog-breed-identification/train/84accc2dc9f5bb3ebee89fe1bf23639c.jpg")

    for train_file in train_files:
        print(f'{data_dir}/train/{train_file}')
        image = torchvision.io.read_image(f'{data_dir}/train/{train_file}')

    print(train_images.shape)

    for test_file in test_files:
        test_images.append(torchvision.io.read_image(
            f'{data_dir}/test/{test_file}'))

    # Save the data


if __name__ == '__main__':
    process_data()
