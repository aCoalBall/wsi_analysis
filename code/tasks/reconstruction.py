import torch
import torch.nn as nn
import os
from tasks.get_features import load_features
from constants import *
from tasks.dataset import *
import h5py
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_size=1024):
        super(Decoder, self).__init__()

        # Initial Linear Layer to reshape vector to starting size
        self.fc = nn.Linear(1024, 256 * 7 * 7)  # Adjust size accordingly

        # Transposed Convolutional Layers
        self.conv_transpose1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_transpose5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Reshape input to suitable size
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)  # Reshape to (batch_size, channels, height, width)

        # Upsample and apply non-linearity
        x = F.relu(self.batch_norm1(self.conv_transpose1(x)))
        x = F.relu(self.batch_norm2(self.conv_transpose2(x)))
        x = F.relu(self.batch_norm3(self.conv_transpose3(x)))
        x = F.relu(self.batch_norm4(self.conv_transpose4(x)))
        x = torch.sigmoid(self.conv_transpose5(x))  # Use sigmoid to output values between 0 and 1
        return x

def reconstruction_test(model_name, decoder=Decoder(), num_epochs=20, batch_size=1):
    torch.manual_seed(42)
    #Set files
    h5_file_train = h5_dict['camelyon16']
    wsi_file_train = wsi_dict['camelyon16']
    h5_file_test = h5_dict['camelyon16_test']
    wsi_file_test = wsi_dict['camelyon16_test']

    #Set`models and datasets
    model = model_dict[model_name]()
    model.eval()
    autoencoder = nn.Sequential(model, decoder).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(decoder.parameters())

    #Train
    for _ in range(num_epochs):
        for h5 in os.listdir(h5_file_train):
            wsi = os.path.join(wsi_file_train, h5[:-3] + '.tif')
            h5 = h5py.File(os.path.join(h5_file_train, h5), 'r')
            coords = h5['coords'][:]
            dataset_train = Whole_Slide_Coords(coord_list=coords, wsi=wsi, transform=TRANSFORM)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            print(wsi)
            for i, sample in enumerate(dataloader_train):
                images, _ = sample
                images = images.to(DEVICE)
                reconstructed_images = autoencoder(images)
                print(images.shape)
                print(reconstructed_images.shape)
                loss = loss_fn(reconstructed_images, images)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                break
            break

    #Test
    decoder.eval()
    total_loss = 0
    count = 0
    for h5 in os.listdir(h5_file_test):
        wsi = os.path.join(wsi_file_test, h5[:-3] + '.tif')
        print(wsi)
        h5 = h5py.File(os.path.join(h5_file_test, h5), 'r')
        coords = h5['coords'][:]
        dataset_test = Whole_Slide_Coords(coord_list=coords, wsi=wsi, transform=TRANSFORM)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

        for i, sample in enumerate(dataloader_test):
            images, _ = sample
            images = images.to(DEVICE)
            reconstructed_images = autoencoder(images)
            loss = loss_fn(reconstructed_images, images)
            total_loss += loss
            count += batch_size
            break
    avg_loss = total_loss / count
    print('loss of %s : '%model_name, avg_loss)
