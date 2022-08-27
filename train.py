"""
This file will train the models created in the model file and generate results
"""
# import all libraries
import torch
import torchvision
from torch import nn
from torch import optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator

# define the hyperparameters and variables
LEARNING_RATE = 0.0005
BATCH_SIZE = 256
IMAGE_SIZE = 64
EPOCHS = 250
image_channels = 1
noise_channels = 256
gen_features = 64
disc_features = 64

# set everything to GPU
device = torch.device("cuda")

# define the transform
data_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
])

# load the dataset 
dataset = FashionMNIST(root="dataset/", train=True, transform=data_transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# load models
gen_model  = Generator(noise_channels, image_channels, gen_features).to(device)
disc_model = Discriminator(image_channels, disc_features).to(device)

# setup optimizers for both models
gen_optimizer = optim.Adam(gen_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(disc_model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# define the loss function 
criterion = nn.BCELoss()

# make both models train
gen_model.train()
disc_model.train()

# deifne labels for fake images and real images for the discriminator
fake_label = 0
real_label = 1

# define a fixed noise 
fixed_noise = torch.randn(64, noise_channels, 1, 1).to(device)

# make the writers for tensorboard
writer_real = SummaryWriter(f"runs/fashion/test_real")
writer_fake = SummaryWriter(f"runs/fashion/test_fake")

# define a step
step = 0

print("Start training...")

# loop over all epochs and all data
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(dataloader):
        # set the data to cuda
        data = data.to(device)

        # get the batch size 
        batch_size = data.shape[0]

        # Train the discriminator model on real data
        disc_model.zero_grad()
        label = (torch.ones(batch_size) * 0.9).to(device)
        output = disc_model(data).reshape(-1)
        real_disc_loss = criterion(output, label)
        d_x = output.mean().item()

        # train the disc model on fake (generated) data
        noise = torch.randn(batch_size, noise_channels, 1, 1).to(device)
        fake = gen_model(noise)
        label = (torch.ones(batch_size) * 0.1).to(device)
        output = disc_model(fake.detach()).reshape(-1)
        fake_disc_loss = criterion(output, label)

        # calculate the final discriminator loss
        disc_loss = real_disc_loss + fake_disc_loss

        # apply the optimizer and gradient
        disc_loss.backward()
        disc_optimizer.step()

        # train the generator model
        gen_model.zero_grad()
        label = torch.ones(batch_size).to(device)
        output = disc_model(fake).reshape(-1)
        gen_loss = criterion(output, label)
        # apply the optimizer and gradient
        gen_loss.backward()
        gen_optimizer.step()

        # print losses in console and tensorboard
        if batch_idx % 50 == 0:
            step += 1

            # print everything
            print(
                f"Epoch: {epoch} ===== Batch: {batch_idx}/{len(dataloader)} ===== Disc loss: {disc_loss:.4f} ===== Gen loss: {gen_loss:.4f}"
            )

            ### test the model
            with torch.no_grad():
                # generate fake images 
                fake_images = gen_model(fixed_noise)
                # make grid in the tensorboard
                img_grid_real = torchvision.utils.make_grid(data[:40], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake_images[:40], normalize=True)

                # write the images in tensorbaord
                writer_real.add_image(
                    "Real images", img_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Generated images", img_grid_fake, global_step=step
                )