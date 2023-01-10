import argparse
import sys

import torch
import click

from torch import nn, optim
from data import CorruptMnist
from model import MyAwesomeModel
from tqdm import tqdm
import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    
    batch_size = 128
    # load the training data 
    train_set = CorruptMnist(train=True)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # define type of loss 
    optimizer = optim.Adam(model.parameters(), lr=lr) # define optimizer

    epochs = 5

    train_losses = []
    train_accuracy = []

    for e in range(epochs):
        accuracy = 0 
        with tqdm(trainloader, unit="batch") as tepoch:
            for images, labels in trainloader:
                tepoch.set_description(f"Epoch {e}")

                # Forward pass 
                outputs = model(images)
                ps = torch.exp(outputs)
                loss = criterion(ps, labels) # train loss 
                train_losses.append(loss.item())

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))/len(trainloader)

                train_accuracy.append(accuracy)

                # Backpropogate 
                loss.backward()

                # Update parameters 
                optimizer.step()

                # Clear gradients 
                optimizer.zero_grad()

                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy.item()*100)

    torch.save(model.state_dict(), 'trained_model.pt')

    figure, axis = plt.subplots(2)
  
    # For loss
    axis[0].plot(train_losses,label="loss")
    axis[0].set_title("Training loss")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")
    
    # For accuracy
    axis[1].plot(train_accuracy,label="accuracy")
    axis[1].set_title("Training accuracy")
    axis[0].set_xlabel("iterations")
    axis[0].set_ylabel("loss")

    plt.show()

    return model
    


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

#   # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint)
    
    batch_size = 128
    # load the training data 
    test_set = CorruptMnist(train=False)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss() # define type of loss 

    test_losses = []
    test_accuracy = []
    accuracy = 0 
    with torch.no_grad():
        for images, labels in testloader:
            # Forward pass 
            outputs = model(images)
            ps = torch.exp(outputs)
            loss = criterion(ps, labels) # test loss 
            test_losses.append(loss.item())

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))/len(testloader)

            test_accuracy.append(accuracy)
    
    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    