'''
- Program to build a deep neural network model
- Train the model
- Save and load the model
- Test the model

'''

# Import statements
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, transforms, Grayscale, Resize, ToPILImage
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2


# Class definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 filters of size 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 filters of size 5x5
        self.fc1 = nn.Linear(16820, 500)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16820) # Flatten
        x = F.relu(self.fc1(x)) # Fully connected layers
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # softmax activation function


# Function to train the model
def train_model(myModel, my_train_dataset, myOptimizer, epochs):
  train_losses = []
  train_acc = []
  for epoch in range(epochs):
    myModel.train() # train model
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in my_train_dataset:
      myOptimizer.zero_grad()
      outputs = myModel(inputs)
      loss = F.nll_loss(outputs, labels) # negative loglikelihood loss
      loss.backward()
      myOptimizer.step()

      train_loss += loss.item()
      _, predicted = torch.max(outputs, 1)

      correct += (predicted == labels).sum().item()
      total += labels.size(0)

    epoch_loss = train_loss / len(my_train_dataset)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_acc.append(epoch_acc)

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}\n')

  return train_losses, train_acc # Return training loss and accuracy


# Model testing
def testing(loaded_model, image_tensor):
  loaded_model.eval() # Evaluation mode on

  with torch.no_grad():
    image_output = loaded_model(image_tensor)
  _, predicted = torch.max(image_output, 1) # Predict input image

  # Display predicted image
  plt.figure(figsize=(2, 2))
  plt.imshow(image_tensor[0], cmap='gray', interpolation=None)
  if(predicted.item()==0):
    pred_title = "Aframe"
  elif(predicted.item()==1):
    pred_title = "Jersey Barrier"
  elif(predicted.item()==2):
    pred_title = "Pedestrian Crossing"
  else:
    pred_title = "Duck Crossing"
  plt.title("Prediction: {}".format(pred_title))
  plt.show()


# Plot training errror
def plot(train_losses): 
  plt.plot(train_losses)
  plt.title('Training Error')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()

# Main function
def main(argv):
  training_set_path = "Dataset"

  my_train_dataset = DataLoader(datasets.ImageFolder(training_set_path,
                                                    transform=Compose([
                                                      Grayscale(),
                                                      Resize((128, 128)),
                                                      ToTensor(),
                                                      Normalize((0.5,), (0.5,))
                                                      ])),
                                batch_size=10,
                                shuffle=True) 

  torch.manual_seed(1) # seed randomness

  samples = enumerate(my_train_dataset)
  sample_idx, (sample_data, sample_targets) = next(samples)

  # Plot examples of training data
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(sample_data[i][0], cmap='gray', interpolation='none')
    if(sample_targets[i]==0):
      l = "Aframe"
    elif(sample_targets[i]==1):
      l = "Jersey Barrier"
    elif(sample_targets[i]==2):
      l = "Pedestrian Crossing"
    else:
      l = "Duck Crossing"
    plt.title("True Label: {}".format(l))
    plt.xticks([])
    plt.yticks([])
  plt.show()

  # DN model
  myModel = Net()
  myOptimizer = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.9)
  print(myModel) 

  epochs = 10
  train_losses, train_acc = train_model(myModel, my_train_dataset, myOptimizer, epochs) # Train model

  plot(train_losses) # Plot training loss

  # Save trained model
  torch.save(myModel.state_dict(), "myModel.pth")
  torch.save(myOptimizer.state_dict(), "myOptimizer.pth")
  print("Model saved successfully!")


  # Tranform the input image
  image_transform = Compose([
      ToPILImage(),
      Grayscale(),
      Resize((128, 128)),
      ToTensor(),
      Normalize((0.5,), (0.5,))
  ])

  # Read test image
  image = cv2.imread('Test set/test_duck4.jpg')
  image_tensor = image_transform(image)

  # Load pretrained model
  loaded_model = Net()
  loaded_optimizer = optim.SGD(myModel.parameters(), lr=0.01, momentum=0.9)
  loaded_model.load_state_dict(torch.load("myModel.pth"))
  loaded_optimizer.load_state_dict(torch.load("myOptimizer.pth"))

  # Testing
  testing(loaded_model, image_tensor)


if __name__ == "__main__":
    main(sys.argv)
  
