'''
 Program to recognize real-world objects on live video stream using deep neural networks
'''

# Import statements
import sys
import cv2
import torch
from torch import nn
from torchvision.transforms import Compose, ToPILImage, Grayscale, Resize, ToTensor, Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# Class definitions
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 10 filters of size 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 20 filters of size 5x5
        self.fc1 = nn.Linear(16820, 500)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Max pooling, RelU activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 16820) # Flatten
        x = F.relu(self.fc1(x)) # Fully connected layers
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) # Softmax activation func


''' Main Function '''
def main(argv):
    loaded_model = Net() 

    loaded_model.load_state_dict(torch.load('myModel.pth')) # Load pre-trained model
    print("Model loaded successfully!")
    loaded_model.eval()

    # Image tranformation
    image_transform = Compose([
        ToPILImage(),
        Grayscale(),
        Resize((128, 128)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    # Turn camera capture on
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read() # Read the frames
        if not ret:
            break
        
        # Tranfrom the frames
        image_tensor = image_transform(frame)

        # Predict the frames
        with torch.no_grad():
            output = loaded_model(image_tensor)
            _, predicted = torch.max(output, 1)
            if(predicted.item()==0):
                label = "Aframe"
            elif(predicted.item()==1):
                label = "Jersey Barrier"
            elif(predicted.item()==2):
                label = "Pedestrian Crossing"
            elif(predicted.item()==4):
                label = "Duck Crossing"
            else:
                label = "None"

        # Overlay the prediction on the screen
        cv2.putText(frame, str(label), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Live Recognition', frame)

        # Wait for user input
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    
    # Close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
