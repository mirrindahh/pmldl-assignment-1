from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import uvicorn
from fastapi import FastAPI, File

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16 * 54 * 54)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        
        return F.log_softmax(X, dim=1)

# Load model.
model = ConvolutionalNetwork()
model.load_state_dict(torch.load("./cats-and-dogs.pt", weights_only=True))
model.eval()

# We need to use same as for train.
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

app = FastAPI(
    title="Cat Or Dog? API", 
    version="0.0.1",
    description="This api provides an endpoint to determine whether the image contains a cat or a dog."
)

@app.post("/predict")
def info(file: bytes = File()):
    with BytesIO(file) as buffer:
        image = Image.open(buffer)
        image.load()

    # We've trained on RGB.
    image = image.convert("RGB")
    preprocessed = transform(image)
    pred = model(preprocessed)
    # Index of label.
    index = int(torch.max(pred.data,1)[1][0])
    # Same labels as traing.
    return ("cat", "dog")[index]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
