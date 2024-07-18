import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import json
from numpyencoder import NumpyEncoder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Global Configs
DELTA = 1
MODEL_PATH = "./models/mnist_fc_64x4_adv_1.model"
EXPT_NAME = "torch_test"
LAYERS = 4
NEURONS_WIDTH = 64
INPUT_SHAPE = (1, 1, 28, 28)

# Define models
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def mnistfc():
    return nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# Load the model
model = mnistfc()
model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=50000, shuffle=True)

imgs, labels = next(iter(trainloader))

# Hook to capture activations
activations = []
def get_activation(name):
    def hook(model, input, output):
        activations.append(output.detach())
    return hook

# Register hooks
model.fc1.register_forward_hook(get_activation('fc1'))
model.fc2.register_forward_hook(get_activation('fc2'))
model.fc3.register_forward_hook(get_activation('fc3'))
model.fc4.register_forward_hook(get_activation('fc4'))

# Going through each label
P = {
    'config':
    {
        'net': MODEL_PATH,
        'delta': DELTA,
        'data_len': labels.shape[0],
        'neurons_width': NEURONS_WIDTH,
        'layers': LAYERS,
        'input_shape': INPUT_SHAPE
    }
}

for label in range(10):
    # Filtering relevant data alone
    mask = (labels == label)
    S = imgs[mask]

    # Initializing a counter
    count = np.zeros((LAYERS, NEURONS_WIDTH))

    # Counting across relevant data
    print(f"Processing label {label}...")

    for example in tqdm(S):
        activations.clear()
        with torch.no_grad():
            model(example.unsqueeze(0))
        
        for layer_idx, activation in enumerate(activations):
            whether_activated = (activation > 0).numpy().flatten()
            count[layer_idx] += whether_activated

    # Adding neuron indices in A or D based on whether their fr (frequency ratio) is greater than or lesser than delta
    fr = count / S.shape[0]
    greater_than_delta = np.where(fr >= DELTA)
    lesser_than_delta = np.where(fr < (1 - DELTA))
    A = list(zip(greater_than_delta[0], greater_than_delta[1]))
    D = list(zip(lesser_than_delta[0], lesser_than_delta[1]))

    P[label] = {
        "A": {
            "len": len(A),
            "indices": A
        },
        "D": {
            "len": len(D),
            "indices": D
        }
    }

# Writing it out into a file
with open(f"./NAPs/{EXPT_NAME}.json", "w") as f:
    json.dump(P, f, cls=NumpyEncoder)
