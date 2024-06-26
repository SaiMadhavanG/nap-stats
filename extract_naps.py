import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import json
from numpyencoder import NumpyEncoder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Global Configs
DELTA = 1
MODEL_PATH = "./models/mnist_fc_64x4_adv_1.model"
EXPT_NAME = "torch_test"
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
model = mnistfc().to(device)
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
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu()
    return hook
# Getting layers of interest
layers = {}

for module in model.named_modules():
    if isinstance(module[1], nn.modules.activation.ReLU):
        if isinstance(prev[1], nn.modules.linear.Linear):
            layers[prev[0]] = prev[1].out_features
    prev = module
# Registering hooks for required layers
for layer in layers:
    model.__getattr__(layer).register_forward_hook(get_activation(layer))
# Going through each label
P = {
    'config':
    {
        'net': MODEL_PATH,
        'delta': DELTA,
        'data_len': labels.shape[0],
        'layers': layers,
        'input_shape': INPUT_SHAPE
    }
}
class ActivationCounter:
    def __init__(self, layers: dict) -> None:
        self.layers = layers
        self.init_counter()
        
    def init_counter(self):
        self.counter = {}

        for layer in self.layers:
            self.counter[layer] = np.zeros(self.layers[layer])

    def add(self, layer, whether_activated):
        self.counter[layer] += whether_activated

    def getAD(self, delta, total_num):
        A = []
        D = []
        for layer_idx, layer in enumerate(layers):
            fr = self.counter[layer]/total_num
            greater_than_delta = np.where(fr >= DELTA)
            lesser_than_delta = np.where(fr < (1 - DELTA))
            for neuron_idx in greater_than_delta[0]:
                A.append((layer_idx, neuron_idx))
            for neuron_idx in lesser_than_delta[0]:
                D.append((layer_idx, neuron_idx))
    
        return A, D
for label in range(10):
    # Filtering relevant data alone
    mask = (labels == label)
    S = imgs[mask]

    # Initializing a counter
    counter = ActivationCounter(layers)

    # Counting across relevant data
    print(f"Processing label {label}...")

    for example in tqdm(S):
        activations.clear()
        with torch.no_grad():
            model(example.unsqueeze(0).to(device))
        
        for layer, activation in activations.items():
            whether_activated = (activation > 0).numpy().flatten()
            counter.add(layer, whether_activated)

    A, D = counter.getAD(DELTA, S.shape[0])

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