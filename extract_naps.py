import torch
import onnx
import onnxruntime as ort
import numpy as np
from tqdm.auto import tqdm
import json
from numpyencoder import NumpyEncoder

# Global Configs

DELTA = 0.0
NET_PATH = "./256x4_scratch.onnx"
DATA = torch.load("./dataset/training.pt")
LAYERS = 4
EXPT_NAME = "256x4s_delta0"

# Data preprocessing

imgs = DATA[0]/255
imgs = imgs.reshape(60000, 1, 784).numpy()

# Modifying model to get neuron activations

model = onnx.load(NET_PATH)

def add_intermediate_outputs(model):
    graph = model.graph
    for node in graph.node:
        if node.op_type == "Relu":
            for output in node.output:
                intermediate_value_info = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)
                graph.output.append(intermediate_value_info)
    return model

modified_model = add_intermediate_outputs(model)

modified_model_path = f'{NET_PATH.split(".")[1][1:]}_modified.onnx'
onnx.save(modified_model, modified_model_path)

# Setting up inference in onnx

session = ort.InferenceSession(modified_model_path)
input_name = session.get_inputs()[0].name

# Going through each label
P = {
    'config':
    {
        'net': NET_PATH,
        'delta': DELTA,
        'data_len': DATA[1].shape[0]
    }
}

for label in range(10):
    # Filetering relevant data alone
    mask = (DATA[1] == label).numpy()
    S = imgs[mask]

    # Initializing a counter
    count = np.zeros((LAYERS, 256))

    # Counting across relevant data
    print(f"Processing label {label}...")

    for example in tqdm(S):
        outputs = session.run(None, {input_name: example})
        neuron_activations = np.concatenate(outputs[1:])
        whether_activated = neuron_activations > 0
        count += whether_activated

    # Adding neuron indices in A or D based on whether their fr (frequency ratio) is greater than or lesser than delta
    fr = count / S.shape[0]
    greater_than_delta = np.where(fr>=DELTA)
    lesser_than_delta = np.where(fr<=(1-DELTA))
    A = list(zip(greater_than_delta[0], greater_than_delta[1]))
    D = list(zip(lesser_than_delta[0], lesser_than_delta[1]))
    
    P[label] = {
        "A":
        {
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