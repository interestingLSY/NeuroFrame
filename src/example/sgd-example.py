import numpy as np
import time
import tqdm
import sys, os
if os.environ.get("USE_LOCAL_DYNLIB", False):
    sys.path.append(os.environ.get("USE_LOCAL_DYNLIB"))
import neuroframe as nf

n = 1024
learning_rate = 1e-3
num_steps = 32
dtype = nf.float32
learning_rate_decay = 0.96

nf.Device.set_default_device(nf.Device.cuda(0))

# Network: output = sigmoid(input @ w1) @ w2
input = nf.Tensor.randu((1, n), dtype)
expected_output = nf.Tensor.randu((1, n), dtype)
w1 = nf.Tensor.randu((n, n), dtype)
w2 = nf.Tensor.randu((n, n), dtype)

losses = []

for i in range(num_steps):
    nf.cgraph.clear_graph()
    
    # Forward pass
    inter = input @ w1
    pred_output = nf.ops.sigmoid(inter) @ w2
    delta = pred_output - expected_output
    loss = nf.ops.tensor_reduction_sum(delta*delta)
    loss_val = loss.numpy()
    print(f"loss: {loss_val}")
    losses.append(loss_val)
    
    # Backward pass
    nf.cgraph.perform_backward(loss, nf.Tensor.fill(nf.Scalar(1.0), loss.shape, loss.dtype))
    
    # Gradient descent
    w1 = w1 - nf.ops.tensor_muls(nf.cgraph.get_computed_grad(w1), learning_rate)
    w2 = w2 - nf.ops.tensor_muls(nf.cgraph.get_computed_grad(w2), learning_rate)
    
    # Learning rate decay
    learning_rate *= learning_rate_decay

import matplotlib.pyplot as plt
plt.plot(losses)
# log scale
plt.yscale("log")
plt.ylabel("loss")
plt.xlabel("iteration")
plt.title("loss vs iteration")
plt.show()
