---
layout: page
title: Prod_jit_and_torchscript
permalink: /Prod_jit_and_torchscript
---

# Deep Learning in Produnction, JIT and Python dependencies

In computing, just-in-time (JIT) compilation (also dynamic translation or run-time compilations) is a way of executing computer code that involves compilation during execution of a program (at run time) rather than before execution. 

### TensorFlow1.0 and Graph execution
Let's dive into this, compare the Python and Tensorflow outputs in the code below:

```
#Python
a = 1
b = 2
c = a + b
print(c)
>>> 3

#Tensorflow 1.0
a = tf.Variable(1)
b = tf.Variable(2)
c = a + b
print(c)
>>> <tf.Variable ... shape()>
```
Wtf? Traditionally, Theano and consequently Tensorflow1.0 came with graph execution for JIT compilation as a default. This means that, when you define a variable in TF that variable is not evaluated but is rather a placeholder. This placeholder informs TF how variables are connected so that the program constructs a computation graph upon execution. **Specifically, `a = tf.Variable(1)` is just a placeholder.**

The TF code above will only evaluate the value of c when `a tf.session()` is wrapped around the graph definition and then executed. For example:

```
a = tf.Variable(1)
b = tf.Variable(2)
x = tf.placeholder()
yhat = a * x + b

with tf.session() as session:
  session.run(yhat, feed_dict={x:. 3})
  print(yhat)
>>> [0.5]
```
The above rewuirent is not pythonic and also generates the need for boilerplate code overhead. However, an additional mode for execution of operations as they are defined can be enabled i.e. eager mode, using `tf.compat.v1.enable_eager_execution()`. Nonetheless, jit compiling is significantly faster than eager mode which is crucial for models in production.


### Pytorch and eager model
Converesly, Pytorch comes with eager mode (as now in TF2.0) as default and constructs the computation graph dynamically. For example:

```
a = torch.tensor(1)
b = torch.tensor(2)
c = a + b
print(c)
>>> tensor(3)
```

This allows for 
- pythonic expression: models are object oriented python programs
- hacking: use any python library
- debugging and research: print, pdb debugger, REPL interpreter

However, the above are an issue for production requirements in terms of portability (model serialisation and export to variety of enviroments) and performance (optimise graph execution for inference latency, throughput etc). For example, in terms of portability pytorch models are tightly coupled to Python's REPL interpreter. In terms of performance, python's restrictive dynamism allows for computational optimisation only to a certain extend and therefore high inference latency and low service throughput.

### TorchScript

To overcome these issues Pytorch comes with Torchscript. Torchscript's purpose is to port serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency. 

Torchscript comes with a scipt and a tracing mode. For either modes the beneftis are 
1. The ability to serialize models and later run them outside of Python, via LibTorch, a C++ native module. In this way DL models  can be embedded in various production environments.
2. The ability to compile jit-able modules rather than running them as an interpreter, allowing for various optimizations and performancem improvements, both during training and inference. This is equally helpful for development and production.

See below an example of scripting a Pytorch model.

# Python on CPU

```python
X = torch.randn(128, 3, 28, 28, requires_grad=True)
y = torch.randint(0, 10, size=(128,))
```

```python
import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

class MyConv(torch.nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        self.linearnet = nn.Sequential(
            nn.Linear(64*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
        )
        
    def forward(self, x):
        batch = x.size(0)
        x = self.convnet(x)
        x = x.reshape(batch, -1)
        return x
```

```python
model_python = MyConv()
model_python
```




    MyConv(
      (convnet): Sequential(
        (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU()
        (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU()
      )
      (linearnet): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): ReLU()
        (2): Linear(in_features=1024, out_features=10, bias=True)
        (3): LogSoftmax(dim=1)
      )
    )



```python
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(params=model_python.parameters())

def train():
    y_pre = model_python(X)
    loss = loss_fn(y_pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("loss: ", loss.detach().numpy())
```

```python
%%time
for _ in range(20):
    train()
```

    loss:  -0.5149412
    loss:  -0.7885114
    loss:  -1.09506
    loss:  -1.4270493
    loss:  -1.7669119
    loss:  -2.0963206
    loss:  -2.4091487
    loss:  -2.703685
    loss:  -2.9786377
    loss:  -3.2331269
    loss:  -3.468272
    loss:  -3.6846206
    loss:  -3.882977
    loss:  -4.063745
    loss:  -4.22716
    loss:  -4.37317
    loss:  -4.503025
    loss:  -4.6171966
    loss:  -4.718364
    loss:  -4.8084326
    CPU times: user 8.28 s, sys: 266 ms, total: 8.55 s
    Wall time: 9.31 s

# Torch on CPU

```python
X = torch.randn(128, 3, 28, 28, requires_grad=True)
y = torch.randint(0, 10, size=(128,))
```

```python
class MyConv(torch.jit.ScriptModule):
    def __init__(self):
        super(MyConv, self).__init__()
        
        self.convnet = torch.jit.trace(nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ), torch.randn(1, 3, 28, 28))
        
        self.linearnet = torch.jit.trace(nn.Sequential(
            nn.Linear(64*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
        ), torch.randn(1, 64*28*28))
    
    @torch.jit.script_method
    def forward(self, x):
        batch = x.size(0)
        x = self.convnet(x)
        x = x.reshape(batch, -1)
        return x
```

```python
model_torch = MyConv()
MyConv()
```




    MyConv(
      (convnet): Sequential(
        original_name=Sequential
        (0): Conv2d(original_name=Conv2d)
        (1): BatchNorm2d(original_name=BatchNorm2d)
        (2): ReLU(original_name=ReLU)
        (3): Conv2d(original_name=Conv2d)
        (4): BatchNorm2d(original_name=BatchNorm2d)
        (5): ReLU(original_name=ReLU)
        (6): Conv2d(original_name=Conv2d)
        (7): BatchNorm2d(original_name=BatchNorm2d)
        (8): ReLU(original_name=ReLU)
      )
      (linearnet): Sequential(
        original_name=Sequential
        (0): Linear(original_name=Linear)
        (1): ReLU(original_name=ReLU)
        (2): Linear(original_name=Linear)
        (3): LogSoftmax(original_name=LogSoftmax)
      )
    )



```python
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(params=model_torch.parameters())

def train_jit():
    y_pre = model_torch(X)
    loss = loss_fn(y_pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("loss: ", loss.detach().numpy())
```

```python
%%time
for _ in range(20):
    train_jit()
```

    loss:  -0.32451892
    loss:  -0.5912205
    loss:  -0.89638066
    loss:  -1.2159026
    loss:  -1.5312963
    loss:  -1.8265008
    loss:  -2.1042957
    loss:  -2.3606114
    loss:  -2.5947688
    loss:  -2.8110933
    loss:  -3.0142124
    loss:  -3.2080374
    loss:  -3.3949592
    loss:  -3.5764592
    loss:  -3.7519906
    loss:  -3.9212832
    loss:  -4.0830455
    loss:  -4.235887
    loss:  -4.3776855
    loss:  -4.50814
    CPU times: user 7.95 s, sys: 161 ms, total: 8.11 s
    Wall time: 8.03 s

# Python on GPU

```python
model_python.to("cuda")
optimizer = torch.optim.Adam(params=model_python.parameters())
```

```python
def train_gpu():
    y_pre = model_python(X.to("cuda"))
    loss = loss_fn(y_pre, y.to("cuda"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("loss: ", loss.to("cpu").detach().numpy())
```

```python
%%time
for _ in range(20):
    train_gpu()
```

    loss:  -4.3880954
    loss:  -4.534538
    loss:  -4.658385
    loss:  -4.766274
    loss:  -4.862524
    loss:  -4.9499626
    loss:  -5.0297465
    loss:  -5.103152
    loss:  -5.171183
    loss:  -5.234377
    loss:  -5.2934456
    loss:  -5.349206
    loss:  -5.401966
    loss:  -5.4523873
    loss:  -5.501034
    loss:  -5.5479217
    loss:  -5.593396
    loss:  -5.6380644
    loss:  -5.68185
    loss:  -5.72521
    CPU times: user 597 ms, sys: 16.7 ms, total: 614 ms
    Wall time: 752 ms

# torch on GPU

```python
class MyConv(torch.jit.ScriptModule):
    def __init__(self):
        super(MyConv, self).__init__()
        
        self.convnet = torch.jit.trace(nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ).to("cuda"), torch.randn(1, 3, 28, 28, device="cuda"))
        
        self.linearnet = torch.jit.trace(nn.Sequential(
            nn.Linear(64*28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1),
        ).to("cuda"), torch.randn(1, 64*28*28, device="cuda"))
    
    @torch.jit.script_method
    def forward(self, x):
        batch = x.size(0)
        x = self.convnet(x)
        x = x.reshape(batch, -1)
        return x
    
model_torch = MyConv()
optimizer = torch.optim.Adam(params=model_torch.parameters())
```

```python
def train_jit_gpu():
    y_pre = model_torch(X.to("cuda"))
    loss = loss_fn(y_pre, y.to("cuda"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("loss: ", loss.to("cpu").detach().numpy())
```

```python
%%time
for _ in range(20):
    train_jit_gpu()
```

    loss:  -0.38337812
    loss:  -0.69447917
    loss:  -1.0345545
    loss:  -1.3835523
    loss:  -1.7232059
    loss:  -2.0454595
    loss:  -2.3477736
    loss:  -2.6333952
    loss:  -2.9032087
    loss:  -3.1584203
    loss:  -3.4010634
    loss:  -3.6302164
    loss:  -3.8446865
    loss:  -4.0430136
    loss:  -4.22342
    loss:  -4.3851805
    loss:  -4.529235
    loss:  -4.656223
    loss:  -4.76757
    loss:  -4.866083
    CPU times: user 1.11 s, sys: 34.5 ms, total: 1.14 s
    Wall time: 1.39 s

```python

```

```python

y
```




    tensor([4, 6, 6, 1, 8, 9, 0, 0, 9, 7, 1, 8, 0, 4, 6, 4, 4, 1, 3, 2, 3, 4, 3, 1,
            9, 7, 9, 5, 4, 3, 9, 6, 0, 8, 5, 2, 9, 2, 2, 5, 7, 8, 6, 7, 8, 1, 7, 6,
            7, 8, 1, 1, 1, 7, 4, 9, 4, 6, 2, 1, 0, 8, 3, 2, 8, 5, 4, 0, 0, 9, 1, 4,
            5, 1, 1, 5, 6, 1, 8, 5, 0, 0, 5, 8, 4, 9, 7, 0, 5, 4, 7, 0, 6, 9, 5, 6,
            2, 8, 9, 5, 6, 1, 7, 6, 4, 0, 3, 4, 4, 5, 0, 3, 1, 8, 9, 4, 9, 6, 8, 0,
            3, 0, 0, 6, 0, 8, 6, 1])


