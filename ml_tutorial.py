#!/usr/bin/env python
# coding: utf-8

# # Small Neural Network tutorial: 
# Needed packages torch, torchvision   
#                            SMA May 2023  
#                            First part it's from pytorch tutorial: https://pytorch.org/tutorials/  
#                            There's more information about NN in:  https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

# In[1]:


import torch
import numpy as np


# The advantage of pytorch is that it uses tensors, that's one of the main features that makes it powerful for ML

# In[2]:


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data) #We can create a tensor from numpy and also a numpy array from pytorch
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


# In[3]:


print(x_data[0])


# In[4]:


x_ones = torch.ones_like(x_data) # retains the properties of x_data (the size)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# In[5]:


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


# In[6]:


tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)


# In[7]:


t1 = torch.cat([tensor, tensor, tensor], dim=1) #Joins tensors
print(t1)


# In[8]:


print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n") #It can calculate the transpose matrix and do normal multiplication
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")


# In[9]:


print(tensor, "\n")
tensor.add_(5)
print(tensor)


# In[10]:


import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)


# In[11]:


prediction = model(data) # forward pass


# In[12]:


loss = (prediction - labels).sum()
loss.backward() # backward pass


# In[13]:


optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)


# In[14]:


optim.step() #gradient descent


# ### Neural Network

# A typical training procedure for a neural network is as follows:
# 
# - Define the neural network that has some learnable parameters (or weights)
# 
# - Iterate over a dataset of inputs
# 
# - Process input through the network
# 
# - Compute the loss (how far is the output from being correct)
# 
# - Propagate gradients back into the network’s parameters
# 
# - Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
# 
# From: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

# In[15]:


import torch
import torch.nn as nn #This is the package from torch to create neural networks
import torch.nn.functional as F


# This one is for image recognition

# In[16]:


class Net(nn.Module):

    def __init__(self): #This is for defining the inputs
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): #You have to define how to go forward 
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #It is using relu
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)


# In[17]:


params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


# In[18]:


input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# In[19]:


net.zero_grad()
out.backward(torch.randn(1, 10))


# In[20]:


output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)


# In[21]:


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# In[22]:


import torch.optim as optim


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


# ### Building a NN with python

# In[23]:


import numpy as mp
import math
import matplotlib.pyplot as plt

x=np.linspace(-math.pi, math.pi, 2000)
y=np.sin(x)

a=np.random.randn()
b=np.random.randn()
c=np.random.randn()
d=np.random.randn()

learning_rate=1e-6
for t in range(20000):
    y_pred=a+b*x+c*x**2+d*x**3
    loss=np.square(y_pred-y).sum()
    if t%100==99:
       print(t,loss)
    # Now I need to train the model by calculating gradients
    grad_y_pred=2.0*(y_pred-y)
    grad_a=grad_y_pred.sum()
    grad_b=(grad_y_pred*x).sum()
    grad_c=(grad_y_pred*x**2).sum()
    grad_d=(grad_y_pred*x**3).sum()
    
    #Update my weights
    a -= learning_rate*grad_a
    b -= learning_rate*grad_b
    c -= learning_rate*grad_c
    d -= learning_rate*grad_d
    
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
res=a=+b*x+c*x**2+d*x**3
fig=plt.figure
print(res)
plt.plot(x,res)
plt.plot(x,y, 'or')
plt.show()


# Using pytorch: tensors and autograd

# In[24]:


#Now with pytorch
import torch
import math

dtype=torch.float
device=torch.device("cpu")

x=torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

a=torch.randn((),device=device, dtype=dtype, requires_grad=True)
b=torch.randn((),device=device, dtype=dtype, requires_grad=True)
c=torch.randn((),device=device, dtype=dtype, requires_grad=True)
d=torch.randn((),device=device, dtype=dtype, requires_grad=True)

learning_rate=1e-6
for t in range(2000):
    y_pred=a+b*x+c*x**2+d*x**3
    loss=(y_pred-y).pow(2).sum()
    if t % 100==99:
        print(t,loss.item)
    #Here I Use pytorch for the gradients instead of doing them by hand
    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()
    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
print(f'Resultpr: y = {a} + {b} x + {c} x^2 + {d} x^3')
print(a)
x_n=np.array(x)

res=a.detach().numpy()+b.detach().numpy()*x_n+c.detach().numpy()*x_n**2+d.detach().numpy()*x_n**3

print(res)
fig=plt.figure
print(res)
plt.plot(x,res)
plt.plot(x,y, 'or')
plt.show()
    
    


# Using pytorch to define new functions in this case $y=a+bP_3(c+dx)$ where $P_3(x)=\frac{1}{2}(5x^3-3x)$, and computing with an optimizer

# In[25]:


import torch
import math

#You can define classes: the main thing is that they need a forward and a backward functiont.
#Forward:Computes output Tensor from input
#Backward: Receives the gradient of the output
class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod 
    def forward(ctx,input):
        #It recieves a Tensor with the input end return the output. 
        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)
    @staticmethod
    def backward(ctx, grad_output):
        #Here we recieve a tensor containing gradients of the loss with respect output, and it needs to compute the gradient of the loss with the input
        input,= ctx.saved_tensors
        return grad_output*1.5*(5*input**2-1)
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")  # Uncomment this to run on GPU


x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

#Create random tensor for weights
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-7
for t in range(200000):
    # To apply our Function, we use Function.apply method.
    P3 = LegendrePolynomial3.apply
    y_pred = a + b * P3(c + d * x)
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
     # Use autograd to compute the backward pass.
    loss.backward()
       # Update weights using gradient descent
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
x_n=np.array(x)
print(x_n)
#print(f'Resultpr: y = {a} + {b} x + {c} x^2 + {d} x^3')
#print(a)
x_n=np.array(x)
print(type(a.item()))
print(type(x_n))
res=a.item()+b.item()*0.5*(5*(c.item()+d.item()*x_n)**3-3*((c.item()+d.item()*x_n)))

#res=a.detach().numpy()+b.detach().numpy()*0.5*(5*c.detach().numpy()**3-3*c.detach().numpy()+d.detach().numpy()*x_n

print(res)
#fig=plt.figure
#print(res)
plt.plot(x,res)
plt.plot(x,y, 'or')
plt.show()
    
    
    
    


# ### Using neural networks from pytorch

# In PyTorch, the nn package serves this same purpose. The nn package defines a set of Modules, which are roughly equivalent to neural network layers. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. The nn package also defines a set of useful loss functions that are commonly used when training neural networks.

# In[26]:


# -*- coding: utf-8 -*-
import torch
import math


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Prepare the input tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# In the above code, x.unsqueeze(-1) has shape (2000, 1), and p has shape
# (3,), for this case, broadcasting semantics will apply to obtain a tensor
# of shape (2000, 3) 

# Use the nn package to define our model and loss function.
#The Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
# The Flatten layer flatens the output of the linear layer to a 1D tensor,
# to match the shape of `y`.
model = torch.nn.Sequential( #Helps to define layers
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')

#print(res)
fig=plt.figure
#print(res)
plt.plot(x,res)
plt.plot(x,y, 'or')
plt.show()
    
    
    


# ## Random Information

# In[27]:


#import Libraries/
import torch
import torch.nn as nn
import numpy as np
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')
print(targets[0])
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(inputs[0])


# In[28]:


#Tensor Data set 
from torch.utils.data import TensorDataset
# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
#Data loader
from torch.utils.data import DataLoader
# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# In[29]:


print(train_ds[0])


# In[30]:


# Define linear model
model = nn.Linear(3, 2) #torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
print(model.weight)
print(model.bias)
# Parameters
list(model.parameters())
# Define Loss
import torch.nn.functional as F
loss_fn = F.mse_loss
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# In[31]:


device=torch.device("cpu")

class LinearRegression(torch.nn.Module): #The thing in parenthesis is the parent from where you are inherencing 
    def __init__(self, inputSize, outputSize): #Initialize and passes variables
        super(LinearRegression, self).__init__() #Inherent the attribute from linearRegression
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


# In[32]:


# Define model
model = LinearRegression(3, 2) # nn.Linear(in_features,out_features)
#model=nn.Linear(3,2)
print(list(model.parameters()))
#Define the loss function
loss_fun = nn.MSELoss()
# Define SGD optimizer with learning rate 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# In[33]:


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
           #print(xb[0])
            #print(model)
            #print(xb)
            #print(yb)
            #print(xb.size(1))
            #print(xb.type())
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#fit model for 1000 epochs
fit(10000, model, loss_fn , opt ,train_dl)
# Generate predictions
preds = model(inputs)
preds


# # Autoencoder

# In[34]:


# import general libraries
import sys
import os
import math
import numpy as np

# import torch modules for autoencoder
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

# import libraries for plots
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# import GMM
from sklearn import mixture


# In[35]:


# load training data
particle_type = "large"
file_name = "bop_" + particle_type + ".dat"
file_in = open(file_name, 'r')
dataset = []
lines = file_in.readlines()
Ndata = len(lines)
for i in range(Ndata):
    temp = [float(n) for n in lines[i].split()]
    dataset.append([])
    Ninputs = len(temp)
    for j in range(Ninputs):
        dataset[i].append(temp[j])
file_in.close()
dataset = np.array(dataset)
d = len(dataset[0])
print("Vector dimension:           " + str(d))
print("Number of training vectors: " + str(Ndata))


# # Define Autoencoder and training parameters

# In[36]:


#print torch.cuda.device_count()
#print torch.cuda.is_available()
#torch.cuda.set_device(0)
device = torch.device('cpu')

# Xavier initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    def __init__(self, Ninput, Nhidden, Noutput):
        super(Encoder, self).__init__()
        self.hidden = nn.Linear(Ninput, Nhidden)
        self.output = nn.Linear(Nhidden, Noutput)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

class Decoder(nn.Module):
    def __init__(self, Ninput, Nhidden, Noutput):
        super(Decoder, self).__init__()
        self.hidden = nn.Linear(Ninput, Nhidden)
        self.output = nn.Linear(Nhidden, Noutput)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def mse_loss(data_out_pred, data_out):
    diff = data_out_pred - data_out
    diff = diff*diff
    return torch.mean(diff)


def get_var_from_np(np_array, cuda=False, requires_grad=False):    # convert numpy to PyTorch variable for training
    temp = Variable(torch.from_numpy(np_array), requires_grad=requires_grad).type(torch.FloatTensor)
    if cuda: temp = temp.cuda()
    return temp


def train(model, data):
    
    # compute dataset variance
    std = np.std(data, axis=0)
    msd = 0.0
    for i in range(len(std)):
        msd += std[i]*std[i]
    msd /= len(std)

    train_data = My_dataset(get_var_from_np(data), get_var_from_np(data))
    
    n_epoch = 30
    batch_size = 100
    learning_rate = 0.5
    n_batch = math.ceil(1. * len(train_data) / batch_size)
    loss_train = np.zeros(n_epoch)
    fve_train = np.zeros(n_epoch)

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9, weight_decay=1e-5)

    for epoch in range(n_epoch):

        dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        batch_idx = 0
        for train_in, train_out in dataset:
            optimizer.zero_grad()
            out = model(train_in)
            loss = nn.MSELoss()(out, train_out)
            loss.backward()
            optimizer.step()
            batch_idx += 1

        data_out_pred = model(get_var_from_np(data))
        mse = mse_loss(data_out_pred, get_var_from_np(data))
        fuv = mse/msd
        fve = 1. - fuv
        loss_train[epoch] = mse
        fve_train[epoch] = fve
        print("Epoch = %5d,\tmse = %5.5e \tfuv = %5.5f\tfve = %5.5f" % (epoch+1, mse, fuv, fve))

    return loss_train, fve_train

class My_dataset(Dataset):      # construct dataset object for mini-batch gradient descent
    def __init__(self, data_in, data_out):
        self._data_in = data_in
        self._data_out = data_out

    def __len__(self):
        return len(self._data_in)

    def __getitem__(self, index):
        return self._data_in[index], self._data_out[index]


c = 2    
encoder = Encoder(d, 5*d, c)
print(encoder)
decoder = Decoder(c, 5*d, d)
print(decoder)
ae = AE(encoder, decoder)
ae.apply(init_weights)
print(ae)


# # Train Autoencoder

# In[37]:


# perform training
loss_train, fve_train = train(ae, dataset)
print("Training complete\n")
# save network
if not os.path.exists('./Net'):
    os.mkdir('./Net')
# saving trained model
save_path = "./Net/ae.pyt"
torch.save(ae.state_dict(), save_path)
save_path = "./Net/encoder.pyt"
torch.save(encoder.state_dict(), save_path)
save_path = "./Net/decoder.pyt"
torch.save(decoder.state_dict(), save_path)
# plot error and fve
fig, axs = plt.subplots(1,2, figsize=(15,5))
axs[0].plot(np.arange(len(loss_train)) + 1, loss_train, 'r-o')
axs[1].plot(np.arange(len(fve_train)) + 1, fve_train, 'r-o')
axs[1].set_ylim(0,1)
axs[0].set_xlabel('epoch')
axs[1].set_xlabel('epoch')
axs[0].set_ylabel('MSE')
axs[1].set_ylabel('FVE')
plt.show()


# # Reduce dimensionality

# In[39]:


# load encoder
#encoder = Encoder(d, 5*d, c)
#encoder.load_state_dict(torch.load('./Net/encoder.pyt'))

# project data onto low-dimensional space
proj = encoder(get_var_from_np(dataset))
proj = proj.detach().numpy()

# plot projection
x = proj[:,0]
y = proj[:,1]
fig = plt.figure()
ax = fig.add_subplot(111)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.5)
plt.scatter(x, y, c= 'grey', edgecolors= 'black')
# plt.xticks([])
# plt.yticks([])
plt.show()


# In[ ]:




