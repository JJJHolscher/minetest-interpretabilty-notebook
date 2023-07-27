# %% [markdown]
"""
# Jono's Minetester PPO Interpretabilty Notebook
"""

# %% [markdown]
"""
## Policy and Image Paths
"""

# %%
# Define Paths
SAVED_MODEL_PATH = "ppo_treechop-v0.model"

IMAGE_FOLDER = "screenshots/"


# %% [markdown]
"""
## Load model, define utility functions, etc...
"""

# %%
#Import dependencies

import sys
import os
import numpy as np
import jax
import flax
import flax.linen as nn
import jax.numpy as jnp
import gym
import cv2

from typing import Sequence, Callable
from flax.linen.initializers import constant, orthogonal
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# %matplotlib inline

# %%
#Define Neural Networks

class Network(nn.Module):
    def setup(self):
        self.Conv_0 = nn.Conv(
                32,
                kernel_size=(8, 8),
                strides=(4, 4),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        self.Conv_1 = nn.Conv(
                64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        self.Conv_2 = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )
        self.Dense_0 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        # Can this be more elegant?
        self.layers = [self.l0, self.l1, self.l2, self.l3]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def l0(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        return self.Conv_0(x)

    def l1(self, x):
        x = nn.relu(x)
        return self.Conv_1(x)

    def l2(self, x):
        x = nn.relu(x)
        return self.Conv_2(x)

    def l3(self, x):
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return self.Dense_0(x)

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

# %%
#Load model in 

SEED=42
ACTION_DIMENSION = 36
OBS_SHAPE = (1, 4, 64, 64) # (batch, timesteps, x, y)

network = Network()
actor = Actor(action_dim=ACTION_DIMENSION)
critic = Critic()

key = jax.random.PRNGKey(SEED)

key, network_key, actor_key, critic_key = jax.random.split(key, 4)
sample_obs = np.zeros(OBS_SHAPE,dtype=np.float32)
print(network.tabulate(jax.random.PRNGKey(0), sample_obs, console_kwargs={"width": 200}))
network_params = network.init(network_key, sample_obs)
actor_params = actor.init(actor_key, network.apply(network_params, sample_obs)[0])
critic_params = critic.init(critic_key, network.apply(network_params, sample_obs)[0])

with open(SAVED_MODEL_PATH, "rb") as f:
    (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
        (None, (network_params, actor_params, critic_params)), f.read()
    )

# %%
#action/id mapping
#Punching is enabled by default
# mouse values are (x,y) pairs
action_mapping = {
    0:"mouse -25 -25",
    1:"mouse -25 0",
    2:"mouse -25 25",
    3:"mouse 0 -25",
    4:"mouse 0 0",#null op
    5:"mouse 0 25",
    6:"mouse 25 -25",
    7:"mouse 25 0",
    8:"mouse 25 25",
    9:"mouse -25 -25",
    10:"mouse -25 0, JUMP",
    11:"mouse -25 25, JUMP",
    12:"mouse 0 -25, JUMP",
    13:"mouse 0 0, JUMP",
    14:"mouse 0 25, JUMP",
    15:"mouse 25 -25, JUMP",
    16:"mouse 25 0, JUMP",
    17:"mouse 25 25, JUMP",
    18:"mouse -25 -25, JUMP",
    19:"mouse -25 0, FORWARD",
    20:"mouse -25 25, FORWARD",
    21:"mouse 0 -25, FORwARD",
    22:"mouse 0 0, FORWARD",
    23:"mouse 0 25, FORWARD",
    24:"mouse 25 -25, FORWARD",
    25:"mouse 25 0, FORWARD",
    26:"mouse 25 25, FORWARD",
    27:"mouse -25 -25, FORWARD, JUMP",
    28:"mouse -25 0, FORWARD, JUMP",
    29:"mouse -25 25, FORWARD, JUMP",
    30:"mouse 0 -25, FORWARD, JUMP",
    31:"mouse 0 0, FORWARD, JUMP",
    32:"mouse 0 25, FORWARD, JUMP",
    33:"mouse 25 -25 FORWARD, JUMP",
    34:"mouse 25 0, FORWARD, JUMP",
    35:"mouse 25 25, FOWARD, JUMP",
    
}
forward_mask = jnp.array([0]*18+[1]*18)
jump_mask = jnp.array(([0]*9+[1]*9)*2)
up_mask = jnp.array([1,0,0]*12)
down_mask = jnp.array([0,0,1]*12)
left_mask = jnp.array(([1]*3+[0]*6)*4)
right_mask = jnp.array(([0]*6+[1]*3)*4)

# %%
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # add more image types if necessary
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                img_arr = np.array(img)
                images.append(img_arr)
    return images

# %%
def plot_frames(frames):

    # Create a figure with 4 subplots, one for each frame
    fig, axs = plt.subplots(1, frames.shape[0], figsize=(12, 3))

    # Loop through each frame and plot it on a separate subplot
    for i in range(frames.shape[0]):
        axs[i].imshow(frames[i], cmap='gray')
        axs[i].axis('off')

# %%
def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(
            image, (64,64), interpolation=cv2.INTER_AREA
        )
    return image
    

# %%
# Load screen shots from folder
images = load_images_from_folder(IMAGE_FOLDER)

# %%
def network_gradient(output_fn, target=0):
    def partial_forward(x):
        for layer in network.layers[target:]:
            x = layer(x)
        return x

    def f(x):
        hidden = network.apply(network_params, x, method=partial_forward)
        action_logits = actor.apply(actor_params, hidden)
        critic_values = critic.apply(critic_params, hidden)
        params = (p for p in network_params["params"]) # Dictionaries retain ordering since py3.7.
        output = output_fn(action_logits, critic_values, hidden, params)
        return output
    return jax.value_and_grad(f)

def deep_dream(init_input, output_fn, target="input", lr=1e3, n_iter=600, clip_low=0, clip_high=255):
    
    hidden,layers = network.apply(network_params,init_input)
    if target == "input":
        x = init_input
    if target == "l1":
        x = layers[0]
    if target == "l2":
        x = layers[1]
    if target == "l3":
        x = layers[2]
    if target == "network":
        x = hidden
    
    f = jax.jit(network_gradient(output_fn, target=target))
    
    for i in range(n_iter):
        value, grad = f(x)
        x += lr*grad
        x = jnp.clip(x, clip_low, clip_high)
        if i % 200 == 0:
            print("Iteration:", i, "Value", value)
    
    return x

# %%
#Define the yaw probablity as p(turn left)-p(turn right)

def yaw_probabilty(action_logits, critic_output, network_output, layers, orientation):
    if orientation == "left":
        x = 1
    if orientation == "right":
        x = -1
    action_ps = jax.vmap(jax.nn.softmax)(action_logits)
    yaw_values = jax.vmap(lambda x: jnp.dot(left_mask,x)-jnp.dot(right_mask,x))(action_ps)
    return x*yaw_values[0]

# %%
def plt_raw_image(image):
    plt.imshow(image)
    plt.gca().axis('off')
    plt.show()
    
def plt_network_image(image, cmap='viridis'):
    plt.imshow(transform_image(image), cmap=cmap)
    plt.gca().axis('off')
    plt.show()
    
def plt_latent(image, channel, layer="network",cmap='viridis', is_active=False):
    a = jnp.stack([transform_image(image)]*4)[np.newaxis,...]
    hidden, layers = network.apply(network_params,a)
    if layer == "l1":
        layer_data = layers[0][0,:,:,channel]
    if layer == "l2":
        layer_data = layers[1][0,:,:,channel]
    if layer == "l3":
        layer_data = layers[2][0,:,:,channel]
    if layer == "network":
        layer_data = hidden.reshape(16,32)
    
    if is_active:
        plt.imshow(layer_data > 0, cmap=cmap)
    else:
        plt.imshow(layer_data, cmap=cmap)
    plt.gca().axis('off')
    plt.show()
    print("min_val:", jnp.min(layer_data), "max_val:", jnp.max(layer_data))
    print("------------------------------")
    
def plt_gradient(image, channel, output_fn, layer="network",cmap='viridis'):
    network_input = jnp.stack([transform_image(image)]*4)[np.newaxis,...]
    hidden, layers = network.apply(network_params,network_input)
    if layer == "input":
        x = network_input
    if layer == "l1":
        x = layers[0]
    if layer == "l2":
        x = layers[1]
    if layer == "l3":
        x = layers[2]
    if layer == "network":
        x = hidden
    
    _,gradient = jax.jit(network_gradient(output_fn, target=layer))(x)
    
    if layer == "network":
        gradient = gradient.reshape(1,16,32,1)
    
        
    plt.imshow(gradient[0,:,:,channel], cmap=cmap)
    plt.gca().axis('off')
    plt.show()
    print("min_val:", jnp.min(gradient[0,:,:,channel]), "max_val:", jnp.max(gradient[0,:,:,channel]))
    print("------------------------------")
    
def print_actions(image):
    a = jnp.stack([transform_image(image)]*4)[np.newaxis,...]
    network_state, layers = network.apply(network_params,a)
    action_logits = actor.apply(actor_params,network_state)
    action_ps = jax.nn.softmax(action_logits)[0]
    
    yaw_value = jnp.dot(left_mask,action_ps)-jnp.dot(right_mask,action_ps)
    pitch_value = jnp.dot(up_mask,action_ps)-jnp.dot(down_mask,action_ps)
    jump_value = jnp.dot(jump_mask,action_ps)
    forward_value = jnp.dot(forward_mask,action_ps)
    
    print("Yaw:", yaw_value)
    print("Pitch:", pitch_value)
    print("Forward:", forward_value)
    print("Jump:", jump_value)

# %% [markdown]
"""
## Interpreting the policy
"""

# %% [markdown]
"""
### Deep Dreaming
We can use the `deep_dream` function to probe the model for high value states and inputs that trigger particular actions
"""

# %%
#Deep dreaming to find high value states

frames = jnp.array(np.random.rand(1,4,64,64), dtype=jnp.float32)*0.1+128

def get_critic(a,critic,c,d):
    return critic[0][0]
print("Mean brightness before optimization:", jnp.mean(frames))
optimized_frames = deep_dream(frames,get_critic,n_iter=500)
plot_frames(optimized_frames[0])

# %%
#Deep dreaming to find high yaw states

#Effects are clearly visible when we initialize from a roughly uniform value
frames = jnp.array(np.random.rand(1,4,64,64))*0.1+128

def f(frames):
    last_frame = frames[-1] #The last frame is most salient
    last_frame_squared = np.array((last_frame-np.mean(last_frame))**2)# Look at squared deviation
    blurred_squared_last_frame = cv2.GaussianBlur(last_frame_squared,(5,5),0) #Apply smoothing
    return blurred_squared_last_frame[np.newaxis,:,:]

optimized_frames_left = deep_dream(frames,lambda a,b,c,d: yaw_probabilty(a,b,c,d,"left"))
optimized_frames_right = deep_dream(frames,lambda a,b,c,d: yaw_probabilty(a,b,c,d,"right"))

amplitude_left = f(optimized_frames_left[0])
amplitude_right = f(optimized_frames_right[0])
output = np.concatenate([amplitude_left, amplitude_right])

plot_frames(output)


# %% [markdown]
"""
### Alignment between actor and critic

We notice that the actor an critic pay attention to the same set of features more than one would expect if their values were random.
"""

# %%
actor_matrix = actor_params["params"]["Dense_0"]["kernel"]
critic_vector = critic_params["params"]["Dense_0"]["kernel"]
similarity_scores = cosine_similarity(critic_vector.reshape(1, -1), actor_matrix.T)
plt.hist(similarity_scores.reshape(-1))
similarity_scores = cosine_similarity(np.random.randn(512).reshape(1, -1), actor_matrix.T)
plt.hist(similarity_scores.reshape(-1))
plt.legend(['Actor/Critic cosine similarity', 'Random cosine similarity'])

# %% [markdown]
"""
### What the model sees vs what humans see

The input image is downscaled from 600x1024x3 RGB -> 64x64 greyscale
"""

# %%
image_id = 12
image  = images[image_id]
print("What humans see:")
plt_raw_image(image)
print("What the network sees:")
plt_network_image(image, cmap='gray')

# %% [markdown]
"""
### The network's control system

We can see for many images with trees in them, that by mirroring the image, the sign of the yaw probablity is inverted.
"""

# %%
image_id = 1
image  = images[image_id]
plt_network_image(image)
print_actions(image)

image  = images[image_id][:,::-1,:]
plt_network_image(image)
print_actions(image)

# %% [markdown]
"""
### Generalization to the night

We see that the control system seems to keep working with different tree types and even at night, where colors are inverted.
"""

# %%
image_id = 30
image  = images[image_id]
plt_network_image(image)
print_actions(image)

image  = images[image_id][:,::-1,:]
plt_network_image(image)
print_actions(image)

# %% [markdown]
"""
### Drilling into Layer 1

We can plot the activation for typical images to see how the internal activations react. Values above 0 will cause a relu network to pass the value forward. For instance, we can see channel 21 acting as a left/right edge detector
"""

# %%
image_id =1
image = images[image_id]
print("What the network sees:")
plt_network_image(image)

print("Pre ReLU activation:")
plt_latent(np.array(image, dtype=np.float32),21,layer="l1")
print("Which ReLU's get triggered:")
plt_latent(np.array(image, dtype=np.float32),21,layer="l1", is_active=True)

# %% [markdown]
"""
### Tree detectors in layers 2 and 3

We found that channel 56 in layer 2 and channel 0 in layer 3 acted somewhat reliably as a tree detectors, even at night.
"""

# %%
image_id = 15
image = images[image_id]
print("What the network sees:")
plt_network_image(image)

print("Pre ReLU activation:")
plt_latent(np.array(image, dtype=np.float32),56,layer="l2")
print("Which ReLU's get triggered:")
plt_latent(np.array(image, dtype=np.float32),56,layer="l2", is_active=True)

# %%
image_id = 15
image = images[image_id]
print("What the network sees:")
plt_network_image(image)

print("Pre ReLU activation:")
plt_latent(np.array(image, dtype=np.float32),0,layer="l3")
print("Which ReLU's get triggered:")
plt_latent(np.array(image, dtype=np.float32),0,layer="l3", is_active=True)

# %%
image_id = 12
image  = images[image_id][:,::-1,:]
print("What humans see:")
plt_raw_image(image)
print("What the network sees:")
plt_network_image(image)
print("Activations:")
plt_latent(image,0,layer="l3")
print("Gradient:")
plt_gradient(image,0, lambda a,b,c,d: yaw_probabilty(a,b,c,d,"right"),layer="l3")
print_actions(image)

