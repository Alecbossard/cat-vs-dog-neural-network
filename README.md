🧠 Cat vs. Dog Classification from Scratch

This project implements several neural network models built entirely from scratch using NumPy — no TensorFlow, no PyTorch, no Keras.
The objective is to understand the inner mechanics of deep learning by manually coding forward propagation, backpropagation, activation functions, and gradient descent.

🐱🐶 Goal

Classify images of cats and dogs using a deep neural network (DNN) written from scratch.
The project progresses from a single-neuron classifier to a multi-layer architecture, while keeping the math explicit and educational.

🚧 Why it doesn’t (yet) work well on real images

This network currently uses only fully connected (Dense) layers, which are not well suited for image data.
Here’s why:

Loss of spatial structure → images are flattened into a 1D vector, so the model ignores spatial patterns (edges, shapes, textures).

Too many parameters → a 64×64×3 image = 12,288 inputs; even one dense layer with 100 neurons has over 1.2 million weights, making it inefficient and unstable.

No convolution or pooling → the model cannot detect local features; CNNs solve this by learning filters over small regions.

Limited generalization → the small dataset and lack of normalization lead to overfitting and poor test accuracy.

👉 In short: this project focuses on how deep learning works mathematically, not on achieving production-level accuracy.

🧩 Educational goal

This repository is meant to be a learning tool, showing:

how data flows through a neural network,

how gradients are computed,

and how updates modify weights step by step.
