# a little bit more details on how to train neural networks

highly recommended to watch: 
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [But how do AI images and videos actually work?](www.youtube.com/watch?v=iv-5mZ_9CPY)

plan for today: implement [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)

## glossary:
- neural networks: parameterized function composed of layers of linear transformations and non-linear activations; can approximate ~any function
- loss: a function comparing the outputs of the neural network with the desired outputs
- gradient descent: the algorithm to "train" neural nets, that is iteratively change their *parameters* so their outputs minimize the loss function
- backpropagation: the algorithm to compute the gradients of the parameters w.r.t the loss, telling gradient descent in which direction to change the outputs
- hyperparameters: the parameters to the gradient descent algorithm
- batching: instead of computing gradients over the entire dataset (expensive) or a single sample (noisy), we compute gradient estimates from small groups of samples called mini-batches.
