# Hidden-Fluids-Mechanics-Pytorch
I have translated the original code of Hidden Fluids Models in PyTorch and trained the codes with data (cylinder_nektar_wake.mat) and achieved the same results as obtained by Raissi et al (reference). 

Three variants of neural networks: 
- "vanilla" - plain MLP, 
- "resnet" - residual networks with skip connections and 
- "Denseresnet" - residual network with implementation of fourier features. The denseresnet NN is not yet fully validated.

The sine activation function ws implemented with option of use of tanh activation and sigmoid linear activation functions respectively.

Sparse spatio and temporal data training are implemented respectively.

![u_compared](https://user-images.githubusercontent.com/17880412/168983735-c5398ab8-840f-4050-8a47-b0e98ef40641.gif)

# References:

- Raissi, Maziar, and George Em Karniadakis. "Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations." arXiv preprint arXiv:1708.00588 (2017).

- Raissi, Maziar, and George Em Karniadakis. "Hidden physics models: Machine learning of nonlinear partial differential equations." Journal of Computational Physics 357 (2018): 125-141.
