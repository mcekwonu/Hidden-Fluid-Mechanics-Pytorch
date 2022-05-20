# Hidden-Fluid-Mechanics-Pytorch
Original codes of Hidden Fluids Models in PyTorch and trained the codes with data (cylinder_nektar_wake.mat) of Raissi et al (reference). 

Three variants of neural networks: 
- "vanilla" - plain MLP, 
- "resnet" - residual networks with skip connections and 
- "Denseresnet" - residual network with implementation of fourier features. The denseresnet NN is not yet fully validated.

The sine activation function is implemented with options of tanh and sigmoid linear activation functions respectively.

Sparse spatio and temporal data training are implemented respectively with the velocity fields and predicted the pressure and vorticity.

![u_compared](https://user-images.githubusercontent.com/17880412/168983735-c5398ab8-840f-4050-8a47-b0e98ef40641.gif)  
![v_compared](https://user-images.githubusercontent.com/17880412/168983837-cbb87af0-a7ab-48b4-b8f2-619308fb9e16.gif)
![vorticity](https://user-images.githubusercontent.com/17880412/168983999-24399b39-6c42-4961-aa9f-ccf94441cb5e.gif)

# References:
- Raissi, Maziar, Alireza Yazdani, and George Em Karniadakis. ["Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations."] Science 367.6481 (2020): 1026-1030. (https://www.science.org/doi/full/10.1126/science.aaw4741)

- Raissi, Maziar, Alireza Yazdani, and George Em Karniadakis. ["Hidden Fluid Mechanics: A Navier-Stokes Informed Deep Learning Framework for Assimilating Flow Visualization Data."] arXiv preprint arXiv:1808.04327 (2018).(https://arxiv.org/pdf/1808.04327.pdf)

- Raissi, Maziar, and George Em Karniadakis. ["Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations." arXiv preprint arXiv:1708.00588 (2017).] (https://arxiv.org/pdf/1708.00588.pdf)

- Raissi, Maziar, and George Em Karniadakis. "Hidden physics models: Machine learning of nonlinear partial differential equations." Journal of Computational Physics 357 (2018): 125-141.
