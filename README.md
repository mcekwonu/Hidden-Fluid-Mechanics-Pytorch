# Hidden-Fluids-Mechanics-Pytorch
I have translated the original code of Hidden Fluids Models in PyTorch and trained the codes with data (cylinder_nektar_wake.mat) and achieved the same results as obtained by Raissi et al. 

Three variants of neural networks: "vanilla"; plain MLP, "resnet"; Residual networks with skip connections and "Denseresnet"; residual network with implementation of fourier features. The denseresnet NN is not yet fully validated.

The sine activation function ws implemented with option of use of tanh activation and sigmoid linear activation functions respectively.

This implementation uses two dimensional cylinder pass flow data from Raissi(see reference)

Sparse spatio and temporal data training are implemented respectively.

Plots of comparison can be found  as .png and gifs inbuilt the HFM class.




# References:

@article{raissi2017hidden,
  title={Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations},
  author={Raissi, Maziar and Karniadakis, George Em},
  journal={arXiv preprint arXiv:1708.00588},
  year={2017}
}

@article{raissi2017hidden,
  title={Hidden physics models: Machine learning of nonlinear partial differential equations},
  author={Raissi, Maziar and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  year={2017},
  publisher={Elsevier}
}
