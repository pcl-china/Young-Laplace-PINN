# Young-Laplace PINN
This is an open source PINN code for solving the Young-Laplace equation in a tubular domain.
# Paper Citation
This paper is in the under review stage.
# Case
To facilitate replication of this work by other scholars, we provide below a detailed explanation of the calculations in Fig. 4 of the paper. The results of Fig. 4 can be reproduced by running the **capillary_rise_1.py** file available at https://github.com/pcl-china/Young-Laplace-PINN/blob/main/capillary_rise_1.py

In this file:

•	Lines 1-8 are used to import external libraries in Python.

•	Lines 11-27 define the **DNN** class, which inherits the **torch.nn.Module** module and is used to construct a custom network structure for the neural network.

•	Lines 30-33 indicate that if a **CUDA** device is available, it will be used preferentially for **GPU** parallel computing.

•	Lines 35-69 define a **PhysicsInformedNN** class, which is used for PINN computations and to store PINN computation information. When declaring a **PhysicsInformedNN** object, it is necessary to input the coordinates of points within the domain **x_area**, boundary point coordinates **x_bc**, unit normal vectors of the boundary points **n_bc**, material contact angle cos_angle, elemental area **dxdy**, network structure **layers**, capillary diameter **d**, capillary length **l**, and the domain size **ub** and **lb**. Additionally, **lambda_lamb** is included as a learnable parameter in the network optimization. Two optimizers are also defined in this class, with learning rates, maximum iteration numbers, and other parameters defined through **torch.optim.LBFGS()** and **torch.optim.Adam()**, details of which can be referred to in the PyTorch official documentation (torch.optim — PyTorch 2.1 documentation).

•	Lines 71-75 perform normalization of the data before inputting it into the network. This is done because if the input coordinates exceed the range of -1 to 1, the network could face the issue of gradient vanishing.

•	Lines 77-89 define a function **net_f** for computing the domain residual **f** and the fluid volume **v** of the Young-Laplace equation, where **torch.autograd.grad()** utilizes PyTorch's built-in Automatic Differentiation technology.

•	Lines 91-96 define a function **net_bc_natural** to compute the residuals of Young's equation on the boundary.

•	Lines 98-120 compute the boundary loss, PDE loss within the domain, and the volume loss, and then sum these losses with weights to obtain the total loss. It should be noted that, based on our experience, the weights for each term should be set such that the three types of losses are on the same order of magnitude. The values of each loss are then recorded and output.

•	Lines 122-126 update the L-BFGS iterations, as **optimizer_LBFGS.step()** requires a closure function, details of which can be found in the documentation (torch.optim — PyTorch 2.1 documentation).

•	Lines 128-137 use the ADAM and L-BFGS optimizers sequentially to train the network.

•	Lines 140-158 output the final results of the network. This module is for display purposes and does not affect the training process; readers can customize it as needed.

•	Lines 161-175 include a function named find_normal() to determine the unit normal vector at any given boundary point.

•	Lines 178-186 involve inputting domain points, labeled x_area, and boundary points, labeled bc, with an option to specify the deletion of points either inside or outside the boundary.

•	Lines 189-210 describe the main function of the code.

•	Lines 211-end focus on the post-processing of the computational results.

# Note
The PINN program is written based on the open source deep learning framework Pytorch version 1.10.2 in python.
