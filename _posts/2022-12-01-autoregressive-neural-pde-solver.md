---
layout: distill
title: Autoregressive Renaissance in Neural PDE Solvers
description: Recent developments in the field of neural partial differential equation (PDE) solvers have placed a strong emphasis on neural operators. However, the paper "Message Passing Neural PDE Solver" by Brandstetter et al. published in ICLR 2022 revisits autoregressive models and designs a message passing graph neural network that is comparable with or outperforms both the state-of-the-art Fourier Neural Operator and traditional classical PDE solvers in its generalization capabilities and performance. This blog post delves into the key contributions of this work, exploring the strategies used to address the common problem of instability in autoregressive models and the design choices of the message passing graph neural network architecture.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
 authors:
   - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2022-12-01-autoregressive-neural-pde-solver.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#toc:
#  - name: Equations
#  - name: Images and Figures
#    subsections:
#    - name: Interactive Figures
#  - name: Citations
#  - name: Footnotes
#  - name: Code Blocks
#  - name: Layouts
#  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

#	Autoregressive Renaissance in Neural PDE Solvers

## Introduction
> Improving PDE solvers has trickle down benefits to a vast range of other fields.

Partial differential equations (PDEs) play a crucial role in modeling complex systems and understanding how they change over time and in space. Not only are they ubiquitous across physics and engineering, modeling a wide range of physical phenomena, including heat transfer, sound waves, electromagnetism, and fluid dynamics, but they can also be used in finance to model the behavior of financial markets, in biology to model the spread of diseases, and in computer vision to model the processing of images.

Despite their long history, dating back to equations first formalized by Euler over 250 years ago, finding numerical solutions to PDEs continues to be a challenging problem. Solving PDEs involves finding the solution for the unknown variables that describe the system, which can be challenging because of the complexity of the equations and the large amount of data that must be processed.

Classical numeric solvers, such as the finite difference method (FDM) and spectral methods, have proven to be effective in solving PDEs when analytical solutions are not possible, but they can become computationally complex, particularly in the face of nonlinear, high-dimensional, and/or multiscale systems.

The recent advances in machine learning and artificial intelligence have opened up new possibilities for solving PDEs in a more efficient and accurate manner. These developments have the potential to revolutionize many fields by providing a more efficient and accurate way of solving PDEs, leading to a better understanding of complex systems and the ability to make more informed predictions about their behavior.

## Background
### Let\'s brush up on the basics...
*The notation and definitions provided match those in the paper for consistency, unless otherwise specified.*

Ordinary differential equations (ODEs) describe the change of a dependent variable with respect to a single independent variable. In contrast, PDEs are mathematical equations that describe the behavior of a dependent variable as it changes with respect to several independent variables over a region of space and time.

Formally, for one time dimension and possibly multiple spatial dimensions denoted by $\textbf{x}=[x_{1},x_{2},x_{3},\text{...}]^{\top} \in \mathbb{X}$, a general (temporal) PDE may be written as

$$\partial_{t}\textbf{u}= F\left(t, \textbf{x}, \textbf{u},\partial_{\textbf{x}}\textbf{u},\partial_{\textbf{xx}}\textbf{u},\text{...}\right) \qquad (t,\mathbf{x}) \in [0,T] \times \mathbb{X}$$

 - Initial condition:
 $\mathbf{u}(0,\mathbf{x})=\mathbf{u}^{0}(\mathbf{x})$ for $\mathbf{x} \in \mathbb{X}$
 
 - Boundary conditions:
 $B[ \mathbf{u}](t,x)=0$ for $(t,\mathbf{x}) \in [0,T] \times \partial \mathbb{X}$

<div class="fake-img l-gutter">
  <p>

  Many equations are solutions to such PDEs alone. For example, the wave equation is given by $\partial_{tt}u = \partial_{xx}u$. You will find that any function in the form $u(x,t)=F(x-ct)+G(x+ct)$ is a potential solution. Initial conditions are used to specify how a PDE "starts" in time, and boundary conditions determine the value of the solution at the boundaries of the region where the PDE is defined.

  </p>
</div>

The study of PDEs is in itself split into many broad fields. Briefly, these are two other important properties in addition to the initial and boundary conditions:

<details><summary>Linearity</summary>

- Linear: the highest power of the unknown function appearing in the equation is 1 (i.e., a linear combination of the unknown function and its derivatives)
- Nonlinear: the highest power of the unknown function appearing in the equation is greater than 1

</details>

<details><summary>Homogeneity</summary>

- Homogeneous: PDEs with no constant terms (i.e., the right-hand side is equal to zero)
- Inhomogeneous: PDEs with a non-zero constant term on the right-hand side

</details><br/>


PDEs can be either linear or nonlinear, homogeneous or inhomogeneous, and can contain a combination of constant coefficients and variable coefficients. They can also involve a variety of boundary conditions, such as Dirichlet, Neumann, and Robin conditions, and can be solved using analytical, numerical, or semi-analytical methods.

Brandstetter et al. follow precedence set by Li et al. and Bar-Sinai et al. to focus on PDEs written in conservation form:

$$\partial_{t} \mathbf{u} + \nabla \cdot \mathbf{J}(\mathbf{u}) = 0$$

 - $J$ is the flux, or the amount of a physical quantity that is flowing through a particular surface or region at a given time

 - $\nabla \cdot J$ is the divergence of the flux, which can be thought of as the amount of outflow of the flux at a given point

Additionally, they consider Dirichlet and Neumann boundary conditions. Dirichlet boundary conditions prescribe a fixed value of the solution at a particular point on the boundary of the domain. Neumann boundary conditions, on the other hand, prescribe the rate of change of the solution at a particular point on the boundary. Not considered are mixed boundary conditions, which involve both Dirichlet and Neumann conditions, and Robin boundary conditions, which involve a linear combination of the solution and its derivatives at the boundary.

### Solving PDEs the classical way
A brief search in a library will find numerous books detailing how to solve various types of PDEs. A few notable classical methods are introduced; since Brandstetter et al. proposes to numerically solve PDEs, numerical methods are discussed in more detail.

<details><summary>Analytical methods, where an exact solution to a PDE can be found by mathematical means.</summary><br/>

 - Separation of Variables
	 - This method involves expressing the solution as the product of functions of each variable, and then solving each function individually. It is mainly used for linear PDEs that can be separated into two or more ordinary differential equations.
 - Green's Functions
	 - This method involves expressing the solution in terms of a Green's function, which is a particular solution to a homogeneous equation with specified boundary conditions.
</details><br/>

<details><summary>Semi-analytical methods: where an analytical solution is combined with numerical approximations to find a solution to a PDE.</summary><br/>

- Perturbation methods
	- This method is used when the solution to a PDE is close to a known solution or is a small deviation from a known solution. The solution is found by making a perturbation to the known solution and solving the resulting equation analytically.
- Asymptotic methods
	- In this method, the solution is represented as a series of terms that are solved analytically. The solution is then approximated by taking the leading terms of the series.
	</details><br/>

Very few PDEs have analytical solutions, and so numerical methods have been developed to approximate PDE solutions over a much wider range of potential problems.

#### Numerical Methods
When the solution to a PDE cannot be found analytically, numerical approximations are used instead. Often, approaches for temporal PDEs follow the method of lines where the domain is discretized onto a grid. Every point of the grid is then thought of as a separate ODE evolving in time, enabling the use of ODE solvers such as Runge-Kutta methods.

<details open><summary>1. Partitioning the problem onto a grid</summary><br/>

The discretization process is a topic of continual discussion. In the most basic case, arbitrary spatial and temporal resolutions $\mathbf{n_{x}}$ and $n_{t}$ can be chosen and thus used to create a grid where $\mathbf{n_{x}}$ is a vector containing a resolution for each spatial dimension $\mathbf{x}$. The domain may also be irregularly sampled, resulting in a grid-free discretization. FDM or any other time discretization technique can be used to discretize the time domain. One direction of ongoing research seeks to determine discretization methods which can result in more efficient numerical solvers (for example, take larger steps in flatter regions and smaller steps in rapidly changing regions).

</details><br/>

<details open><summary>2. Estimating the spatial derivatives</summary><br/>

How to approximate the spatial derivatives is the second major choice in the method of lines. A popular choice when using a gridded discretization is the finite difference method (FDM). Spatial derivative operators are replaced by a stencil which indicates how values at a finite set of neighboring grid points are combined to approximate the derivative at a given position. This stencil is based on the Taylor series expansion.

-INSERT VIDEO HERE-

The finite volume method (FVM) is another approach which works for irregular geometries. Rather than requiring a grid, the computation domain can be divided into discrete, non-overlapping control volumes used to compute the solution for that portion. For every control volume, a set of equations describing the balance of some physical quantities (in essence, estimating the flux at control volume boundaries) can be solved which results in the approximated spatial derivative. While this method only works for conservation form equations, it can handle complex problems with irregular geometries and fluxes that are difficult to handle with other numerical techniques such as the finite difference method (FDM).

The basic idea behind the pseudospectral method (PSM) is to represent the solution as a sum of basis functions, typically in Fourier space. The coefficients of the basis functions are determined by using a set of collocation points, which are chosen to enforce the solution at these points. The partial differential equation (PDE) is transformed into a discrete form by integrating the PDE over the control volume and substituting the basis function representation of the solution into the PDE. The solution (i.e. the approximated spatial derivatives) is then calculated by evaluating the basis function representation at the collocation points. The pseudo-spectral method is well-suited for solving problems with smooth solutions and periodic boundary conditions, but its performance drops for irregular or non-smooth solutions.

</details>

<details open><summary>3. Time updates</summary><br/>

The resulting problem is a set of temporal ODEs which can be solved with classical ODE solvers such as any member of the Runge-Kutta method family.

</details>

#### Limitations of Classical Methods

From the high level descriptions alone, certain methods are best suited for specific types of PDEs. The properties of a PDE, such as its order, linearity, homogeneity, and boundary conditions, determine its solution method. Different methods have been developed based on the different properties and requirements of the problem at hand. Brandstetter at al. categorizes these requirements into the following:

| User | Structural | Implementational |
|--|--|--|
| Computation efficiency, computational cost, accuracy, guarantees (or uncertainty estimates), generalization across PDEs | Spatial and temporal resolution, boundary conditions, domain sampling regularity, dimensionality | Stability over long rollouts, preservation of invariants |

The countless combinations of requirements resulted in what Bartels defines as a *splitter field* in CITE. A specialized classical solver is developed for many sub-problems.

These methods, while effective and reliable due often having error guarantees, often come at high computation costs. Taking into account that PDEs often exhibit chaotic behaviour and are sensitive to any changes in their parameters, re-running a solver every time a coefficient or boundary condition changes can be computationally expensive.

Furthermore, classical schemes become intractable or run into computation walls for a variety of reasons including complex geometries, large systems, high dimensionality, nonlinearities, and coupled problems (in which some unknowns are interdependent).

<div class="fake-img l-gutter">
  <p>

A formalized example rises from Courant–Friedrichs–Lewy (CFL) condition, which states that the maximum time step size should be proportional to the minimum spatial grid size. According to this condition, as the number of dimensions increases, the size of the temporal step must decrease and therefore numerical solvers become very slow for complex PDEs.

  </p>
</div>


### Neural Solvers

Neural solvers offer some very desirable properties that may serve to unify some of this splitter field. Neural networks can learn and generalize to new contexts such as different initial/boundary conditions, coefficients, or even different PDEs entirely. They can also circumvent the CFL condition, making them a promising avenue for solving highly complex PDEs such as those found in weather prediction. Brandstetter et al. categorizes neural solvers into two categories, neural operator methods and autoregressive methods; one more called "PINN methods" is added to address the large family of PINNs. These are not mutually exclusive - a key example being that PINNs can be considered a finite dimensional neural operator.

**PINN methods**

Raissi et al. coined the physics-informed neural network (PINN) in 2017. The problem is set such that the network $\mathcal{N}$ satisfies $\mathcal{N}(t,\mathbf{u}^{0}) = \mathbf{u}(t)$ where $\mathbf{u}^{0}$ are the initial conditions. The main principle behind PINNs is to enforce the governing physical laws of the problem on the network's predictions by adding loss term(s) to the network's objective function. Since the model can still learn and make predictions that deviate from the constraint, this is a common method to impose a soft constraint in the form of a physics prior which encourages the model to converge to a more optimal solution.

For a typical loss function
$$\theta = \text{argmin}_{\theta} \mathcal{L}(\theta)$$

the loss with a physics prior may be defined as follows.

 $$\mathcal{L}(\theta) = \omega_{\mathcal{F}} \mathcal{L}_{\mathcal{F}}(\theta) + \omega_{\mathcal{B}} \mathcal{L}_{\mathcal{B}}(\theta) + \omega_{d} \mathcal{L}_{\text{data}}(\theta)$$
 
 
| Term | Definition | Effect |
|--|--|--|
| $$\mathcal{L}_{\mathcal{B}}$$ | Loss wrt. the initial and/or boundary conditions | Fits the known data over the network |
| $$\mathcal{L}_{\mathcal{F}}$$ | Loss wrt. the PDE | Enforces DE $\mathcal{F}$ at collocation points;  Calculating using autodiff to compute derivatives of $\mathbf{\hat{u}_{\theta}(\mathbf{z})}$ |
| $$\mathcal{L}_{\text{data}}$$ | Validation of known data points | Fits the known data over the NN and forces $\mathbf{\hat{u}}_{\theta}$ to match measurements of $\mathbf{u}$ over provided points |

In practice, each loss term could be implemented using a mean square error formulation. As a whole, solving a PDE becomes a loss function optimization problem and this approach can be applied in both forward and inverse problems and using both unsupervised and supervised learning methodologies (depending on whether the underlying PDE is known or not).

PINNs were first introduced using a standard multilayer perceptron architecture, but the methodology has now been applied to CNNs, RNNs, Bayesian neural networks, and even GANs. Some interesting developments aside from architectural changes include the Deep Ritz Method CITE, where the loss is defined as the energy of a problem's solution, and the Deep Galerkin Method CITE, where the loss is given by multiplying the residual by a test function.

The success of this loss-based approach is apparent when considering the rapid growth of papers which extend the original iteration of the PINN. However, Krishnapriyan et al. [10] has shown that even though standard fully-connected neural networks are theoretically capable of representing any function given enough neurons and layers, a PINN may still fail to approximate a solution due to the complex loss landscapes arising from soft PDE constraints. Even when extended to other simple cases such as advection, a fully connected neural network with the physics prior fails to solve the problem [10]. While the soft penalty is relatively quick to implement and can incorporate multiple rules to a model, it often results in complex and non-convex loss functions which are challenging to optimize. Another major barrier is that PINNs are trained for specific parameters and do not tend to generalize well to new instances of even the same PDE.

**Neural operator methods**

Neural operator methods model the solution of a PDE as an operator that maps inputs to outputs. Much the same as above, the problem is set such that a neural operator $\mathcal{M}$ satisfies $\mathcal{M}(t,\mathbf{u}^{0}) = \mathbf{u}(t)$ where $\mathbf{u}^{0}$ are the initial conditions. While the PINN methods represent the solution as a set of parameters learned by a neural network, neural operator methods model the solution as an operator.

> Intuitively, a neural operator can be thought of to map functions to other functions, both of which are infinite-dimensional (as opposed PINNs which input and output discretized data sampled from such functions).

The deep O-net architecture was the first of its kind and consisted of a branch net which encodes discrete inputs to an input function space, and a single trunk net which encodes the locations to evaluate the output function.

IMAGE, change/extend the caption

CAPTION: Illustrations of the problem setup and architectures of DeepONets. (A) The network to learn an operator $G:u \mapsto G(u)$ takes two inputs $[u(x_{1}), u(x_{2}), . . . , u(x_{m})]$ and $y$. (B) Illustration of the training data. For each input function $u$, there must be the same number of evaluations at the same scattered sensors $x_{1}, x_{2}, . . . , x_{m}$. However, there are no constraints on the number or locations for the evaluation of output functions. (C) The stacked DeepONet has one trunk network and $p$ stacked branch networks. (D) The unstacked DeepONet has one trunk network and one branch network.

One of the current state-of-the-art models is the FNO. It operates within Fourier space and takes advantage of the convolution theorem (The Fourier transform of the convolution of two signals is equal to the pointwise product of their individual Fourier transforms) to place the integral kernel in Fourier space as a convolutional operator. These global integral operators (implemented as Fourier space convolutional operators) are combined with local nonlinear activation functions, resulting in an architecture which is highly expressive yet computationally efficient, as well as being resolution-invariant. While the vanilla FNO required the input function to be defined on a grid due to its reliance on the FFT, further work developed mesh-independent variations as well.

IMAGE

Neural operators on the whole generalize better than PINN methods since they directly approximate operators. They are typically able to operate on multiple domains and can be implemented to be completely data-driven. However, these models do not tend to predict out-of-distribution $t$ and are therefore limited when dealing with temporal PDEs. A major barrier to both neural operator and PINN methods remains to be their relative lack of interpretability and guarantees compared to classical solvers. 

**Autoregressive methods**

While PINN and neural operator methods directly mapped inputs to outputs, autoregressive methods take an iterative approach instead. For example, iterating over time results in a problem such as $\mathbf{u}(t+\Delta t) = \mathcal{A}(\Delta t, \mathbf{u}(t))$ where $\mathcal{A}$ is some temporal update. Similarly to RNNs, autoregressive models take previous time steps to predict the next time step. However, autoregressive models are entirely feed-forward and take the previous predictions as inputs rather than storing them in some hidden state. These sequence models have been used for images (the first deep autoregressive model being PixelCNN CITE), audio (WaveNet), and text (Transformer). When applied to PDEs, one key benefit is their iterative nature which brings classical solvers back to mind: three autoregressive works mentioned by Brandstetter et al. are hybrid methods which use neural networks to predict certain parameters for finite volume, multigrid, and iterative finite elements methods. All three retain a (classical) computation grid which makes them somewhat interpretable.

Hsieh et al., for example, develops a neural network-accelerated iterative finite elements method. For any PDE with an existing linear iterative solver, a learned iterator can replace a handcrafted classical iterator resulting in faster convergence and better generalization to different geometries and boundary conditions. Most significantly, their approach offers theoretical guarantees of convergence and correctness.

However, autoregressive models have not gained the acclaim seen by neural operators as a whole. This is on one hand due to their limitations in scope - in Hsieh et al.'s case, an existing numerical method must already exist, and while prediction times may be faster than the classical comparisons, designing and training this learned iterator may counterweight the perceived benefit. Another major concern is the accumulation of error, which is particularly detrimental for PDE problems that often exhibit chaotic behavior.

## Message Passing Neural PDE Solver (MP-PDE)

Brandstetter et al. propose a fully neural PDE solver which capitalizes on neural message passing. The overall architecture is laid out below, consisting of an MLP encoder, a GNN processor, and a CNN decoder.

At its core, this model is autoregressive and thus faces the same challenge listed above. Two key contributions of this work are the pushforward trick and temporal bundling which mitigate the potential butterfly effect of error accumulation.

### The Pushforward Trick and Temporal Bundling

IMAGE

During testing, the model should predict time steps that are then used as inputs to predict the following time steps. This results in a distribution shift problem, because the inputs are no longer solely from data and the distribution learned during training will always be an approximation of the true data distribution. The model will appear to overfit to the one-step training distribution and perform poorly the further it continues to predict.

An adversarial-style stability loss is added to the one-step loss so that the training distribution is brought closer to the test time distribution:

$$L_{\text{one-step}} = \mathbb{E}_{k}\mathbb{E}_{\mathbf{u^{k+1}|\mathbf{u^{k},\mathbf{u^{k} \sim p_{k}}}}}[\mathcal{L}(\mathcal{A}(\mathbf{u}^{k}),\mathbf{u}^{k+1}]$$

$$L_{\text{stability}} = \mathbb{E}_{k}\mathbb{E}_{\mathbf{u^{k+1}|\mathbf{u^{k},\mathbf{u^{k} \sim p_{k}}}}}[\mathbb{E}_{\epsilon | \mathbf{u}^{k}}[\mathcal{L}(\mathcal{A}(\mathbf{u}^{k}+\epsilon),\mathbf{u}^{k+1}]]$$

$$L_{\text{total}} = L_{\text{one-step}} + L_{\text{stability}}$$

where $k$ is the iteration, $\mathbf{u^{k}}$ is the solution at iteration $k$, $p_{k}$ is the distribution at $k$, $\mathcal{A}$ is the temporal update, and $\epsilon | \mathbf{u}^{k}$ is an adversarial perturbation. The pushforward trick lies in the choice of $\epsilon$ such that $\mathbf{u}^{k}+\epsilon = \mathcal{A}(\mathbf{u}^{k-1})$, similar to the test time distribution. Practically, $\epsilon$ is implemented to be noise from the network itself so that as the network improves, the loss decreases.

Necessarily, the noise of the network must be known or calculated to implement this loss term. So, the model is unrolled for 2 steps but only backpropagated over the most recent unroll step which already has the neural network noise.

**Temporal bundling**

IMAGE

This trick complements the previous by reducing the amount of times the test time distribution changes. Rather than predicting a single value at a time, the MP-PDE predicts multiple time-steps at a time, as seen above.

### Network Architecture

Similarly to the hybrid autoregressive methods, the MP-PDE also draws inspiration from classical methods. The steps taken in classical numerical solvers can be related to the network architecture.

IMAGE

|Classical Numerical Method| MP-PDE Network Component |
|--|--|
| Partitioning the problem onto a grid | Encoder <br />*Encodes a vector of solutions into node embeddings* |
| Estimating the spatial derivatives | Processor <br />*Estimates spatial derivatives via message passing* |
| Time updates | Decoder <br />*Combines some representation of spatial derivatives smoothed into a time update* |

1. Encoder

The encoder is implemented as a two-layer MLP which computes an embedding for each node $i$ using a vector of previous solutions (the length equaling the temporal bundle length), the node's position, the current timestep, and equation parameters.

2. Processor

The node embeddings from the encoder are then used in a message passing GNN. Since there are no restrictions on the node positions, they form a potentially non-regular integration grid of sorts. The message passing algorithm is run $M$ steps using the following updates.

$\text{edge } j \to i \text{ message:} \qquad \mathbf{m}_{ij}^{m} = \phi (\mathbf{f}_{i}^{m}, \mathbf{f}_{j}^{m},$  <span style="color:steelblue;">$\mathbf{u}_{i}^{k-K:k}-\mathbf{u}_{j}^{k-K:k}$</span>, <span style="color:#6546b4;">$\mathbf{x}_{i}-\mathbf{x}_{j}$</span>, <span style="color:#46b49c;">$\theta_{PDE})$</span>
$\text{node } i \text{ update:} \qquad \mathbf{f}_{i}^{m+1} = \psi (\mathbf{f}^{m}_{i}, \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{m}, \phi_{PDE}) \qquad \quad \:$

