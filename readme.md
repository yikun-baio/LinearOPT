# Linear optimal partial transport 
Example code for linear optimal partial transport as introduced in the paper. The code repdocues the numerical examples in section 4 in the paper. 

# Required packages
Install [numba](https://numba.pydata.org/), [pytorch](https://pytorch.org), [pythonOT](https://pythonot.github.io) before clone this repository. 


# Bakcground: Optimal partial transport  
Choose open set $\Omega\subset\mathbb{R}^d$, let $M_+(\Omega)$ denote the set of all positive randon measures defined on $\Omega$ and select $\mu^0,\mu^1,\mu^2$. 
The following are the kantorovich formulation: 
$$OPT(\mu^0,\mu^1):=\inf_{\gamma\in\Gamma_{\leq}(\mu^0,\mu^1)}\int_{\Omega^2}\|x-y\|^2d\gamma(x,y)+\lambda(|\mu^0|+|\mu^1|-2|\gamma|) \hspace{1em}(1)$$

where $\Pi_{\leq}(\mu^0,\mu^1)=\{\gamma\in \mathcal{M}_+(\Omega^2): \gamma_0\leq \mu^0,\gamma_1\leq \mu^1\},$ $\gamma_0,\gamma_1$ are the first and second marginals of $\gamma$. 

# Outline of repository
- lib/lin_lopt.py contains a simple OPT solver based on LP in [PythonOT](https://pythonot.github.io/), basic implementations for linearized OT and OPT embeddings, and a OT barycenter solver. 
- lib/geodesic.py contains basic implementations for OT geodesic, LOT geodesic, OPT interporlation and LOPT interporlation. 

# Examples: 
- Run accuracy_test.ipynb to see the accuracy performance of LOPT distance. 
- Run wall_clock_time.ipynb to see the wall clock test of OPT and LOPT. 
- Run geodesic_MNIST.ipynb to see the comparison between OT geodesic, LOT geodesic, OPT interpolation, LOPT interporlations on MNIST data. 
- Run pca.ipynb to see the comparison between LOT embeddings and LOPT embeddings 
