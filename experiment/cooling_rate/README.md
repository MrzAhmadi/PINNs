# Mathematical Formulations for the Cooling Process PINN

### Newton’s Law of Cooling

The code models a process based on Newton’s Law of Cooling. This law says that how quickly an object loses heat depends on the difference between its temperature and the temperature of its surroundings.

The governing Ordinary Differential Equation (ODE) is:

$$\frac{dT(t)}{dt} = r(T_{env} - T)$$

Where:

* $T$ is the temperature of the object at time $t$.
* $T_{env}$ is the ambient environmental temperature.
* $r$ is the cooling rate constant.

Analytical Solution

Integrating the ODE yields an exact solution. This solution is used in the cooling_law function:

$$T(t) = (T(0) - T_{env})e^{-rt} + T_{env}$$

$T_0$ is the object’s starting temperature when $t = 0$.

---

### Neural Network Optimization (Data Loss)

A standard Multi-Layer Perceptron (Net) learns to predict temperature $T$ from time $t$. The data loss uses Mean Squared Error (MSE) to compare the network’s predictions with the noisy training data.

$$\mathcal{L}{data} = \frac{1}{N} \sum{i=1}^{N} \left( T_{pred}(t_i) - T_{true}(t_i) \right)^2$$

Where $N$ is the number of training data points.

---

### Physics-Informed Loss (Forward Problem)

To make sure the neural network follows the physics, a physics loss is added in the physics_loss function. This loss calculates the ODE residual by using automatic differentiation to find the gradient $\frac{dT}{dt}$.

The residual $f$ is defined as:

$$f = \frac{dT}{dt} - r(T_{env} - T)$$

The physics loss is the Mean Squared Error of this residual, calculated over a set of collocation points $N_f$:

$$\mathcal{L}{physics} = \frac{1}{N_f} \sum{j=1}^{N_f} \left( \left. \frac{dT}{dt} \right|{t_j} - r(T{env} - T_{pred}(t_j)) \right)^2$$

The total loss for training the network combines the data loss and the weighted physics loss:

$$\mathcal{L}{total} = \mathcal{L}{data} + \lambda \mathcal{L}_{physics}$$

$\lambda$ is the loss2_weight parameter.

---

### Parameter Discovery (Inverse Problem)

In the NetDiscovery class, the cooling rate $r$ is unknown. The network must fit the temperature curve and also find the true cooling rate.

A lA learnable parameter $\hat{r}$ is added. The new residual is:$f_{discovery} = \frac{dT}{dt} - \hat{r}(T_{env} - T)$$

The corresponding discovery physics loss implemented in physics_loss_discovery is:

$$\mathcal{L}{physics_discovery} = \frac{1}{N_f} \sum{j=1}^{N_f} \left( \left. \frac{dT}{dt} \right|{t_j} - \hat{r}(T{env} - T_{pred}(t_j)) \right)^2$$

The network uses backpropagation to minimize this loss. This process helps it adjust both the weights used to predict $T(t)$ and the parameter $\hat{r}$ to find the true cooling rate $r$.