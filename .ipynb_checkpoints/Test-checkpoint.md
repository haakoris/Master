# Tensorflow Probability - An introduction to concepts and functions

This is based on the *Probabilistic Deep Learning With Python*, and is intended to give an introduction to the notation and syntax of the *Tensorflow Probability* package
___
## The *DistributionLambda* layer
The `tfp.layers.DistributionLambda` layer can be used to construct the conditional probability distribution $P(y|,x,w)$.<br>

### Constant standard deviation
As an example, let's use a normal distribution $N(\mu_x, \sigma^2)$ with constant $\sigma=1$ and a location parameter $\mu_x$. Example code:
```Python
inputs = tf.keras.layers.Input(shape=(1,))
params = tf.keras.layers.Dense(1)(inputs)
dist = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(loc=t, scale=1)
)(params)
model = tf.keras.models.Model(inputs, outputs=dist)
model.compile(tf.keras.optimizisers.Adam(), loss=negloglik)
```
The `dist` layer creates a distributional layer using a normal distribution where $\mu_x$ is given by the output from the `Dense` layer, and the standard deviation is constant as 1. The loss `negloglik` is the negative log-likelihood of the distribution. Finally, $\sigma$ can be estimated from the variance of the residuals. 
___
### Nonconstant standard deviation
Using a constant standard deviation will often yield worse results. Hence, allowing the model to learn this parameter as well is desirable.  To avoid negative outputs for $\sigma$, it can be fed into an exponential function, e.g., the *softplus* function. Example code:
```Python
inputs = tf.keras.layers.Input(shape=(1,))
params = tf.keras.layers.Dense(2)(inputs)
dist = tfp.layers.DistributionLambda(
    lambda t: tfd.Normal(loc=t[..., 0:1], 
                         scale=1e-3 + tf.math.softplus(0.05 * t[...,1:2])
                        ))(params)
model = tf.keras.models.Model(inputs, outputs=dist)
model.compile(tf.keras.optimizisers.Adam(), loss=negloglik)
```
The change from earlier is reflected in two steps:
* To estimate the standard deviation, the `Dense` layer now outputs two values
* The `DistributionLambda` uses the first output as the mean, and the second as the standard deviation. It is accomplished by using `t[..., 0:1]` - corresponding to the mean - and `t[..., 1:2]`, which equates to taking the second output as $\sigma$.

To ensure a non-negative scale, a small constant of `1e-3` is added to the `softmax` output. 
