# A Differentiable LCP Solver in JAX

[JAX](https://github.com/google/jax) is the new kid on the blok in the machine learning community.
It's fairly young but has quickly grown in popularity.
JAX is the love child of [Autograd](https://github.com/HIPS/autograd) and TensorFlow's [XLA](https://www.tensorflow.org/xla), meaning it combines elegant autograd (`grad`), with fast just-in-time compiled code (`jit`) on CPU or GPU. And all that in native python as a `numpy` drop-in replacement, making it very easy to get started.

I've been looking into JAX for a while now for my research on differentiable physics and decided to write this blog post covering a particular part of the work I've done recently.
Using an autograd library like JAX means you don't have to care about the gradients and the library figures them out for you on it's own. But there are some specific cases where it's beneficial to dig a bit deeper and do some of the legwork yourself.
For me, that was when I implemented a linear complementarity solver (LCP) and needed it to be differentiable. If you don't know or care what an LCP is, this blog post can still be usefull because it shows how te derive the math and implement it in a custom gradient function in JAX (both in forward mode and backward mode). The methodology is the same for other solvers.

## Linear Complementarity Problem (LCP)

A [linear complementarity problem (LCP)](https://en.wikipedia.org/wiki/Linear_complementarity_problem) is probably most used in physics engines to solve contact problems. Solving an LCP corresponds to finding new positions and velocities of parts so that contact constraints are valid.
Generally speaking, you transform the contact constraints in your physics engine to the form of a standard LCP problem. There's a few ways to do that but that's outside the scope of this blog post. [^lcp] provides an example of how to do this.
What you need to know to understand this blog post is that you define a matrix $M \in \mathbb{R}^{n \times n}$ and a vector $q \in \mathbb{R}^n$ (defining the contact problem), and the solver finds vectors $z, w \in \mathbb{R}^n$ (used to correct the position and velocities of the objects) which satisfy these constraints:

* $w = M z + q,$

* $w, z \geqslant 0,$ (that is, each component of these two vectors is non-negative)

* $z^Tw = 0$ or equivalently $\sum\nolimits_i w_i z_i = 0.$ This is the complementarity condition, since it implies that, for all $i$, at most one of $w_i$ and $z_i$ can be positive.

$w$ is a [slack variable](https://en.wikipedia.org/wiki/Slack_variable), which means we generally don't need it and it's just there to convert an inequality ($0 \leqslant Mz+q$) to an equation ($w = M z + q$). We can omit $w$ in the complementarity condition which will be the starting point for the mathematical derivation below:

$$ z^{\mathrm{T}}(Mz+q) = 0$$

Solving the LCP is not possible analytically, so we need a numerical solver which will give us an aproximate solution. I adapted [this]() implementation of [Lemke's algorithm](). I had to change a few things so that the solver was `jit`able in JAX. You can see the result [here]().
Although it could be possible to tweak the solver a bit further so that the JAX autograd operater `grad` would also work, that's not a good idea. You would run into trouble with the variable number of solver steps, and the gradients would have to flow back through all of them, which is not efficient to say the least.
Luckily, we can use a bit of math to circumvent this problem, and define the gradients ourselves using the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem).

### Differentiable LCP

To make the LCP solver differentiable, we define a custom jacobian vector product (JVP) function in backward mode, or a vector jacobian product (VJP) function in forward mode.
Most deep learning research uses backward mode, because you typically take the gradient of a scalar loss with respect to lots of input parameters. Forward mode is more suited if you have only a few inputs and lots of outputs.
You can read more about the difference and how it's implemented in JAX [here](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).

One of the nice things about JAX is that you can define a custom VJP function, and JAX will automagickally transpose it into the corresponding JVP function, if the gradient is taken in backward mode. For me, this was really usefull because I found the mathematical derivation a lot easier in forward mode.

We start with the implicit equation we derived above

$$z^{\mathrm{T}}(Mz+q) = 0$$

and take the differentials of all terms

$$\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}}\mathrm{d}(Mz+q) = 0$$

$$\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}}(\mathrm{d}M z + M\mathrm{d}z+\mathrm{d}q) = 0 .$$
Now we have an expression with six variables. $M$ and $q$ are given, $z$ is solved by the LCP solver, and in forward mode, $\mathrm{d}M$ and $\mathrm{d}q$ are also given. In fact, the only thing we need is $\mathrm{d}z$, so we have to solve this equation for it:
$$\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0$$

We reintroduce $w$ to make the equation simpler:

$$\mathrm{d}z^{\mathrm{T}}w + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0$$

Since $\mathrm{d}z^{\mathrm{T}}w$ is a dot product, we can commutate it:

$$w^{\mathrm{T}} \mathrm{d}z + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0$$

And use the distributive property to isolate $\mathrm{d}z$:

$$(w^{\mathrm{T}} + z^{\mathrm{T}} M) \mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0$$

Arriving at the solution for $\mathrm{d}z$

$$\mathrm{d}z = -(w^{\mathrm{T}} + z^{\mathrm{T}} M)^{-1} z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q)$$

This is the implementation in JAX:

```python
import jax.numpy as np

def lcp_jvp(primals, tangents):
    M, q = primals
    dM, dq = tangents
    z = lcp_forward(M, q)
    w = np.matmul(M, z) + q
    dz = - np.linalg.pinv(np.diag(w) + z[:, np.newaxis] * M) @ (z * (dM @ z + dq))
    return z, dz

lcp_forward.defjvp(lcp_jvp)
```

where `lcp_forward()` is the LCP solver function. You can read more on how to define custom JVP's or VJP's [here]().

test

backward mode

benefits


**Conclusion**

Other work: talk from matthew, jax docs.


## References

[^lcp]: Glocker, Christoph, and Christian Studer. "Formulation and preparation for numerical evaluation of linear complementarity systems in dynamics." Multibody System Dynamics 13.4 (2005): 447-463.