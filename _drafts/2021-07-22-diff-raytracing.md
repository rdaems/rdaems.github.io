---
layout:     post
title:      "Implicit function theorem and JAX"
subtitle:   "Application of a differentiable ray tracer with signed distance fields"
date:       2021-07-22 12:00:00
author:     "Rembert Daems"
header-style: text
catalog:    true
mathjax:    true
tags:
    - research
---

[JAX](https://github.com/google/jax) is the new kid on the blok in the machine learning community.
It's fairly young but has quickly grown in popularity.
JAX is the love child of [Autograd](https://github.com/HIPS/autograd) and TensorFlow's [XLA](https://www.tensorflow.org/xla), meaning it combines elegant autograd (`grad`), with fast just-in-time compiled code (`jit`) on CPU or GPU (or TPU). And all that in native python as a `numpy` drop-in replacement, making it very easy to get started.

Using an autograd library like JAX means you don't have to care about the gradients and the library figures them out for you on it's own. But there are some specific cases where it's beneficial to dig a bit deeper and do some of the legwork yourself. This blog post is about a very specific case in differentiable rendering, but I hope it is usefull for a broader audience since it shows how to derive the math and implement custom gradient functions in JAX (both in forward mode and backward mode).

## Ray Tracing with Signed Distance Fields

Ray tracing is a specific type of rendering, where rays are shot from the camera focal point into space, until they enounter objects. It's sort of the opposite of what is physically actually happening, because it's a lot easier to trace rays starting from the camera, then tracing rays starting at all the light sources and hoping enough of them eventually hit the camera.
Because you can model all kinds of rendering effects by letting the rays bounce of objects and reflect in different ways, it allows very realistic rendering.

At the core of ray tracing lies an algorithm to find the intersection point of a ray with an object in the scene.
One way to do that is using signed distance fields (SDF). For any point $x$ and any object in the scene, an SDF provides you with the signed distance (a positive value if $x$ is outside the object, a negative value if $x$ is inside the object).
The raymarching algorithm iteratively marches along the ray with a distance equal to the signed distance field of the scene. That way, the ray will never pass through an object.
You can read more about it in this other excellent [blog post](https://blog.evjang.com/2019/11/jaxpt.html).

This blog focuses on implementing a more efficient differentiable raymarching algorithm using the implicit function theorem.
By utilizing the implicit function theorem[^1], we can derive the gradient thourgh the raymarching algorithm without having to backprop through all the iterations.

### Differentiable Raymarching

To make the raymarching algorithm differentiable, we define a custom jacobian vector product (JVP) function in backward mode, or a vector jacobian product (VJP) function in forward mode.
Most deep learning research uses backward mode, because you typically take the gradient of a scalar loss with respect to lots of input parameters. Forward mode is more suited if you have only a few inputs and lots of outputs.
You can read more about the difference and how it's implemented in JAX [here](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).

One of the nice things about JAX is that you can define a custom VJP function, and JAX will automagickally transpose it into the corresponding JVP function, if the gradient is taken in backward mode.
<!-- For me, this was really usefull because I found the mathematical derivation a lot easier in forward mode. -->

We start with the implicit equation we derived above

\[z^{\mathrm{T}}(Mz+q) = 0\]

and take the differentials of all terms

\[\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}}\mathrm{d}(Mz+q) = 0\]

\[\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}}(\mathrm{d}M z + M\mathrm{d}z+\mathrm{d}q) = 0\]

Now we have an expression with six variables. $M$ and $q$ are given, $z$ is solved by the LCP solver, and in forward mode, $\mathrm{d}M$ and $\mathrm{d}q$ are also given. In fact, the only thing we need is $\mathrm{d}z$, so let's solve this equation for $\mathrm{d}z$:

\[\mathrm{d}z^{\mathrm{T}}(Mz+q) + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0\]

We reintroduce $w$ to make the equation simpler:

\[\mathrm{d}z^{\mathrm{T}}w + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0\]

Since $\mathrm{d}z^{\mathrm{T}}w$ is a dot product, we can commutate it:

\[w^{\mathrm{T}} \mathrm{d}z + z^{\mathrm{T}} M\mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0\]

And use the distributive property to isolate $\mathrm{d}z$:

\[(w^{\mathrm{T}} + z^{\mathrm{T}} M) \mathrm{d}z + z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q) = 0\]

Arriving at the solution for $$\mathrm{d}z$$:

\[\mathrm{d}z = -(w^{\mathrm{T}} + z^{\mathrm{T}} M)^{-1} z^{\mathrm{T}}(\mathrm{d}M z+\mathrm{d}q)\]

This is the implementation in JAX:

```python
import jax.numpy as np

def lcp_forward(M, q):
    # lcp solver
    # ...
    return z

def lcp_jvp(primals, tangents):
    M, q = primals
    dM, dq = tangents
    z = lcp_forward(M, q)
    w = np.matmul(M, z) + q
    dz = - np.linalg.pinv(np.diag(w) + z[:, np.newaxis] * M) @ (z * (dM @ z + dq))
    return z, dz

lcp_forward.defjvp(lcp_jvp)
```

where `lcp_forward()` is the LCP solver function. The full implementation is available [here](...). You can read more on how to define custom JVP's or VJP's [here]().

It's always a good idea to test your mathematical derivation and implementation with some numerical tests. JAX offers some utility functions which make this fairly easy. In the code below, `check_vjp` will compare our VJP implementation with numerical gradients (numerical differences) for some random $M$ and $q$. If the two are not equal (up to a small numerical error) `check_vjp` throws an error.

```python
from jax.test_util import check_jvp

n = 3
key = random.PRNGKey(42)
keys = random.split(key, 2)

M = random.normal(keys[0], (n, n))
q = random.normal(keys[1], (n,))

check_jvp(lcp_forward, lcp_jvp, (M, q))
```

As you can see in my full implementation, my tests are a bit more elaborate. I also test the solution of the LCP solver, and do this `check_vjp` on 100 different random $M$'s and $q$'s.

backward mode

benefits


**Conclusion**

Other work: talk from matthew, jax docs.


## References

[^1]: The NeurIPS 2020 tutorial [Deep Implicit Layers](http://implicit-layers-tutorial.org/) offers a great introduction on some applications of the implicit function theorem.