---
layout: post
categories: posts
title: "A practical introduction to GNNs - Part 2"
image: /images/2021-03-03/presentation-1.png
tags: [GNN, lecture]
date-string: MARCH 12, 2021
---

_This is Part 2 of an introductory lecture on graph neural networks that I gave for the "Graph Deep Learning" course at the University of Lugano._

_After a practical introduction to GNNs in [Part 1]({{site.url}}/posts/2021-03-03/gnn-lecture-part-1.html), here I show how we can formulate GNNs in a much more flexible way using the idea of message passing._

_First, I introduce message passing. Then, I show how to implement message-passing networks in Jax/pseudocode using a paradigm called "gather-scatter". Finally, I show how to implement a couple of more advanced GNN models._

[The full slide deck is available here]({{ site.url }}/files/talks/2021-03-01-USI_GDL_GNNs.pdf).

---

In [Part 1]({{ site.url}}/posts/2021-03-03/gnn-lecture-part-1.html) of this series we constructed our first kind of GNN by replicating the behavior of conventional CNNs on data supported by graphs. 

The core building block that we used in our simple GNNs looked like this: 

$$
    \mathbf{X}' = \mathbf{R}\mathbf{X}\mathbf{\Theta}
$$

which, as we saw, has two effects: 

1. All node attributes $$\mathbf{X}$$ are transformed using the learnable matrix $$\mathbf{\Theta}$$;
2. The attribute of each node gets replaced with a weighted sum of its neighbors via the reference operator $$\mathbf{R}$$ (also, sometimes we can include the node itself in the sum);

By combining these two ideas we were able to get a very good approximation of a CNN for graphs. 

In this part of the lecture, we will take these two ideas and describe them a little more formally, distilling the essential role that they have in a GNN. 

We will see a general framework called **message passing**, which will allow us to describe more complex GNNs than those we have seen so far.

## Message Passing Networks

<img src="{{ site.url }}/images/2021-03-03/presentation-14.svg" width="100%" style="border: solid 1px;"/>

The idea of message passing networks was introduced in a paper by [Gilmer et al.]() in 2017 and it essentially boils GNN layers down to three main steps:

1. Every node in the graph computes a __message__ for each of its neighbors. Messages are a function of the node, the neighbor, and the edge between them.
2. Messages are sent, and every node __aggregates__ the messages it receives, using a permutation-invariant function (i.e., it doesn't matter in which order the messages are received). This function is usually a sum or an average, but it can be anything.
3. After receiving the messages, each node __updates__ its attributes as a function of its current attributes and the aggregated messages.

This procedure happens synchronously for all nodes in the graph, so that at each message passing step all nodes are updated. 

If we look back at our super-simple GNN formulation $$\mathbf{X}' = \mathbf{R}\mathbf{X}\mathbf{\Theta}$$, we can easily see the three message-passing steps: 

1. __Message__ - Each node $$i$$ will receive the same kind of message $$\mathbf{\Theta}^\top\mathbf{x}_j$$ from all its neighbors $$j \in \mathcal{N}(i)$$.
2. __Aggregate__ - Messages are aggregated with a weighted sum, where weights are defined by the reference operator $$\mathbf{R}$$. 
3. __Update__ - Each node simply replaces its attributes with the aggregated messages. <br>
If $$\mathbf{R}$$ has a non-zero diagonal, then each node also computes a message "from itself to itself" using $$\mathbf{\Theta}$$.


<img src="{{ site.url }}/images/2021-03-03/presentation-15.svg" width="100%" style="border: solid 1px;"/>

Message passing is usually formalized with the equation in the slide above. 

While it may look complicated at first, the formula simply describes the three steps that we saw before, and if you wanted to write it in pseudo-Python it would look something like this: 

```py
# For every node in the graph
for i in range(n_nodes):

    # Compute messages from neighbors
    messages = [message(x[i], x[j], e[i, j]) for j in neighbors(i)]

    # Aggregate messages
    aggregated = aggregate(messages)

    # Update node attributes
    x[i] = update(x[i], aggregated)
```

As long as `message`, `aggregate`, and `update` are differentiable functions, we can train any neural network to transforms its inputs like this. <br>
In fact, this framework is so general that virtually all libraries that implement GNNs are based on it. 

For example, [Spektral](https://graphneural.network), [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/), and [DGL](https://www.dgl.ai/) all have a `MessagePassing` class which looks like this: 

```py
class MessagePassing(Layer): # Or `Module`

    def call(self, inputs, **kwargs):  # Or `forward`
        # This is the actual message-passing step
        return self.propagate(*inputs)

    def propagate(self, x, a, e, **kwargs):
        # process arguments and create *_kwargs
        ...

        # Message
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        aggregated = self.aggregate(messages, **agg_kwargs)

        # Update
        output = self.update(aggregated, **upd_kwargs)

        return output

    def message(self, x, **kwargs):
        ...

    def aggregate(self, messages, **kwargs):
        ...

    def update(self, aggregated, **kwargs):
        ...
```

## Gather-Scatter

The cool thing about message passing is that it lets us define the operations that our GNN computes, without necessarily resorting to matrix multiplication. 

In fact, the only thing that we specify is how the GNN acts on a generic node $$i$$ as a function of its generic neighbors $$j \in \mathcal{N}(i)$$.

For instance, let's say that we wanted to implement the "Edge Convolution" operator from the paper ["Dynamic Graph CNN for Learning on Point Clouds"](https://arxiv.org/abs/1801.07829).

In the message-passing framework, we write its effect as:

$$
    \mathbf{x}_i' = \sum\limits_{j \in \mathcal{N}(i)} \textrm{MLP}\big( \mathbf{x}_i \| \mathbf{x}_j - \mathbf{x}_i \big)
$$


If we wanted to implement this as a simple matrix multiplication we would have some troubles, because GNNs of the form $$\mathbf{R}\mathbf{X}\mathbf{\Theta}$$ assume that every node sends the same message to each of its neighbors. Here, instead, messages are a function of edges $$j \rightarrow i$$. 

In fact, this is a limitation of every GNN with edge-dependent messages.

We could still implement our Edge Convolution using broadcasting operations, but it would not be efficient at all. Here's one way we could do it: 

```py
import jax, jax.numpy as jnp

x = ...  # Node attributes of shape [n, f]
a = ...  # Adjacency matrix of shape [n, n]

# Compute all pairwise differences between nodes
x_diff = x[None, :, :] - x[:, None, :]  # shape: (n, n, f)

# Repeat the nodes so that we can concatenate them to the differences
x_repeat = jnp.repeat(x[:, None, :], n, axis=1)  # shape: (n, n, f)

# Concatenate the attributes so that, for each edge, we have x_i || (x_i - x_j)
x_all = jnp.concatenate([x_repeat, x_diff], axis=-1)  # shape: (n, n, 2 * f)

# Give x_i || (x_i - x_j) as input to an MLP
messages = mlp(x_all)  # shape: (n, n, channels)

# Broadcast-multiply `a` to keep only "real" messages
output = a[..., None] * messages  # shape: (n, n, channels)

# Sum along the "neighbors" axis.
output = output.sum(1)  # shape: (n, channels)
```

__This is not ideal__, because it cost us $$O(N^2)$$ to do something that should have a cost linear in the number of edges (which is a big difference when working with real-world graphs, which are usually very sparse). 

In general, using broadcasting to define edge-dependent GNNs means that we have to compute the messages for __all possible edges__ and then simply multiply some of the messages by zero by broadcasting `a`.

This is because broadcasting is a "dense" operation.

A much better way to achieve our goal is to exploit the advanced indexing features offered by all libraries for tensor manipulation, using a technique called __gather-scatter__.

The gather-scatter technique requires us to think a bit differently, using node indices to access __only the nodes that we are interested in__, in a sparse way. 

This is much easier done than said, so let's see an example.

Let us consider an adjacency matrix `a`: 

```py
a = [[1, 0, 1],
     [0, 0, 1],
     [1, 1, 0]]
```

This matrix is equivalently represented in the sparse COOrdinate format: 

```py
row = [0, 0, 1, 2, 2]  # Nodes that are sending a message
col = [0, 2, 2, 0, 1]  # Nodes that are receiving a message
```

which simply tells us the indices of the non-zero entries of `a` (we usually also have an extra array that tells us the actual values of the entries, but we won't need it for now).

It's easy to see, now, that if we look at all edges of the form $$j \rightarrow i$$, then the attributes of all nodes that are sending a message can be retrieved with `x[row]`. 
Similarly, the attributes of nodes that are receiving a message can be retrieved with `x[col]`.

This is called __gathering__ the nodes.

In our case, if we want to take the difference of the nodes at the opposite side of an edge, we can simply do `x[row] - x[col]`. 
Instead of computing the difference `x[j] - x[i]` for all possible pairs `j, i`, like we did before, now we only compute the differences that we are really interested in. 

All these operations will give us matrices that have as many rows as there are edges. So for instance, `x[row]` will look like this: 

```py
[x[0],
 x[0],
 x[1],
 x[2],
 x[2]]  # shape: (n_edges, f)
```

The other half of this story tells us how to aggregate the messages after we have gathered them. We call this __scattering__.

For all nodes $$i$$, we want to aggregate all messages that are being sent via edges that have index $$i$$ on the __receiving__ end, i.e., all edges of the form $$j \rightarrow i$$.
For instance, in the small example above we know that node 2 will receive a message from nodes 0 and 1.

We can do this using some special operations available more or less in all libraries for tensor manipulation:

- In TensorFlow, we have `tf.math.segment_[sum|prod|mean|max|min]`.
- For PyTorch, we have the [Torch Scatter](https://github.com/rusty1s/pytorch_scatter) library by ‪Matthias Fey.
- In Jax, we only have `jax.ops.segment_sum`.

These operations apply a reduction to "segments" of a tensor, where the segments are defined by integer indices. Something like this: 

```py
# Example: segment sum
data = [5, 1, 7, 2, 3, 4, 1, 3]      # A tensor that we want to reduce
segments = [0, 0, 0, 1, 2, 2, 3, 3]  # Segment indices (we have 4 segments)

output = [0] * (max(segments) + 1)   # One result for each segment
for i, s in enumerate(segments):
    output[s] += data[i]             # It could be a product, max, etc...

>>> output 
[13, 2, 7, 4]
```

So for instance, if we want to sum all messages based on their intended recipient, we can do: 

```py
# recipients = col
aggregated = jax.ops.segment_sum(messages, recipients)
```

Now we can put all of this together to create our Edge Convolution layer with a gather-scatter implementation: 

```py
x = ...  # Node attributes of shape [n, f]
a = ...  # Adjacency matrix of shape [n, n]

# Get indices of the non-zero entries of the adjacency matrix
import scipy
senders, recipients, _ = scipy.sparse.find(a)

# Calculate difference of nodes for each edge j -> i
x_diff = x[senders] - x[recipients]  # shape: (n_edges, f)

# Concatenate x_i with (x_i - x_j) for each edge j -> i
x_all = jnp.concatenate([x[recipients], x_diff], axis=-1)  # shape: (n_edges, 2 * f)

# Give x_i || (x_i - x_j) as input to an MLP
messages = mlp(x_all)  # shape: (n_edges, channels)

# Aggregate all messages according to their intended recipient
output = jax.ops.segment_sum(messages, recipients)  # shape: (n, channels)
```

Wrap this up in a layer and we're done!

Here's what it looks like [in Spektral](https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/edge_conv.py) and [in Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/edge_conv.html#EdgeConv).

## Methods

Since now we've moved past the simple models based on reference operators and edge-independent messages that we saw in the first part of this series, we can look at some more advanced methods.

<img src="{{ site.url }}/images/2021-03-03/presentation-17.svg" width="100%" style="border: solid 1px;"/>

For instance, the popular [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Veličković et al. can be implemented as a message-passing network using gather-scatter: 

```py
# Transform node attributes with a dense layer (defined elsewhere)
h = dense(x)

# Concatenate attributes of recipients/senders
h_cat = jnp.concatenate([h[recipients], h[senders]], axis=-1)

# Compute attention logits w/ a dense layer (single output, LeakyReLU)
logits = dense(h_cat)

# Apply softmax only to the logits in the same segment, as defined by recipients
# i.e., normalize the scores only among the neighbors of each node.
#
# Note that segment_softmax does **not** reduce the tensor: `coef` has the same 
# size as `logits`.
#
# This function is available in Spektral and PyG.
coef = segment_softmax(logits, recipients)

# Now we aggregate with a weighted sum (weights given by coef)
output = jax.ops.segment_sum(coef * h[senders], recipients)
```

<img src="{{ site.url }}/images/2021-03-03/presentation-18.svg" width="100%" style="border: solid 1px;"/>

And, easily enough, we can also define a message-passing network that includes edge attributes in the computation of messages. One of my favorite models is the [Edge-Conditioned Convolution](https://arxiv.org/abs/1704.02901) by Simonovsky & Komodakis, of which I've summarized the math in the slide above. 

To implement it with gather-scatter we can do: 

```py
# Use a Filter-Generating Network to create the weights (defined elsewhere)
kernel = fgn(e)

# Reshape the weights so that we have a matrix of shape (f, f_) for each edge
kernel = jnp.reshape(kernel, (-1, f, f_))

# Multiply the node attribute of each neighbor by the associated edge-dependent
# kernel. 
# We can use einsum to do this efficiently
messages = jnp.einsum("ab,abc->ac", x[senders], kernel)

# Aggergate with a sum
output = jax.ops.segment_sum(messages, recipients)
```

Once you get the hang of it, building GNNs becomes so intuitive that you'll never want to go back to the matrix-multiplication-based implementations. 
Although, sometimes, it makes sense to do it. But that's a story for another day. 

---

With the first two parts of this blog series in your toolbelt, you should be able to go a long way in the world of GNNs. 

The next and final part will take a more historical and mathematical journey in the world of GNNs. We'll cover spectral graph theory and how we can define the operation of __convolution__ on graphs. 

I have left this for last because it is not _essential_ to understand and use GNNs in practice, although I think that understanding the historical perspective that led to the creation of modern GNNs is very important. 

Stay tuned.  
