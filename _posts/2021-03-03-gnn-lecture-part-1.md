---
layout: post
categories: posts
title: "A practical introduction to GNNs - Part 1"
image: /images/2021-03-03/presentation-1.png
tags: [GNN, lecture]
date-string: MARCH 3, 2021
---

_This is Part 1 of an introductory lecture on graph neural networks that I gave for the "Graph Deep Learning" course at the University of Lugano._

_At this point in the course, the students had already seen a high-level overview of GNNs and some of their applications. My goal was to give them a practical understanding of GNNs._

_Here I show that, starting from traditional CNNs and changing a few underlying assumptions, we can create a neural network that processes graphs._

[The full slide deck is available here]({{ site.url }}/files/talks/2021-03-01-USI_GDL_GNNs.pdf).

---

My goal for this lecture is to show you how Graph Neural Networks (GNNs) can be obtained as a generalization of traditional convolutional neural networks (CNNs), where instead of images we have graphs as input. 

**But what does it mean that a CNN can be made more general? Why are graphs a more general version of images?**

We know that CNNs are designed to process data that describe the world through a collection of discrete data points: time steps in a time series, pixels in an image, pixels in a video, etc.

However, one aspect of images and time series that we rarely (if at all) consider explicitly is the fact that the collection of data points alone is not enough. The order in which pixels are arranged to form an image is possibly more important than the pixels themselves. <br>
An image can be in color or in grayscale but, as long as the arrangement of pixels is the same, we'll likely be able to recognize the image for what it is.

We could go as far as saying that an image is only an image because its pixels are arranged in a particular structure: pixels that represent points close in space or time should also be next to each other in the collection. Change this structure, and the image loses meaning. 

<img src="{{ site.url }}/images/2021-03-03/presentation-4.svg" width="100%" style="border: solid 1px;"/>

CNNs are designed to take this __locality__ into account. They are designed to transform the value of each pixel, not as a function of the whole image (like a MLP would do), but as a function of the pixel's immediate surroundings. Its neighbors.

Since **locality is a kind of relation** between pixels, it is natural to represent the underlying structure of an image using a graph.
And, by requiring that each pixel is related only to the few other pixels that are closer to it, our graph will be a **regular grid**. Every pixel has 8 neighbors (give or take boundary conditions), and the CNN uses this fact to compute a localized transformation.

You can also interpret it the other way around. The kind of processing that the CNN does means that the transformation of each pixel will only depend on the few pixels that fall under the convolutional kernel. We can say that the grid structure emerges as a consequence of the CNN's inductive bias.

In any case, the important thing to note is that the grid structure does not depend on the specific pixel values. __We separate the values of the data points from the underlying structure that supports them.__

<img src="{{ site.url }}/images/2021-03-03/presentation-5.svg" width="100%" style="border: solid 1px;"/>

With this perspective in mind, the question of "how to make CNNs work on graphs" becomes: 

_Can we create a neural network in which the structure of the data is no longer a regular grid, but an arbitrary graph that we give as input?_

In other words, since we know that data and structure are different things, can we change the structure as we please?

The only thing that we require is that the CNN does the same kind of local processing as it did for the regular grid: transform each node as a function of its neighbors. 

<img src="{{ site.url }}/images/2021-03-03/presentation-6.svg" width="100%" style="border: solid 1px;"/>

If we look at what this request entails, we immediately see some problems: 

1. In the "regular grid" case, the learnable kernel of the CNN is compact and has a fixed size: one set of weights for each possible neighbor of a pixel, plus one set for the pixel itself. In other words, the kernel is supported by a smaller grid. 
We can't do that easily for an arbitrary graph. Since nodes can have a variable number of neighbors, we also need a kernel that varies in size. Possible, but not straightforward. 

2. In the regular grids processed by CNNs, we have an implicit notion of directionality. We always know where up, down, left and right are. When we move to an arbitrary graph, we might not be able to define a direction. Direction is, in essence, a kind of attribute that we assign to the edges, but in our case we also allow graphs that have no edge attributes at all. Ask yourself: do you have an up-and-to-the-left follower on Twitter?

To go from CNN to GNN we need to solve these problems.

<img src="{{ site.url }}/images/2021-03-03/presentation-7.svg" width="100%" style="border: solid 1px;"/> 

_[I recall notation here because the students had already seen most of these things anyway, but the concept of "reference operator" gave me a nice segue into the next slide.]_

All this talking about edge attributes also made me remember that now is a good time to do a notation check. Briefly:

- We define a graph as a collection of nodes and edges. 

- Nodes can have vector attributes, which we represent in a neatly packed matrix $$\mathbf{X} \in \mathbb{R}^{N \times F}$$ (sometimes called a _graph signal_).
Same thing for edges, with attributes $$\mathbf{e}_{ij} \in \mathbb{R}^S$$ for edge i-j.

Then there are the characteristic matrices of a graph:

- The adjacency matrix $$\mathbf{A}$$ is binary and has a 1 in position i-j if there exists an edge from node i to node j. All entries are 0 otherwise.

- The degree matrix $$\mathbf{D}$$ counts the number of neighbors of each node. It's a diagonal matrix so that the degree of node i is in position i-i.

- The Laplacian, which we will use a lot later, is defined as $$\mathbf{L} = \mathbf{D} - \mathbf{A}$$.

- Finally, the normalized adjacency matrix is $$\mathbf{A}_n = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$.

Note that $$\mathbf{A}$$, $$\mathbf{L}$$, and $$\mathbf{A}_n$$ __share the same sparsity pattern, if you don't count the diagonal__. Their only non-zero entries are in position i-j only if edge i-j exists. 

Since we're more interested in this specific property than in the actual values that are stored in the non-zero entries, let's give it a name: we call any matrix that has the same sparsity pattern of $$\mathbf{A}$$ a __reference operator__ (sometimes a _structure_ operator, sometimes a _graph shift_ operator, it's not important).

Also note: so far we are considering graphs with undirected edges. This means that all reference operators will be symmetric (if edge i-j exists, then edge j-i exists).

<img src="{{ site.url }}/images/2021-03-03/presentation-8.svg" width="100%" style="border: solid 1px;"/>

Reference operators are nice. 

First of all, they are operators. You multiply them by a graph signal and you get a new graph signal in return. Let's look at the "shape" of the multiplication: N-by-N times N-by-F equals N-by-F. Checks out.

But not only that. By their own definition, multiplying a reference operator by a graph signal will compute a weighted sum of each node's neighborhood. Let's expand the matrix multiplication from the slide above to see what happens to node 1 when we apply a reference operator.

All values $$\mathbf{r}_{ij}$$ that are not associated with an edge are 0, so we have:

$$
    (\mathbf{R}\mathbf{X})_1 = \mathbf{r}_{12}\cdot\mathbf{x}_2 + \mathbf{r}_{13}\cdot\mathbf{x}_3 + \mathbf{r}_{14}\cdot\mathbf{x}_4
$$


Look at that: with a simple matrix multiplication we can now do the same kind of local processing that the CNN does. 

Since applying a reference operator results in a simple sum-product, the result will not depend on the particular order in which we consider the nodes. As long as row $$i$$ of the reference operator describes the connections of the node with attributes $$\mathbf{x}_i$$, the result will be the same. We say that this kind of operation is **equivariant to permutations of the nodes**. <br>
This is good, because the particular order with which we consider the nodes is not important. Remember: we're only interested in the structure -- which nodes are connected to which.

Now that we are able to aggregate information from a node's neighborhood, we only need to solve the issue of how to create the learnable kernel and we will have a good first approximation of a CNN for graphs. Remember the two issues that we have: 

1. Neighborhoods vary in size;
2. We don't know how to orient the kernel (i.e., we may not have attributes that allow us to distinguish a node's neighbors);

These problems are also related to our request that the GNN must be equivariant to permutations. We cannot simply assign a different weight to each neighbor because we would need to train the GNN on all possible permutations of the nodes in order to make it equivariant.

However, there is a simple solution: __use the same set of weights for each node in the neighborhood.__<br>
Let our weights be a matrix $$\mathbf{\Theta} \in \mathbb{R}^{F \times F'}$$, so that the output will have $$F'$$ "feature maps". 

Now, we simply use $$\mathbf{\Theta}$$ to transform the node attributes, then sum them over using a reference operator. 

Let's check the shapes to make sure that it works out: N-by-N times N-by-F times F-by-F' equals N-by-F'.<br>
We went from graph signal to graph signal, with new node attributes that we obtain as a local, learnable, and differentiable transformation.

Done! We have our first GNN: $$\mathbf{X}' = \mathbf{R} \mathbf{X} \mathbf{\Theta}$$.

<img src="{{ site.url }}/images/2021-03-03/presentation-9.svg" width="100%" style="border: solid 1px;"/>

One thing that is still missing from our relatively simple implementation is the ability to have kernels that span more than the immediate neighborhood of a node. In fact, in a CNN this is usually a hyperparameter. Also, depending on the reference operator that we use, we may or may not consider a node itself when computing its transformation: it depends on whether $$\mathbf{R}$$ has a non-zero diagonal.

Luckily we can generalize the idea of a bigger kernel to the graph domain: we simply process each node as a function of its neighbors up to $$K$$ steps away from it.

We can achieve this by considering that applying a reference operator to a graph signal has the effect of making node attributes _flow_ through the graph. 
Apply a reference operator once, and all nodes will "read" from their immediate neighbors to update themselves. Apply it again, and all nodes will read again from their neighbors, except that this time the information that they read will be whatever the neighbors computed at the previous step.

In other words: if we multiply a graph signal by $$\mathbf{R}^{K}$$, each node will update itself with the node attributes of nodes $$K$$ steps away.

<img src="{{ site.url }}/images/2021-03-03/presentation-10.svg" width="100%" style="border: solid 1px;"/>

In a CNN, this would be equivalent to having a kernel shaped like an empty square.
To make the kernel full, we simply sum all "empty square" kernels up to the desired size. In our case, instead of considering $$\mathbf{R}^{K}$$, we consider a polynomial of $$\mathbf{R}$$ up to order $$K$$.

This is called a **polynomial graph filter**, and we will see a different interpretation of it in Part 3 of this series.

Note that this filter solves both problems that we had before, and also makes our GNN more expressive: 

1. The value of a node itself is always included in the transformation, since $$\mathbf{R}^{0} = \mathbf{I}$$;
2. The sum of polynomials up to order $$K$$ will necessarily cover all neighbors in a radius of $$K$$ steps;
3. Since we can treat neighborhoods separately, we can also have different weights $$\mathbf{\Theta}^{(k)}$$ for each $$k$$-hop neighborhood. This is like having a radial filter, a function that only depends on the radius from the origin.


<img src="{{ site.url }}/images/2021-03-03/presentation-11.svg" width="100%" style="border: solid 1px;"/>

This idea of using a polynomial filter to create a GNN was first introduced in a paper by [Defferrard et al.](https://arxiv.org/abs/1606.09375), which can be seen as the first scalable and practical implementation of a GNN ever proposed.

In that paper they used a particular choice of polynomial, namely one for which different powers are defined in a recursive manner, called a **Chebyshev polynomial**.

In particular, as reference operator they use a version of the graph Laplacian that is first normalized and then rescaled so that its eigenvalues are between -1 and 1.
Then, using the recursive formulation of Chebyshev polynomials, they build a polynomial graph filter. 

The reason why they use these polynomials and not the simple ones we saw above is not important, for now. Let us just say: they have some desirable properties and they are fast to compute. 

<img src="{{ site.url }}/images/2021-03-03/presentation-12.svg" width="100%" style="border: solid 1px;"/>

Just a few months after the paper by Defferrard et al. was published on ArXiv, a new paper by [Kipf & Welling]() also appeared online.

In that paper, the authors looked at the Chebyshev filter proposed by Defferrard et al. and introduced a few key changes to make the layer more simple and more scalable. 

1. They changed the reference operator. Instead of the rescaled and normalized Laplacian, they assumed that $$\lambda_{max} = 2$$ so that the whole formulation of the operator was simplified to $$-\mathbf{A}_n$$.
2. They proposed to use polynomials of order 1, following the intuition that $$K$$ layers of order 1 would be equivalent to 1 layer of order $$K$$. In particular, they also added non-linearities between each successive layer, leading to more complex transformations of the nodes at each propagation step.
3. They observed that the same set of weights could be used both for a node itself and its neighbors. No need to have $$\mathbf{\Theta}^{(0)}$$ and $$\mathbf{\Theta}^{(1)}$$ as different weights.
4. After simplifying the layer down to 
$$
    \mathbf{X}' = ( \mathbf{I} + \mathbf{A}_n) \mathbf{X} \mathbf{\Theta},
$$
they observed that a more stable behavior could be obtained by instead using $$\mathbf{R} = \mathbf{D}^{-1/2} (\mathbf{I} + \mathbf{A}) \mathbf{D}^{-1/2}$$ as reference operator.

Putting this all together, we get to what is commonly known as the Graph Convolutional Network (GCN): 

$$
\mathbf{X}' = \mathbf{D}^{-1/2} (\mathbf{I} + \mathbf{A}) \mathbf{D}^{-1/2} \mathbf{X} \mathbf{\Theta}$$

---

What we have seen so far is a very simple construction that takes the general concepts behind CNNs and, by changing a few assumptions, extends them to the case in which the input is an arbitrary graph instead of a grid. 

This is far from the whole story, but it should give you a good starting point to learn about GNNs.

In the [next part of this series]({{ site.url }}/posts/2021-03-12/gnn-lecture-part-2.html) we will see: 

1. How to describe what we just saw as a general algorithm that allows us to describe a much richer family of operations on graphs.
2. How to throw edge attributes in the mix and create GNNs that can treat neighbors differently.
3. How to make the entries of a reference operator a learnable function.
4. A general recipe for a GNN that *should* work well for many problems.

Stay tuned.
