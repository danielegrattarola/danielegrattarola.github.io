---
layout: post
categories: posts
title: "A practical introduction to GNNs"
image: /images/
tags: [GNN, lecture]
date-string: MARCH 1, 2021
---

This is __part 1__ of an introductory lecture on GNNs that I gave for the "Graph Deep Learning" course at USI. 

[The full slide deck is available here]({{ site.url }}/files/talks/2021-03-01-USI_GDL_GNNs.pdf).

---

We often say that Graph Neural Networks (GNNs) are a generalization of traditional convolutional neural networks (CNNs), where we have graphs instead of images. 

But what does it mean to make a CNN more general? **Why are graphs a more general version of images?**<br>
To answer these questions, it's useful to start from CNNs and look at the kind of data that they process. 


We know that CNNs are designed to process data that describe the world through a collection of discrete data points: time steps in a time series, pixels in an image, pixels in a video, etc.

However, one aspect of images and time series that we rarely (if at all) consider explicitly is the fact that a collection of data points alone is not enough. The order in which pixels are arranged to form an image is possibly more important than the pixels themselves. <br>
An image can be in color or in grayscale but, as long as the arrangement of pixels is the same, we'll likely be able to recognize the image for what it is.

We could go as far as saying that an image is only an image because its pixels are arranged in a particular structure: pixels that represent points close in space or time should also be next to each other in the collection. Change this structure, and the image loses meaning. 

<img src="{{ site.url }}/images/2021-03-01/presentation-4.svg" width="100%" style="border: solid 1px;"/>

CNNs are designed to take this __locality__ into account. They are designed to transform the value of each pixel, not as a function of the whole image (like a MLP would do), but as a function of the pixel's immediate surroundings. Its neighbours.

Since **locality is a kind of relation** between pixels, it is natural to represent the underlying structure of an image using a graph.
And, by requiring that each pixel is related only to the few other pixels that are closer to it, our graph will be a **regular grid**. Every pixel has 8 neighbours (give or take boundary conditions), and the CNN uses this fact to compute a localized transformation.

You can also interpret it the other way around. The kind of processing that the CNN does means that the transformation of each pixel will only depend on the few pixels that fall under the convolutional kernel. We can say that the grid structure emerges as a consequence of the CNN's inductive bias.

In any case, the important thing to note is that the grid structure does not depend on the specific pixel values. __We separate the values of the data points from the underlying structure that supports them.__

<img src="{{ site.url }}/images/2021-03-01/presentation-5.svg" width="100%" style="border: solid 1px;"/>

With this perspective in mind, the question of "how to make CNNs work on graphs" becomes: 

_Can we create a neural network in which the structure of the data is no longer a regular grid, but an arbitrary graph that we give as input?_

In other words, since we know that data and structure are different things, can we change the structure as we please?

The only thing that we require is that the CNN does the same kind of local processing as it did for the regular grid: transform each node as a function of its neighbours. 

<img src="{{ site.url }}/images/2021-03-01/presentation-6.svg" width="100%" style="border: solid 1px;"/>

If we look at what this request entails, we immediately see some problems: 

1. In the "regular grid" case, the learnable kernel of the CNN is compact and has a fixed size: one set of weights for each possible neighbor of a pixel (plus one set for the pixel itself). In other words, the kernel is supported by a smaller grid. 
We can't do that easily for an arbitrary graph. Since nodes can have a variable number of neighbors, we also need a kernel that varies in size. Possible, but not straightforward. 

2. In the regular grids processed by CNNs, we have a notion of directionality which is implicit to the grid. We always know where is up, down, left and right. When we move to an arbitrary graph, we might not be able to define a direction. Direction is, in essence, a kind of attribute that we assign to the edges, but in our case we also allow graphs that have no edge attributes at all. Ask yourself: do you have an up-and-to-the-left follower on Twitter?

To go from CNN to GNN we need to solve these problems.

<img src="{{ site.url }}/images/2021-03-01/presentation-7.svg" width="100%" style="border: solid 1px;"/> 

All this talking about edge attributes also made me remember that now is a good time to do a notation check. Briefly:

- We define a graph as a collection of nodes and edges. 

- Nodes can have vector attributes, which we represent in a neatly packed matrix $$\mathbf{X} \in \mathbb{R}^{N \times F}$$ (sometimes called a _graph signal_).
Same thing for edges, with attributes $$\mathbf{e}_{ij} \in \mathbb{R}^S$$ for edge i-j.

Then there are the characteristic matrices of a graph:

- The adjacency matrix is binary and has a 1 in postion i-j if there exists an edge from node i to node j. The entry is 0 otherwise.

- The degree matrix counts the number of neighbors of each node. It's a diagonal matrix so that the degree of node i is in position i-i.

- The Laplacian, which we will use a lot later, is defined as $$\mathbf{L} = \mathbf{D} - \mathbf{A}$$.

- Finally, all matrices can be normalized by pre and post multiplying them by $$\mathbf{D}^{-1/2}$$. For instance the normalized adjacency matrix is $$\mathbf{A}_n = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$

These characteristic matrices (except the degree) share the same sparsity pattern, if you don't count the diagonal. Their only non-zero entries are in position i-j only if edge i-j exists. 

Since we're more interested in this specific property than in the actual values that are stored in the non-zero entries, let's give it a name: we call any matrix that has the same sparsity pattern of $$\mathbf{A}$$ a __reference operator__ (sometimes a _structure_ operator, sometimes a _graph shift_ operator, it's not important).

Also note: so far we are considering graphs with undirected edges. This means that all reference operators will be symmetric (if edge i-j exists, then edge j-i exists).

<img src="{{ site.url }}/images/2021-03-01/presentation-8.svg" width="100%" style="border: solid 1px;"/>

Reference operators are nice. 

First of all, they are operators. You multiply them by a graph signal and you get a new graph signal in return: N-by-N times N-by-F equals N-by-F. Checks out.

But not only that. By their own definition, multiplying a reference operator by a graph signal will compute a weighted sum of each node's neighborhood. Let's expand the matrix multiplication from slide above: 

$$
    (\mathbf{R}\mathbf{X})_1 = \mathbf{r}_{12}\cdot\mathbf{x}_2 + \mathbf{r}_{13}\cdot\mathbf{x}_3 + \mathbf{r}_{14}\cdot\mathbf{x}_4
$$

All values $$\mathbf{r}_{ij}$$ that are not associated with an edge are 0. 

And look at that: with a simple matrix multiplication we can now do the same kind of local processing that the CNN does. 

You may say "I could have written that as a for loop over the indices!" And you would be right. We are instead writing it as a matrix multiplication... which is implemented as a for loop over indices anyway. 

Now that we are able to sum over a node's neighborhood, we only need to solve the issue of how to create the learnable kernel and we will have a good first approximation of a CNN for graphs. Remember the two issues that we have: 

1. Neighborhoods vary in size;
2. We don't know how to orient the kernel;

A simple solution to both: __use the same set of weights for each node in the neighborhood.__

Call them $$\mathbf{\Theta} \in \mathbb{R}^{F \times F'}$$, so that the output will have $$F'$$ "feature maps". Use the weights to transform the node attributes, then sum them over using a reference operator. 

Let's check the shapes to make sure that it works out: N-by-N times N-by-F times F-by-F' equals N-by-F'. 

We went from graph signal to graph signal, with new node attributes that we obtain as a local, learnable, and differentiable transformation.

Done! We have our first GNN: $$\mathbf{X}' = \mathbf{R} \mathbf{X} \mathbf{\Theta}$$.

<img src="{{ site.url }}/images/2021-03-01/presentation-9.svg" width="100%" style="border: solid 1px;"/>

One thing that is still missing from our relatively simple implementation is the ability to have kernels that span more than the immediate neighborhood of a node. In fact, in a CNN this is usually a hyperparameter. Also, depending on the reference operator that we use, we may or may not condsider a node itself when computing its transformation: it depends on whether $$\mathbf{R}$ has a non-zero diagonal.

Luckily we can generalize the idea of a bigger kernel to the graph domain. We simply process each node as a function of its neighborhs up to $$K$$ steps away from it.

We can achieve this by considering that applying a reference operator to a graph signal has the effect of making node attributes _flow_ through the graph. 
Apply a reference operator once, and all nodes will "read" from their immediate neighbors to update themselves. Apply it again, and all nodes will read again from their neighbours, except that this time the information that they read will be whatever the neighbours computed at the previous step.

In other words: if we multiply a graph signal by $$\mathbf{R}^{K} = \mathbf{R} \cdot \dots \cdot \mathbf{R}$$, each node will update itself with the node attributes of nodes $$K$$ steps away.

<img src="{{ site.url }}/images/2021-03-01/presentation-10.svg" width="100%" style="border: solid 1px;"/>

In the CNN parallel, this is equivalent to having a kernel shaped like an empty square.
To make the kernel "full", we simply sum all "empty square" kernels up to the desired size. In our case, instead of considering $$\mathbf{R}^{K}$$, we consider a polynomial of $$\mathbf{R}$$ up to order $$K$$.

This is called a **polynomial graph filter**, and we will see a different interpretation of it in Part 3 of this series.

Note that this filter solves both problems that we had before, and also makes our GNN more expressive: 

1. The value of a node itself is always included in the transformation, since $$\mathbf{R}^{0} = \mathbf{I}$$;
2. The sum of polynomials up to order $$K$$ will necessarily cover all neighbours in a radius of $$K$$ steps;
3. Since we can treat neighborhoods separately, we can also have different weights $$\mathbf{\Theta}^{k}$$ for each $$k$$-hop neighborhood. This is like having a radial filter, a function that only depends on the radius from the origin.


<img src="{{ site.url }}/images/2021-03-01/presentation-11.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-12.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-13.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-14.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-15.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-16.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-17.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-18.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-19.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-20.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-21.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-22.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-23.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-24.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-25.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-26.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-27.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-28.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-29.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-30.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-31.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-32.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-33.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-34.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-01/presentation-35.svg" width="100%" style="border: solid 1px;"/>



