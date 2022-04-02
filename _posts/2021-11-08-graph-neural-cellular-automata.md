---
layout: post
categories: posts
title: "Graph Neural Cellular Automata"
image: /images/2021-11-08/thumbnail.png
tags: [GNN, cellular-automata]
date-string: NOVEMBER 8, 2021
---

![Graph Neural Cellular Automata for morphogenesis]({{ site.url }}/images/2021-11-08/fixed_target_animation.gif){: .centered}

[Cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton) (or CA for short) are a fascinating computational model. 
They consist of a lattice of stateful cells and a transition rule that updates the state of each cell as a function of its neighbourhood configuration. 
By applying this local rule synchronously over time, we see interesting dynamics emerge. 

For example, here is the transition table of [Rule 110](https://en.wikipedia.org/wiki/Rule_110) in a 1-dimensional binary CA:

![Rule 110, transition table]({{ site.url }}/images/2021-11-08/Rule110-rule.png){: .threeq-width}

And here is the corresponding evolution of the states starting from a random initialization (time goes downwards):

![Rule 110, evolution of the states]({{ site.url }}/images/2021-11-08/Rule110rand.png){: .quarter-width}

By changing the rule, we get different dynamics, some of which can be extremely interesting. One example of this is the 2-dimensional [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), with its complex patterns that replicate and move around the grid. 

![Gosper glider gun]({{ site.url }}/images/2021-11-08/glider_gun.gif){: .half-width}

We can also bring this idea of locality to the extreme, by keeping it as the only requirement and making everything else more complicated. 

For example, if we make the states continuous and change the size of the neighbourhood, we get [the mesmerizing Lenia CA](https://arxiv.org/abs/1812.05433) with its _insanely_ life-like creatures that move around smoothly, reproduce, and even organize themselves into higher-order organisms.

<div class='video-container'>
    <iframe src="https://www.youtube-nocookie.com/embed/iE46jKYcI4Y" frameborder="0" allowfullscreen></iframe>
</div>

By this principle, we can also derive an even more general version of CA, in which the neighbourhoods of the cells no longer have a fixed shape and size. Instead, the cells of the CA are organized in an arbitrary graph.

Note that the central idea of locality that characterizes CA does not change at all: we're just extending it to account for these more general neighbourhoods. 

The super-general CA are usually called **Graph Cellular Automata (GCA)**.

![Example of GCA transition]({{ site.url }}/images/2021-11-08/gca_transition.png){: .half-width}

The general form of GCA transition rules is a map from a cell and its neighbourhood to the next state, and we can also make it **anisotropic** by introducing edge attributes that specify a relation between the cell and each neighbour. 

## Learning CA rules

The world of CA is fascinating, but unfortunately, they are almost always considered simply pretty things. 

**But can they be also useful? Can we design a rule to solve an interesting problem using the decentralized computation of CA?**

The answer is yes, but manually designing such a rule may be hard. However, being AI scientists, we can try to learn the rule. 

This is not a new idea.

We can go back to NeurIPS 1992 to find a seminal work on [learning CA rules with neural networks](https://papers.nips.cc/paper/1992/hash/d6c651ddcd97183b2e40bc464231c962-Abstract.html) (they use convolutional neural networks, although back then they were called "sum-product networks with shared weights"). 

Since then, we've seen other approaches to learn CA rules, like these papers using [genetic algorithms](https://mobile.aau.at/~welmenre/papers/elmenreich-iwsos2011.pdf) or [compositional pattern-producing networks](https://ieeexplore.ieee.org/abstract/document/8004527) to find rules that lead to a desired configuration of states, a task called **morphogenesis**.<sup style="font-size: 10px;"> [Papers not on arXiv, sorry](https://sci-hub.se/)</sup>

More recently, convolutional networks have been shown to be extremely versatile in learning CA rules. 
[This work by William Gilpin](https://arxiv.org/abs/1809.02942), for example, shows that we can implement any desired transition rule with CNNs by smartly setting their weights. 

![A planarian flatworm]({{ site.url }}/images/2021-11-08/planarian.jpg){: .half-width}

CNNs have also been used for morphogenesis. Inspired by the regenerative abilities of the flatworm (pictured above), [in this visually-striking paper](https://distill.pub/2020/growing-ca/) they train a CNN to grow into a desired image and to regenerate the image if it is perturbed. 

## Learning GCA rules

So, can we do something similar in the more general setting of GCA? 

Well, let's start with the model. 
Similar to how CNNs are the natural family of models to implement typical grid-based CA rules, the more general family of graph neural networks is the natural choice for GCA. 

We call this setting the **Graph Neural Cellular Automata (GNCA)**. 

![Graph Neural Cellular Automata]({{ site.url }}/images/2021-11-08/thumbnail_cut.png){: .threeq-width}

We propose an architecture composed of a pre-preprocessing MLP, a message-passing layer, and a post-processing MLP, which we use as  transition function. 

This model is universal to represent GCA transition rules. We can prove this by making an argument similar to the one for CNNs that I mentioned above. 

I won't go into the specific details here, but in short, we need to implement two operations: 

1. One-hot encoding of the states;
2. Pattern-matching for the desired rule. 

The first two blocks in our GNCA are more than enough to achieve this. 
The pre-processing MLP can compute the one-hot encoding, and by using edge attributes and [edge-conditioned convolutions](https://arxiv.org/abs/1704.02901) we can implement pattern matching easily.

## Experiments

However, regardless of what the theory says, we want to know whether we can learn a rule in practice. Let's try a few experiments. 

### Voronoi GCA

![Voronoi GCA]({{ site.url }}/images/2021-11-08/voronoi.png){: .half-width}

We can start from the simplest possible binary GCA, inspired by the 1992 NeurIPS paper I mentioned before. The difference is that our CA cells are the [Voronoi tasselletion](https://en.wikipedia.org/wiki/Voronoi_diagram) of some random points. 
Alternatively, you can think of this GCA as being defined on the [Delaunay triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) of the points. 

We use an [outer-totalistic rule](https://en.wikipedia.org/wiki/Life-like_cellular_automaton.) that swaps the state of a cell if the density of its alive neighbours exceeds a certain threshold, not too different from the Game of Life. 

We try to see if our model can learn this kind of transition rule. In particular, we can train the model to approximate the 1-step dynamics in a supervised way, given that we know the true transition rule. 

<div style="text-align: center">
<img src="{{ site.url }}/images/2021-11-08/learn_gca_loss_v_epoch.svg" width="30%" style="display: inline-block; margin:auto;">&nbsp;
<img src="{{ site.url }}/images/2021-11-08/learn_gca_acc_v_epoch.svg" width="30%" style="display: inline-block; margin:auto;">
</div>

The results are encouraging. We see that the GNCA achieves 100% accuracy with no trouble and, if we let it evolve autonomously, it does not diverge from the real trajectory. 

### Boids
<div style="text-align: center">
<img src="{{ site.url }}/images/2021-11-08/alignment.png" width="30%" style="display: inline-block; margin:auto;">&nbsp;
<img src="{{ site.url }}/images/2021-11-08/cohesion.png" width="30%" style="display: inline-block; margin:auto;">&nbsp;
<img src="{{ site.url }}/images/2021-11-08/separation.png" width="30%" style="display: inline-block; margin:auto;">
</div>

For our second experiment, we keep a similar setting but make the target GCA much more complicated. 
We consider the [Boids](https://en.wikipedia.org/wiki/Boids) algorithm, an agent-based model designed to simulate the flocking of birds. This can be still seen as a kind of GCA because the state of each bird (its position and velocity) is updated only locally as a function of its closest neighbours.
However, this means that the states of the GCA are continuous and multi-dimensional, and also that the graph changes over time.

Again, we can train the GNCA on the 1-step dynamics. We see that, although it's hard to approximate the exact behaviour, we get very close to the true system. 
The GNCA (yellow) can form the same kind of flocks as the true system (purple), even if their trajectories diverge.

![Boids GCA and trained GNCA]({{ site.url }}/images/2021-11-08/boids_animation.gif){: .centered}

### Morphogenesis

The final experiment is also the most interesting, and the one where we actually design a rule. 
Like previously in the literature, here too we focus on morphogenesis. Our task is to find a GNCA rule that, starting from a given initial condition, converges to a desired point cloud (like a bunny) where the connectivity of the cells has a geometrical/spatial meaning.

In this case, we don't know the true rule, so we must train the model differently, by teaching it to arrive at the target state when evolving autonomously.

![Training scheme for GNCA]({{ site.url }}/images/2021-11-08/gnca_training.png){: .threeq-width}

To do so, we let the model evolve for a given number of steps, then we compute the loss from the target, and we update the weights with backpropagation through time. 
To stabilise training, and to ensure that the target state becomes a stable attractor of the GNCA, we use a cache. This is a kind of replay memory from which we sample the initial conditions, so that we can reuse the states explored by the GNCA during training.
Crucially, this teaches the model to remain at the target state when starting from the target state. 

And the results are pretty amazing... have you seen the gif at the [top of the post](#)? Let's unroll the first few frames here.

A 2-dimensional grid:

![]({{ site.url }}/images/2021-11-08/clouds/Grid_10-20/evolution.png)

A bunny:

![]({{ site.url }}/images/2021-11-08/clouds/Bunny_10-20/evolution.png)

The [PyGSP](https://pygsp.readthedocs.io/en/stable/) logo:

![]({{ site.url }}/images/2021-11-08/clouds/Logo_20/evolution.png)

We see that the GNCA has no trouble in finding a stable rule that converges quickly at the target and then remains there.

Even for complex and seemingly random graphs, like the Minnesota road network, the GNCA can learn a rule that quickly and stably converges to the target: 

![]({{ site.url }}/images/2021-11-08/clouds/Minnesota_20/anim.gif){: .third-width}

However, this is not the full story. Sometimes, instead of converging, the GNCA learns to remain in an orbit around the target state, giving us these oscillating point clouds. 

Grid:

![]({{ site.url }}/images/2021-11-08/clouds/Grid_10/evolution.png)

Bunny:

![]({{ site.url }}/images/2021-11-08/clouds/Bunny_10/evolution.png)

Logo:

![]({{ site.url }}/images/2021-11-08/clouds/Logo_10/evolution.png)

## Now what?

So, where do we go from here?

We have seen that GNCA can reach global coherence through local computation, which is not that different from what we do in graph representation learning. In fact, [the first GNN paper](https://www.researchgate.net/profile/Franco_Scarselli/publication/4202380_A_new_model_for_earning_in_raph_domains/links/0c9605188cd580504f000000.pdf), back in 2005, already contained this idea. 

But moving forward, it's easy to see that the idea of emergent computation on graphs could apply to many scenarios, including swarm optimization and control, modelling epidemiological transmission, and it could even improve our understanding of complex biological systems, like the brain.

GNCA enable the design of GCA transition rules, unlocking the power of decentralised and emergent computation to solve real-world problems.

The code for the paper is available [on Github](https://github.com/danielegrattarola/GNCA) and feel free to reach out via email if you have any questions or comments. 

## Read more

This blog post is the short version of our NeurIPS 2021 paper:

[Learning Graph Cellular Automata](https://arxiv.org/abs/2110.14237)  
_D. Grattarola, L. Livi, C. Alippi_

You can cite the paper as follows: 

```
@inproceedings{grattarola2021learning,
  title={Learning Graph Cellular Automata},
  author={Grattarola, Daniele and Livi, Lorenzo and Alippi, Cesare},
  booktitle={Neural Information Processing Systems},
  year={2021}
}
```
