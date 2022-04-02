---
layout: post
categories: posts
title: "Detecting Hostility from Skeletal Graphs Using Non-Euclidean Embeddings"
image: /images/header/publications.png
tags: [AI, experiment, non-euclidean]
date-string: APRIL 13, 2019
---

The first paper on which I worked during my PhD is about [detecting changes in sequences of graphs using non-Euclidean geometry and adversarial autoencoders](https://arxiv.org/abs/1805.06299). As a real-world application of the method presented in the paper, we showed that we could detect epileptic seizures in the brain, by monitoring a stream of functional connectivity brain networks.

In general, the methodology presented in the paper can work for any data that:

1. can be represented as graphs;
2. has a temporal dimension;
3. has a change that you want to identify somewhere along the stream of data;
4. has i.i.d. samples.

There are [a lot](https://icon.colorado.edu/#!/networks) of temporal networks that can be found in the wild, but not many datasets respect all the requirements at the same time. What's more, many public datasets have very little samples along the temporal axis.  <!--more-->
Recently, however, I was looking for some nice graph classification dataset on which to test [Spektral](https://danielegrattarola.github.io/spektral), and I stumbled upon the [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset released by the Nanyang Technological University of Singapore.  
The dataset consists of about 60 thousand video clips of people performing everyday actions, including mutual actions and some health-related ones. The reason why I found this dataset is that it contains skeletal annotations for each frame of each video clip, meaning lots and lots of graphs that [can be used for graph classification](https://arxiv.org/abs/1801.07455). 

## NTU RGB+D for change detection

![graphs]({{ site.url }}/images/2019-04-13/graphs.svg "Figure 1: examples of hugging and punching graphs."){: .threeq-width}

While reading through the website, however, I realized that this dataset could actually be a good playground for our change detection methodology as well, because it respects almost all requirements:

1. it has graphs;
2. it has a temporal dimension;
3. it has classes, which can be easily converted to what we called the _regimes_ of our graph streams;

The fourth requirement of having i.i.d. samples is due to the nature of the change detection test that we adopted in the paper. The test is able to detect changes in stationarity of a stochastic process, which means that it can tell whether the samples coming from the process have been drawn from a different distribution than the one observed during training.   
In order to do so, the test needs to estimate whether a window of observations from the process is significantly different than what observed in the nominal regime. This requires having i.i.d. samples in each window.  

By their very nature, however, the graphs in NTU RGB+D are definitely not i.i.d. (they would have been, had the subjects been recorded under a strobe light -- dammit!).  
There are several ways of converting a heavily autocorrelated signal to a stationary one, with the simplest one being randomizing along the time axis.
The piece-wise stationarity requirement is a very strong one, and we are looking into relaxing it, but for testing the method on NTU RGB+D we had to stick with it.

## Setting

Defining the change detection problem is easy: have a nominal regime of neutral or positive actions like walking, reading, taking a selfie, or being at the computer, and try to detect when the regime changes to a negative action like falling down, getting in fights with people, or feeling sick (there are at least 5 action classes of people acting hurt or sick in NTU RGB+D).

Applications of this could include: 

- monitoring children and elderly people when they are alone;
- detecting violence in at-risk, crowded situations;
- detecting when a driver is distracted;

In all of these situations, you might have a pretty good idea of what you _want_ to be happening at a given time, but have no way of knowing how things could go wrong. 

We chose the "hugging" action for the nominal, all-is-well regime, and we took the "punching/slapping" class to symbolize any unexpected, undesirable behaviour that deviates from our concept of nominal.
Then, we trained our adversarial autoencoder to represent points on an ensemble of constant-curvature manifolds, and we ran the change detection test. 
At this point, it would probably help if one was familiar with the details of [the paper](https://arxiv.org/abs/1805.06299). In short, what we do is: 

1. take an adversarial graph autoencoder (AAE);
2. train the AAE on the nominal samples that you have at training time;
3. impose a geometric regularization onto the latent space of the AAE, so that the embeddings will lie on a Riemannian constant-curvature manifold (CCM).  
This happens in one of two ways: 
	1. use a prior distribution with support on the CCM to train the AAE;
	2. make the encoder maximise the membership of its embeddings to the CCM (this is the one we use for this experiment);
4. use the trained AAE to represent incoming graphs on the CCM;
5. run the change detection test on the CCM;

![embeddings]({{ site.url }}/images/2019-04-13/embeddings.svg "Figure 2: embeddings produced by the AAE on the three different CCMs. Blue for hugging, orange for punching."){: .full-width}

This procedure can be adapted to learn a representation on more than one CCM at a time, by having parallel latent spaces for the AAE. This worked pretty well in the paper, so we tried the same here. 
We also chose one of the two types of change detection tests that we introduced in the paper, namely the one we called _Riemannian_, because it gave us the best results on the seizure detection problem. 

## Results

Running the whole method on the stream of graphs gave us very nice results. We were able to recognize the change from friendly to violent interactions in most experiments, although sometimes the autoencoder failed to capture the differences between the two regimes (and consequently, the CDT couldn't pick up the change).

![accumulator]({{ site.url }}/images/2019-04-13/accumulator.svg "Figure 3: accumulators of R-CDT (see the paper) for the three CCMs. The change is marked with the red line, the decision threshold with the green line. "){: .full-width}

An interesting thing that we observed is that when using an ensemble of three different geometries, namely spherical, hyperbolic, and Euclidean, the change would only show up in the spherical CCM. 
This was a consistent result that gave us yet another confirmation of two things: 

1. assuming Euclidean geometry for the latent space is not always a good idea;
2. our idea of learning a representation on multiple CCMs at the same time worked as expected. Originally, we suggested this trick to potential adopters of our CDT methodology, in order to not having to guess the best geometry for the representation. Now, we have the confirmation that it is indeed a good idea, because the AAE will choose the best geometry for the task on its own.

Figure 2 above (hover over the images to see the captions) shows the embeddings produced by the encoder on the test stream of graphs. Figure 3 shows the three _accumulators_ used in the change detection test to decide whether or not to raise an alarm indicating that a change occurred. 
In both pictures, the decision for raising an alarm is informed almost exclusively by the spherical CCM. 

## Conclusions

That's all, folks!  
This was a pretty little experiment to run, and it gave us further insights into the world of non-Euclidean neural networks. We have actually [updated the paper](https://arxiv.org/abs/1805.06299) with the findings of this new experiment, and you can also try and play with our algorithm using the [code on Github](https://github.com/danielegrattarola/cdt-ccm-aae) (the code there is for the synthetic experiments of the paper, but you can adapt it to any dataset easily).

If you want to mention our CDT strategy in your work, you can cite: 

```
@article{grattarola2018change,
  title={Change Detection in Graph Streams by Learning Graph Embeddings on Constant-Curvature Manifolds},
  author={Grattarola, Daniele and Zambon, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={IEE Transactions on Neural Networks and Learning Systems},
  year={2019},
  doi={10.1109/TNNLS.2019.2927301}
}
```

Cheers!
