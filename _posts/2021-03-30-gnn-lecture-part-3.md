---
layout: post
categories: posts
title: "A practical introduction to GNNs - Part 3"
image: /images/2021-03-03/presentation-1.png
tags: [GNN, lecture]
date-string: MARCH 30, 2021
---

_This is Part 2 of an introductory lecture on graph neural networks that I gave for the "Graph Deep Learning" course at the University of Lugano._

_After a practical introduction to GNNs in [Part 1]({{site.url}}/posts/2021-03-03/gnn-lecture-part-1.html) and to message-passing netowrks in [Part 2]({{site.url}}/posts/2021-03-03/gnn-lecture-part-2.html), here I go over the classical approach to implement convolution in the spectral domain._

[The full slide deck is available here]({{ site.url }}/files/talks/2021-03-01-USI_GDL_GNNs.pdf).

---

<img src="{{ site.url }}/images/2021-03-03/presentation-24.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-25.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-26.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-27.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-28.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-29.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-31.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-32.svg" width="100%" style="border: solid 1px;"/>
<img src="{{ site.url }}/images/2021-03-03/presentation-33.svg" width="100%" style="border: solid 1px;"/>

---

With the first two parts of this blog series in your toolbelt, you should be able to go a long way in the world of GNNs. 

The next and final part will take a more historical and mathematical journey in the world of GNNs. We'll cover spectral graph theory and how we can define the operation of __convolution__ on graphs. 

I have left this for last because it is not _essential_ to understand and use GNNs in practice, although I think that understanding the historical perspective that led to the creation of modern GNNs is very important. 

Stay tuned.  
