---
layout: post
categories: posts
title: "Telestrations Neural Networks"
image: /images/2020-01-21/bagel.png
tags: [AI, random, code]
date-string: JANUARY 21, 2020
---

![]({{ site.url }}/images/2020-01-21/telestrations.jpg)

Yesterday, it was board game day at [the lab](http://www.neurontobrainlaboratory.ca/) where I have been working recently. 
Everyone got together for lunch at Snakes & Lattes, a Torontonian board game cafè chain, and we spent a couple of hours laughing and chatting and, obviously, playing board games. 

The lab has a go-to traditional game for the occasion: [Telestrations](https://en.wikipedia.org/wiki/Telestrations).
The game is inspired by the classic childhood's game of [Chinese whispers](https://en.wikipedia.org/wiki/Chinese_whispers) (or _Telephone_, or _Wireless phone_, or _Gossip_, there's a bunch of different names for different countries) and its rules are pretty simple. 

Everyone gets a booklet, an erasable sharpie, and a list of random terms like "flamingo" or "pipe dream" or "treehouse". Everyone picks a word and writes it on the first page of the booklet: that's the secret source word. 

At each turn, players pass their booklet to the person on their right, and the rules are as follows: 

- When you see a word, you turn the page and you have sixty seconds to _draw_ whatever the word is;
- When you see a drawing, you turn the page and you write your best guess for what is pictured. 

Players keep alternating between guessing, drawing, and passing down the booklets until every booklet has done a full round of the table and is back in the hands of the original owner. 
For extra fun, everybody gets to draw their secret source word at the very beginning.

In other words, it's a written game of Chinese whispers where every other word is drawn instead of written. 

There are some rules to decide who wins at the end, but the obvious source of entertainment is the complete chaos that ensues as information gets corrupted drawing after drawing. At the end of a round, not one of the original secret words ever survives.

So now the obvious, rational, almost trivial question is: what happens when you use a GAN to draw, and an image classifier to guess?   
Well, here I am to show you! 

<!--more-->

## How-to in three paragraphs

[BigGAN](https://openreview.net/forum?id=B1xsqj09Fm) can generate images conditioned on an ImageNet label. So if you give it label 1, it will generate goldfish, if you give it label 42, it will generate an agama, and so on. 

ResNet does the opposite: if you show it a goldfish, it will try to guess what it is. To make things more interesting, I added a bit of noise to the guessing procedure, so that sometimes we get a random one out of the top-5 guesses. If you think that this is unreasonable, try and play a game with real humans, I dare you.

The idea now is to play the game using BigGAN to draw, and ResNet to guess: you start with a label, you have BigGAN generate an image of that label, you classify that image to get a new label, and so on. 

## Results

I'll start with my favourite sequence: honeycomb to cheeseburger. 
The images below are read top-to-bottom, left-to-right. At the very top you see the source class, then the first generated image, then what that image was classified as, then the next generated image, etc.. 

The first image is generated from class 599 of ImageNet, "honeycomb". It looked a lot like a bagel, I guess because of that bright spot in the middle (?), so the ResNet classified it as such. From that classification, we get a couple of bagel-y looking pieces of bread, which soon become French loafs, then dough.  
Then, that perfect-looking dough in image 6 gets classified as a wooden spoon (probably because of the extra noise that I mentioned). 
Finally, the green spot on the wooden spoon confuses ResNet into thinking it's a cheeseburger, and we get juicy burgers until the end. That burger generation is impressive, not gonna lie. 

![]({{ site.url }}/images/2020-01-21/bagel.png)

Moving on: trilobite to long-horned beetle. 
The first two trilobites look really good, but then get classified as isopods after two turns (curiously, isopods and trilobites look a lot similar but are not that closely related according [to Reddit](https://www.reddit.com/r/geology/comments/lt9so/how_closely_related_are_isopods_to_trilobites/)).
From the isopod label, we get what is clearly a marine creature (look at the background), which unfortunately gets classified as a cockroach. From there, we stay on dry land and just get more and more specialized bugs until the end. 

![]({{ site.url }}/images/2020-01-21/trilobite.png)

The next one is __REALLY__ good because it's remarkably similar to a real game of Telestrations. It could happen. Hell, it probably happened.

We start with a coffeepot. At image three, the coffeepot is a bit ambiguous and becomes a teapot. Understandable, I would probably have made that mistake myself. Then we get a proper teapot, that gets recognized as such. 
The next image, however, is half-assed by the player and it's not clear at all what it is. The next player guesses that it's a pitcher. The next guy tries his best but eventually, the pitcher becomes a vase. 

Nothing more to say, I can see this happening in real life. 

![]({{ site.url }}/images/2020-01-21/coffeepot.png)

Our next and last one is also a likely sequence. 

A volcano. Easy. We get two perfect volcano drawings. Except that the last one gets classified as a type of tent.

The next player over-does it, and draws a full camping spot with caravans instead of a tent. Curiously, we still have a volcano-looking thing in the background, but that's just a coincidence (no information from previous images or labels is preserved between turns).

The camp is seen as a bee house. Next thing we know, there's a weird-looking BigGAN human harvesting honey. 
But ResNet doesn't care about the human and focuses on the crate in the middle, instead. 

We get a good-looking crate, that becomes a chest, and we stay with chests until the end. 

The yurt and the apiary are the only weird ones in this sequence, and the least likely to appear in a human game. I can see someone drawing a full camping spot instead of a single yurt, and I can see how one would mistake a poorly-drawn volcano for a tent, but no human would ignore the beekeeper in image 4. 

![]({{ site.url }}/images/2020-01-21/volcano.png)

I have generated a bunch of these sequences on my laptop, and these are just four random ones that I got. It's really easy to get fun sequences. So here's how I did it.

## Code

First of all, I was not going to spend a single € to train anything involved in this project because, like, let's be real...

So I turned to Google and I found: 

- [A pre-trained BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN)
- [Torchvision's pre-trained ResNet50](https://pytorch.org/docs/stable/torchvision/index.html)

I usually write my stuff in TensorFlow but whatever, let's PyTorch this one. 

We start with some essential imports: 

```python
import numpy as np
import torch
from PIL import Image
from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample, convert_to_images
from spektral.utils import init_logging
from torchvision import models
from torchvision import transforms
```

and we define a couple of useful variables:

```python
iterations = 8  # How many players there are
standard_noise = 0.3  # Some random noise because people are not perfect
current_class = np.random.randint(0, 1001)  # The secret source word is random

# Load ImageNet class list
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
``` 

`imagenet_classes.txt` [can be found online](https://github.com/Lasagne/Recipes/blob/master/examples/resnet50/imagenet_classes.txt), it's just a list of ImageNet class names.

Now, let's create the models that we will use. First we create the GAN: 

```python
gan = BigGAN.from_pretrained('biggan-deep-256')
gan.to('cuda')
```

Then, we create the ResNet50 ImageNet classifier: 

```python
classifier = models.resnet50(pretrained=True)
classifier.eval()  # Do this to set the model to inference mode
```

and its image pre-processor:

```python
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
 )])
```

We will be drawing and guessing images of 256 x 256 pixels (cropped to 224 x 244 for ResNet50). The hard-coded normalization is just something that you have to do for Torchvision models, no biggie.

So now we have loaded the networks. Let's define some helper functions that will compute the main steps of the game for us: 

```python
def draw(label, truncation=1.):
    # Create the inputs for the GAN
    class_vector = one_hot_from_int([label], batch_size=1)
    class_vector = torch.from_numpy(class_vector)
    class_vector = class_vector.to('cuda')

    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
    noise_vector = torch.from_numpy(noise_vector)
    noise_vector = noise_vector.to('cuda')

    # Generate image
    with torch.no_grad():
        output = gan(noise_vector, class_vector, truncation)
    output = output.to('cpu')

    # Get a PIL image from a Torch tensor
    img = convert_to_images(output)

    return img
    

def guess(img, top=5):
    # Pre-process image
    img = transform(img[0])

    # Classify image
    classification = classifier(img.unsqueeze(0))
    _, indices = torch.sort(classification, descending=True)
    percentage = torch.nn.functional.softmax(classification, dim=1)[0]

    # Get the global ImageNet class, labels, and the predicted probabilities
    idxs = np.array([idx for idx in indices[0]][:top])
    labs = np.array([labels[idx] for idx in indices[0]][:top])
    probs = np.array([percentage[idx].item() for idx in indices[0]][:top])
    
    return idxs, labs, probs
```

Now we can start playing!

```python
output_imgs = []  # Stores the drawings
output_labels = []  # Stores the guesses
output_labels.append(labels[current_class])

# Main game loop
for i in range(iterations):
    # Draw an image
    img = draw(current_class)
    output_imgs.append(img[0])

    # Guess what the image is
    idxs, labs, probs = guess(img, top=top)

    # Add noise
    probs += np.random.uniform(0, standard_noise, size=probs.shape)
    probs /= probs.sum() # Re-normalize because of noise

    # Choose from the predictions
    choice = np.random.choice(np.arange(len(labs)), p=probs)
    current_class = idxs[choice]
    output_labels.append(labs[choice])
```

At the end of the game, we will have the generated drawings in `output_imgs` and the guesses in `output_labels`.

Here, instead of copy-pasting from the cells above you can just look at [the full gist](https://gist.github.com/danielegrattarola/8296b9fd29116443da74d0aa2519d7c3).

## Conclusions
What can I say? It's neural networks playing Telestrations. 

"No new knowledge can be extracted from my telling. This confession has meant nothing."

Cheers!

---

In case you didn't know: agamas (label 42 of ImageNet) are extra-fucking-cool lizards.

![]({{ site.url }}/images/2020-01-21/agama.jpg)
