---
layout:     post
title:      "Unsupervised Orientation Learning Using Autoencoders"
subtitle:   "Disentangled Representations of Orientation"
date:       2020-12-01 12:00:00
author:     "Rembert Daems"
permalink: /unsupervised-orientation/
header-style: text
catalog:    true
mathjax:    true
tags:
    - research
---

<a href="https://biblio.ugent.be/publication/8683423/file/8683425" class="btn btn-primary" style="color: white" role="button">paper</a>
<a href="https://slideslive.com/38941768" class="btn btn-primary" style="color: white" role="button">slides</a>

I presented this work on the 2020 NeurIPS workshop on Differential Geometry meets Deep Learning.

### Abstract

We present a method to learn the orientation of symmetric objects in real-world images in an unsupervised way.
Our method explicitly maps in-plane relative rotations to the latent space of an autoencoder, by rotating both in the image domain and latent domain.
This is achieved by adding a proposed crossing loss to a standard autoencoder training framework which enforces consistency between the image domain and latent domain rotations.
This relative representation of rotation is made absolute, by using the symmetry of the observed object, resulting in an unsupervised method to learn the orientation.
Furthermore, orientation is disentangled in latent space from other descriptive factors.
We apply this method on two real-world datasets: aerial images of planes in the DOTA dataset and images of densely packed honeybees.
We empirically show this method can learn orientation using no annotations with high accuracy compared to the same models trained with annotations.

{% include figure.liquid loading="eager" path="assets/img/unsupervised-orientation/planes.gif" class="img-fluid rounded z-depth-1" width="30%" %}
