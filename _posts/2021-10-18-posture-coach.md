---
layout:     post
title:      "Sitting Posture Coach"
subtitle:   "Monitor and improve your sitting posture"
date:       2021-10-18 12:00:00
author:     "Rembert Daems"
header-img: img/bg-posture-coach.jpg
header-mask: 0.3
catalog:    true
mathjax:    False
tags:
    - side project
---

<a href="https://github.com/pderoovere/sitting-posture-coach" class="btn btn-primary" style="color: white" role="button">code</a>
<a href="/sitting-posture-coach/" class="btn btn-primary" style="color: white" role="button">demo</a>

<iframe src="https://player.vimeo.com/video/549610959?h=af4564a9ca" width="640" height="360" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen></iframe>


In the context of [Full Stack Deep Learning](https://fullstackdeeplearning.com/) my friend [Peter](https://twitter.com/peterderoovere) asked me to cooperate on his final project of the course. We came up with **Sitting Posture Coach** and were selected in the [top 10 of 91 projects](https://fullstackdeeplearning.com/spring2021/projects/) by the course TAs.

Maintaining a good sitting posture while working is extremely important. This is especially true when working from home, where ergonomic office equipment might not be available and there is less incentive to sit up straight. Bad sitting posture is a major cause of back pain, neck pain, headaches and even spinal disfunction.

Sitting posture coach aims to provide a solution which is easy to set up and needs no additional infrastructure. A simple web page will provide feedback to help you attain a better sitting posture. It uses an AI system running locally in your browser (and thus preserving your privacy) that analyses live images from your webcam.

A node.js app (running on Amazon LightSail) will serve web pages for inference and data collection. Collected data is stored in the cloud. Images are stored as objects (in an Amazon S3 bucket), the corresponding metadata is stored in a database (PostgreSQL). This data is used to train a classification model. The trained model is converted to run in the browser (TFJS) and made available by the node.js app.

You can [try out the app right now](/sitting-posture-coach/) or view the code and a more detailed technical explanation on [github](https://github.com/pderoovere/sitting-posture-coach)!
