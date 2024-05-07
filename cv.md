---
layout: page
title: "About"
description: ""
header-img: "img/waterlillies.jpg"
---

<div class="en post-container">
    <div class="en post-container" style="text-align: center;">
        <a href="/Rembert_Daems_CV.pdf" class="btn btn-primary" style="color: white" role="button">Download my CV</a>
    </div>
    {% capture about %}{% include about.md %}{% endcapture %}
    {{ about | markdownify }}
</div>
