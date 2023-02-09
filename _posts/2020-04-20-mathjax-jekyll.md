---
title: "Enable mathjax in your jekyll page"
date: 2020-04-20
layout: page
---

Quick note on highlighting how I enabled mathjax on my jekyll blog.

1. Add the following at the top of `_layouts/default.html` (if there isn't one, just create it)

```
<script type="text/javascript" async 
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?
config=TeX-AMS-MML_HTMLorMML"></script>

```
2. In the `_config.yml`, add or make sure the following exist:
```
markdown: kramdown
mathjax: true
```
For renedering Mathjax:

- inline, use `\( ... \)`,
- block, use `\[ ... \]`.

The only thing to look out for is the escaping of the backslash when using markdown, so the delimiters become \\( ... \\) and \\[ ... \\] for inline and block maths respectively.
Example of `\[ \frac{1}{n^{2}} \]`: \[ \frac{1}{n^{2}} \]

Easy right? 😀