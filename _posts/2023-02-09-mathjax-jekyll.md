---
title: "Enable MathJax in your Jekyll page"
date: 2020-04-20
layout: page
---

Jekyll with mathematical annotation is tricky bussiness. If you're having trouble, try following
my recipe below.

---

Append the following script before the `</head>` inside of `_layouts/default.html` (if there isn't one, just create it):
```
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        processEscapes: true
      }
    });
  </script>
      
  <script type="text/javascript"
          src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
```

The above code injection consists of two parts:

- The fisrt part (below) is a configuration script for MathJax which specifies that inline mathematics should be delimited by the characters "$" or "\(" and "\)" and that it should process escape characters. This will ensure that mathematical equations written in TeX are properly formatted and displayed within the HTML document:
    ```html
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            processEscapes: true
        }
        });
    </script>
    ```
- The second part, pulls a copy of MathJax JavaScript library from its respective CDN (group of distributed servers to allow the display of mathematical equations and formulas in HTML web pages,     The `TeX-AMS-MML_HTMLorMML` expression is a configuration option for MathJax, specifying the format in which the equations should be displayed:
    ```
    <script type="text/javascript"
            src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
    ```
Finally, note that the order of these two parts play a role in correctly rendering mathematical
expression in html using jekyll.

Easy right? 😀

### Sources:
- https://tex.stackexchange.com/a/27656
- https://docs.mathjax.org/en/v1.1-latest/options/tex2jax.html 
