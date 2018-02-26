---
layout:     post
title:      "Data science cheatsheet: horizontal bar charts in Python"
date:       2018-01-10
summary:    Quick and pretty horizontal bar charts in Python and R.
categories: dataviz, ds-cheatsheet
published:  false
---

Working with data, I noticed there are things I Google almost on a daily basis. This series of posts is a chaotic collection of simple recipes in Python and R written in the hope that I'll finally memorize them. I took an effort to make the code as copypastable as possible, so all examples should just work™. The code from post is available [here](https://github.com/sebastiandziadzio/data-science-cheatsheet/blob/master/horizontal_bar_chart.ipynb).

### Why horizontal?
It usually makes sense to have the categorical variable on the vertical axis in order to avoid ugly slanted text or tiny font when trying to cram a dozen categories on the horizontal axis. Other than that, there's no reason, I just like them.

### Data representation
A bar chart links categories with corresponding values, so a dictionary feels like a natural way to represent the data:
```python
data = {'sausage': 1, 'spam': 3, 'eggs': 2, 'bacon': 1,'ham': 2}
```

Most plotting libraries want us to split the data into categories and values. Unless the order of categories matters, we can first sort them by value to get a neat chart:

```python
import operator

sorted_data = sorted(data.items(), key=operator.itemgetter(1), reversed=True)
categories, values = zip(*sorted_data) 
```

You can treat the slightly cryptic {% ihighlight python %} zip(*sorted_data) {% endihighlight %} as the [reverse of zip](https://stackoverflow.com/questions/12974474/how-to-unzip-a-list-of-tuples-into-individual-lists). The sorting part is a bit hard to remember, so I often use the following ~~hack~~ trick to get the categories and values in descending order:

```python
from collections import Counter

counter = Counter(data)
categories, values = zip(*counter.most_common())
```

This also allows you to limit the chart to top *n* categories, simply with {% ihighlight python %} most_common(n) {% endihighlight %}.

### Matplotlib
Here's how to get a nice chart with raw Matplotlib:

```python
import matplotlib.pyplot as plt

y_pos = range(len(categories))

# size and style
fig = plt.figure(figsize=(20, 15))
plt.style.use('ggplot')
ax = plt.gca()

# labels
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.invert_yaxis()

# label font size
plt.tick_params(labelsize=20)

# plot
ax.barh(y_pos, values, align='center')
plt.show())
```

{% include image name="chart.png" height="300" caption=""%}

Voilà! Forcing Matplotlib to produce charts that don't look like they belong in a board room meeting of a 1990s corporation used to be difficult, but the introduction of style sheets changed that. If you don't share my love for the `ggplot` style, you can find the full gallery [here](https://matplotlib.org/examples/style_sheets/style_sheets_reference.html).

### Pandas and Seaborn
The chart already looks decent, but getting it to look *exactly* how you want it can be a pain in Matplotlib. It is highly customizable, but I can never remember the appropriate setting. Let's build the chart again, but this time using [Seaborn](https://seaborn.pydata.org/), which is essentially a high-level wrapper for Matplotlib. It works exactly the same under the hood, but it comes with a set of customizable presets and has an overall more user-friendly syntax. Let's also pretend we got the data in a CSV file.

```python
```
