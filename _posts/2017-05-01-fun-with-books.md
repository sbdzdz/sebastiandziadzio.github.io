---
layout:     post
title:      "Fun with Books"
date:       2017-05-01
summary:    Playing around with books from project Gutenberg.
categories: nlp ml
---

### Prologue: In Which the Author Explains Why He's Doing What He's Doing

Books are fun. <sup>[[citation needed](https://xkcd.com/285)]</sup> 

What's even more fun are vector space models and dimensionality reduction algorithms. In this blog post, we're going to combine it all by playing around with a small set of texts from [project Gutenberg](http://www.gutenberg.org/). With a bit of luck, Python, and lots of errors, we might just learn something interesting.

### Chapter One: In Which Books are Fetched and Puns are Made
We should start by fetching some books. There are many ways to do it, but for starters let's just use what [NLTK](http://www.nltk.org/) has to offer: 
{% highlight python %}
>>> from nltk.corpus import gutenberg
>>> gutenberg.fileids()
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
{% endhighlight %}
This rather eclectic collection will serve as our dataset. How about we narrow it down to the cool authors:
{% highlight python %}
from nltk.corpus import gutenberg

files = gutenberg.fileids() 
cool_authors = ('austen', 'blake', 'bryant',
                'burgess', 'carroll', 'chesterton',
                'milton', 'shakespeare', 'whitman')

titles = [f for f in files if f.startswith(cool_authors)]
texts = [gutenberg.raw(title) for title in titles] 
{% endhighlight %}

Conveniently (and completely coincidentally) the remaining titles fall into five distinct categories I spent far too much time naming:
1. Mr. Darcy et al.:
* Emma,
* Persuasion,
* Sense and Sensibility
2. Chestertomes:
* The Ball and the Cross,
* The Wisdom of Father Brown,
* The Man Who Was Thursday
3. The Bard's Tales:
* Julius Caesar,
* [The Scottish Play](https://www.youtube.com/watch?v=h--HR7PWfp0),
* Hamlet
4. BMW (Blake, Milton, Whitman):
* Poems,
* Paradise Lost,
* Leaves of Grass
5. BBC (Bryant, Burgess, Carroll):
* Stories to Tell to Children,
* The Adventures of Buster Bear,
* Alice in Wonderland   
 
PART ONE
1. What is project Gutenberg? What do I want to do? What is tf-idf and why it may be useful?.
2. Corpus from NLTK.
* show the code (perhaps interactive)
* calculate tf-idf
* show common words
* calculate cosine similarity
* show results (visualisation)
3. Reading files from disk.
4. Teaser for part two
PART TWO
1. Downloading books.
2. Calculate tf-idf (perhaps different classes, authors).
3. Calculate cosine similarity.
4. Show results (visulaisation).
5. Train classifier (genres, authors).
6. Conclusions, discussions, full code, links.
