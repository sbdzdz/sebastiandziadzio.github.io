---
layout:     post
title:      "Fun with Books"
date:       2017-05-01
summary:    Playing around with books from project Gutenberg.
categories: nlp
---
Books are fun. <sup>[[citation needed](https://xkcd.com/285)]</sup> 

What's even more fun are vector space models and dimensionality reduction algorithms , and Python. In this blog post, we're going to combine it all by playing around with a small set of texts from [project Gutenberg](http://www.gutenberg.org/).

We should start by fetching some books. There are many ways to do it, but as it's not the main focus of this post, let's just use what [NLTK](http://www.nltk.org/) has to offer: 
```python
>>> from nltk.corpus import gutenberg
>>> gutenberg.fileids()
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
```
This rather eclectic collection will serve as our dataset. How about we get rid of the boring authors:
```python
from nltk.corpus import gutenberg

files = gutenberg.fileids() 
cool_authors = ('austen', 'blake', 'bryant',
                'burgess', 'carroll', 'chesterton',
                'milton', 'shakespeare', 'whitman')

titles = [title for title in files if title.startswith(cool_authors)]
texts = [gutenberg.raw(title) for t in titles] 
```
Note how `str.startswith` allows supplying a tuple of strings to test for. Now, there are fifteen titles in `titles`. Conveniently (and completely coincidentally) they fall into five distinct categories I spent far too much time naming:
* Mr. Darcy et al.: Emma, Persuasion, Sense and Sensibility
* Chestertomes: The Ball and the Cross, The Wisdom of Father Brown, The Man Who Was Thursday
* The Bard's Tales: Julius Caesar, [The Scottish Play](https://www.youtube.com/watch?v=h--HR7PWfp0), Hamlet
* BMW (Blake, Milton, Whitman): Poems, Paradise Lost, Leaves of Grass
* BBC (Bryant, Burgess, Carroll): Stories to Tell to Children, The Adventures of Buster Bear, Alice in Wonderland   
 

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
