---
layout:     post
title:      "Shakespeare's favourite word"
date:       2017-05-01
summary:    Playing around with books from project Gutenberg.
categories: nlp ml
---

### Prologue: In Which the Author Explains Why He's Doing What He's Doing

Books are fun. <sup>[[citation needed](https://xkcd.com/285)]</sup> 

What's even more fun are vector space models, clustering algorithms, and dimensionality reduction techniques. In this blog post, we're going to combine it all by playing around with a small set of texts from project Gutenberg. With a bit of luck, Python, and lots of trial and error, we might just learn something interesting.

### Chapter One: In Which Books are Fetched and Puns are Made
We should start by fetching some books. There are many ways to do it, but for starters let's just use what NLTK has to offer: 
{% highlight python %}
>>> from nltk.corpus import gutenberg
>>> gutenberg.fileids()
['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt',
 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt',
 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt',
 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt',
 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt',
 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
{% endhighlight %}
This rather eclectic collection will serve as our dataset. How about we narrow it down to the cool authors:
{% highlight python %}
from nltk.corpus import gutenberg

fileids = gutenberg.fileids() 
cool_authors = ('austen', 'blake', 'bryant',
                'burgess', 'carroll', 'chesterton',
                'milton', 'shakespeare', 'whitman')

titles = [title for title in fileids if title.startswith(cool_authors)]
texts = [gutenberg.raw(title) for title in titles] 
{% endhighlight %}

Conveniently (and completely coincidentally) the remaining titles fall into five distinct categories I spent far too much time naming:
- Darcy and Company: Emma, Persuasion, Sense and Sensibility
- Bard's Tales: Julius Caesar, [The Scottish Play](https://www.youtube.com/watch?v=h--HR7PWfp0), Hamlet
- Chestertomes: The Ball and the Cross, The Wisdom of Father Brown, The Man Who Was Thursday
- BMW (Blake, Milton, Whitman): Poems, Paradise Lost, Leaves of Grass
- BBC (Bryant, Burgess, Carroll): Stories to Tell to Children, The Adventures of Buster Bear, Alice in Wonderland   

In other words, our modest library contains three Jane Austen's novels, three Shakespeare's plays, three novels by Gilbert K. Chesterton, three poem collections, and three children books (I'm sorry, Mr. Carroll). Let's find out if computers share our intuitions.

### Chapter Two: In which Books are Magically Turned into Numbers and What Happens Then

