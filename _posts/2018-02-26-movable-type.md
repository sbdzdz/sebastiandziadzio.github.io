---
layout:     post
title:      "Movable Type"
date:       2017-05-05
summary:    Playing around with books from project Gutenberg.
categories: nlp ml
custom_js:
    - katex
published: true
---

## Prologue

Books are fun. <sup>[[citation needed](https://xkcd.com/285)]</sup> 

What's even more fun are vector space models, clustering algorithms, and dimensionality reduction techniques. In this post, we're going to combine it all by playing around with a small set of texts from project Gutenberg. With a bit of luck and lots of Python we might just learn something interesting.

## Chapter One: Index Librorum
First, we're going to need some books. Let's be lazy about it and just use what NLTK has to offer: 

```python
from nltk.corpus import gutenberg

fileids = gutenberg.fileids()
print(', '.join(fileids))
```

```sh
austen-emma.txt, austen-persuasion.txt, austen-sense.txt, bible-kjv.txt, blake-poems.txt, bryant-stories.txt, burgess-busterbrown.txt, carroll-alice.txt, chesterton-ball.txt, chesterton-brown.txt, chesterton-thursday.txt, edgeworth-parents.txt, melville-moby_dick.txt, milton-paradise.txt, shakespeare-caesar.txt, shakespeare-hamlet.txt, shakespeare-macbeth.txt, whitman-leaves.txt
```

This rather eclectic collection will serve as our dataset. We can weed out the boring books and fetch the full text for the others. Let's also be pedantic and format the titles:

```python
import re

boring = {'bible-kjv.txt',
          'edgeworth-parents.txt',
          'melville-moby_dick.txt'}

fileids = [f for f in fileids if f not in boring]
texts = [gutenberg.raw(f) for f in fileids]
titles = [t.replace('.txt', '') for t in titles]
```

Conveniently, and completely coincidentally, the remaining titles fall roughly into five categories I spent too much time naming:
 
- Novel and Novelty: *Emma, Persuasion, Sense and Sensibility*

- Bard's Tales: *Julius Caesar, Hamlet, The Scottish Play*[^1]

- Chestertomes: *The Ball and the Cross, The Wisdom of Father Brown, The Man Who Was Thursday*

- BBC (Bryant, Burgess, Carroll): *Stories, The Adventures of Buster Bear, Alice's Adventures in Wonderland*

- BMW (Blake, Milton, Whitman): *Poems, Paradise Lost, Leaves of Grass*

In other words, our modest library contains three Jane Austen's novels, three Shakespeare's plays, three novels by Gilbert K. Chesterton, three children books (I'm sorry, Mr Carroll) and some poetry. Let's see if this classification is equally intuitive to a machine.

## Chapter Two: A is for Aardvark
There are different ways to represent text documents as vectors. We're going to use term frequency-inverse document frequency (tf-idf). Technical [details](https://en.wikipedia.org/wiki/Tf–idf) aside, the tf-idf score of a term in a document is largest when that term occurs frequently in that document, but is rare across all documents in the collection. 

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(texts,
                             max_df=0.5,
                             stop_words='english')

tfidf_matrix = vectorizer.fit_transform(texts)
```

The `TfidfVectorizer` does all[^2] the work – it filters stop words, normalizes the term vectors and even lets us impose constraints on document frequency (`max_df=0.5` means we ignore terms that occur in more than half of the documents). The result is a matrix whose every column corresponds to a term and every row is a vector representing a book:

{% include image name="tfidf.png" width="600" caption=""%}

The `tfidf_matrix` is just a (sparse) numerical matrix, so we need to know which row/column corresponds to which book/term. The order of rows is the same as in `texts`, so that part is easy. Getting the list of terms is a bit more tricky:

```python
terms = vectorizer.get_feature_names()
```
The terms are sorted alphabetically, so the beginning of the list will be mostly numbers. Here's how to see some words:
```python
a = [t for t in terms if t.startswith('a')][:5]
z = [t for t in terms if t.startswith('z')][-5:]
print(f'{" ".join(a)}\n{" ".join(z)')
```

```sh
aaron aback abandon abandoned abandoning
zoological zophiel zso zumpt zuyder
```

Nice! No *aardvarks* or *aardwolves*, but at least *Aaron* is there. We can now find top five terms for each book: 
```python
top_indices = (-tfidf_matrix).toarray().argsort()[:, :5]

for index, title in enumerate(titles):
    top_terms = [vocabulary[t] for t in top_indices[index]]
    print(f'{title}:\n{", ".join(top_terms)}\n\n')
```

```sh
austen-emma: 
emma harriet weston knightley elton

austen-persuasion:
anne elliot wentworth captain charles

austen-sense:
elinor marianne dashwood jennings willoughby

blake-poems:
weep thel infant lamb lyca

bryant-stories:
jackal margery nightingale big brahmin

burgess-busterbrown:
buster joe farmer blacky sammy

carroll-alice:
alice gryphon turtle hatter duchess

chesterton-ball:
turnbull macian evan police gutenberg

chesterton-brown:
flambeau muscari boulnois fanshaw duke

chesterton-thursday:
syme professor gregory marquis bull

milton-paradise:
hath adam eve satan spake

shakespeare-caesar:
brutus cassius caesar haue antony

shakespeare-hamlet:
hamlet haue horatio queene laertes

shakespeare-macbeth:
macbeth haue macduff rosse vpon

whitman-leaves:
states poems cities america chant
```

Our library is rather small, so dull words like *spoke* and *had* unfortunately made the cut, just because Milton and Shakespeare spelled them funny. This could be fixed by adding all the *haues*, *spakes*, *vpons*, and *haths* to the stop word list. Apart from that, the results are pretty informative, with character names generally scoring highest on tf-idf.

Curiously, it seems Brutus stole the show from the eponymous character of Shakespeare's Julius Caesar (*et tu, Brute?*) and Satan is more prominent than God in Paradise Lost. The latter is particularly interesting, as it sheds some light on how tf-idf works. A quick grep shows that 327 verses of Milton's masterpiece take the Lord's name in vain, while only 72 feature Satan, so good triumphs over evil as far as term frequency is concerned. However, Satan is mentioned less often in other books in our collection, so (s)he[^3] scores much higher on inverse document frequency. 

Lyric poetry features fewer characters, so only Blake's Lyca and Thel make it to the top five. Instead, we get an insight into poets' favourite topics – it appears Whitman was really into America, while Blake's poems involve lots of weeping lambs.


## Chapter Three: Wordcloud Atlas
For a more artistic take, let's visualize the tf-idf vectors as [wordclouds](https://github.com/amueller/word_cloud). I'll let you guess which is which.

{% include image name="pride_wordcloud.png" width="800" caption="" %}

{% include image name="alice_wordcloud.png" width="800" caption="" %}

{% include image name="paradise_wordcloud.png" width="800" caption="" %}

{% include image name="hamlet_wordcloud.png" width="800" caption="" %}

Poor Yorick tells us quite a lot about Shakespearean orthography. It seems *i*/*j* were used interchangeably (*ioy*, *poyson*, *lye*) and *u*/*v* could represent either a vowel or a consonant, with *v* used at the beginning of a word (*vpon*, *haue*, *vs*). Moreovere, it woulde seeme silente finale *ees* were quite populare.

## Chapter Four: k-Means to an End

Wordclouds were a cool digression, but it's time to get back to the main question – what happens if we try to automatically classify our books into categories? We'll use the k-means clustering algorithm to find out. It starts by choosing $$k$$ samples to serve as means, creates clusters by assigning every other sample to the nearest mean (using squared Euclidean distance) and moves the means to the centroids of the new clusters. After repeating this process a couple of times, the algorithm usually converges, although often to a local optimum. How do we choose $$k$$? In our case we just set it to five, since we want five clusters, but in general it's a problem deserving at least a [Wikipedia page](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set). Once again, `sklearn` has us covered – it even works on sparse matrices! It lets you find the clusters in two lines of code (and then spend six more trying to print them):

```python
from sklearn.cluster import KMeans
from collections import defaultdict

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(tfidf_matrix)

clustering = defaultdict(list)
for title, label in zip(titles, labels):
    clustering[label].append(title)
    
for cluster, elements in sorted(clustering.items()):
    elements = ', '.join(elements)
    print(f'Cluster {label}: {elements}')
```
```sh
0: bryant-stories, burgess-busterbrown
1: austen-emma, austen-persuasion, austen-sense, chesterton-ball, chesterton-brown, chesterton-thursday
2: shakespeare-caesar, shakespeare-hamlet, shakespeare-macbeth
3: blake-poems, milton-paradise, whitman-leaves
4: carroll-alice
```
Not bad! First of all, I owe Mr [Dogdson](https://en.wikipedia.org/wiki/Lewis_Carroll) another apology. *Alice* is in a class of its own and doesn't belong with children stories. Instead, Austen and G.K. have to share a cluster. The rest is as expected – poets stick together[^4] and Shakespeare's tragedies are in one place. When I ran it with `n_clusters=6`, Chesterton and Austen got a room of their own, but *Alice* was still lonely.

If you run the above code a couple of times, you might notice you're getting different outcomes. This is the local optimum problem. We can try to avoid it by running the algorithm multiple times and choosing the best result. Actually, this already happens behind the scenes: by default, `sklearn` returns the best of 10 runs. Unfortunately, we can still get unlucky and hit a sub-optimal solution every time. Let's run the algorithm 50 times and see if anything changes:

```python
kmeans = KMeans(n_clusters=5, n_init=50)
```
```sh
0: blake-poems, milton-paradise, whitman-leaves
1: austen-emma, austen-persuasion, austen-sense
2: chesterton-ball, chesterton-brown, chesterton-thursday
3: shakespeare-caesar, shakespeare-hamlet, shakespeare-macbeth
4: bryant-stories, burgess-busterbrown, carroll-alice
```
Nice! This time we find a better solution and the clustering is exactly as expected. Unless you're really unlucky, the result should be consistent if you re-run the code. But how do we know it's better? The $$k$$-means algorithm chooses a solution with lowest inertia. We can use it to compare different clusterings:
```python
print(kmeans.inertia_)
```
```sh
8.985937270983072
```
Inertia is the sum of squared Euclidean distances[^5] of samples to their closest cluster center. Minimizing it is a good idea in our case, because we expect a known number of similar-sized, well-separated, convex clusters. Unfortunately, the world is [rarely so perfect](https://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means), so good clustering means different things in different contexts.

## Chapter Five: A Romance of Many Dimensions
The final step is to find a way to visualize the clusters. There is an important problem with visualizing tf-idf vectors – they live in a high-dimensional space:

```python
print(len(terms))
```
```sh
28282
```
People generally have trouble visualizing more than three dimensions [^6]. We'll use principal component analysis to find a two-dimensional representation of our data:
```python
from sklearn.decomposition import PCA

dense_tfidf_matrix = tfidf_matrix.todense()
pca = PCA(n_components=2)
points_pca = sklearn_pca.fit_transform(dense_tfidf_matrix)
```
{% include image name="pca.png" width="800" caption="" %}
The result highlights the differences between poetry, prose, and drama, but the three clusters within prose are not so visible. Still, it's pretty impressive considering we're essentially projecting 28282 dimensions on a plane.

Anyway, PCA is [so 1901](http://www.stats.org.uk/pca/Pearson1901.pdf) and all the cool kids are using t-SNE now. It works just like magic – it's powerful, hard to understand, and easy to abuse. It has three knobs (perplexity, learning rate, and the numbers of iterations), but the effect of those hyperparameters on results might be hard to figure out:
{% include image name="bears.gif" width="500" caption="" %}

If you'd like some intuition about how t-SNE works, this [article](https://distill.pub/2016/misread-tsne/) offers a detailed look inside the black box. The important rule is that t-SNE will freak out if perplexity is greater than the number of points. I used the following configuration:

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,
            learning_rate=10,
            perplexity=10,
            n_iter=500)
points_tsne = tsne.fit_transform(dense_tfidf_matrix)
```
{% include image name="tsne.png" width="800" caption="" %}

These are pretty dramatic clusters, but the sceptic in me remembers that t-SNE sometimes suggests structure which isn't really there. However, with supporting evidence from k-means, we can pretty confidently say that those clusters are not an accident.

## Epilogue
I guess the main lesson here is that simple stuff can get you pretty far. Except for t-SNE, all the techniques we used are several decades old, and yet they enabled us to learn quite a bit about the dataset. It would be tempting to repeat this analysis on a larger corpus.

## Footnotes
[^1]: One should never say the name of the [Scottish Play](https://www.youtube.com/watch?v=h--HR7PWfp0). It is unclear whether the curse extends to web publications, but I want to be on the safe side.
[^2]: Actually, that's not exactly true. I edited the texts a bit before feeding them into the vectorizer. For example, the names of characters in Shakespeare's plays are often abbreviated, so I replaced them with full versions to avoid duplicates. You'll run into problems like that on every text processing task and sometimes there's no way to fully automate it. It's usually a tradeoff between the desire for perfect results and the amount of fucks you're willing to give. 
[^3]: Ok I might [over-research](https://en.wikipedia.org/wiki/Sexuality_in_Christian_demonology) things.
[^4]: Those three have quite a lot in common. Blake was evidently a Milton's fanboy – he illustrated Milton's work more often than that of any other author and even wrote an epic poem called Milton, starring John Milton as a falling star entering Blake's foot ([I'm not making this up](https://en.wikipedia.org/wiki/Milton:_A_Poem_in_Two_Books)). Blake, in turn, was probably a major inspiration for Whitman, who even based the design of his own burial vault on Blake's engraving.
[^5]: The sum of squared distances is more of an intuition here, we're really minimizing within-cluster variance, but they're the same in this case.
[^6]: Mathematicians like to claim it is in fact trivial. To imagine a four-dimensional space, you first imagine an $$n$$-dimensional space and then simply set $$n=4$$.
