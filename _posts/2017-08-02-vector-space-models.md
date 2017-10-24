---
layout:     post
title:      "Vector space models"
date:       2017-08-02
summary:    A quick primer on representing documents with vectors.
categories: nlp ml
custom_js:
    - katex
published: false
---

Before we can run any machine learning algorithms on text documents, we need to find a way to represent them as vectors. An ideal representation would distribute documents in a multi-dimensional vector space in such a way that similar documents end up close to each other (for now let's not worry about what "similar" or "close" means). One idea could be to construct a following term-document matrix:

{% include image name="one_hot.png" width="500" caption=""%}

Each row of the matrix represents a document and each column corresponds to a term. The values are just raw counts of terms in respective documents. In the example above, "Emma" contains one occurence of *aardvark* and two occurences of *aardwolf*, but no occurences of *zulu*, while "Alice in Wonderland" features surprisingly many Zulus and no aard-creatures. 

Now this is a pretty awful idea. First of all, we are using raw counts, so longer documents will result in much larger values. Even if we adjust for the length of document, the most frequent terms will likely be *the*, *be*, *to*, *of*, and the like, so all the vectors will end up pretty similar. What we need is a cunning way to discount unimportant terms and bump up relevant ones. Intuitively, a term is relevant to a document if it occurs frequently in it, but is rare across all documents. Slightly more formally, given a collection of documents $$D$$, we would like to assign to each term $$t$$ in document $$d \in D$$ a weight $$w_{t, d}$$ that is:

* proportional to the frequency of $$t$$ in $$d$$,
* inversely proportional to the percentage of documents in D containing $$t$$.

Just by formulating the problem, we accidentaly discovered a weighting scheme aptly named *term frequency-inverse document frequency*: 

$$ \text{tf-idf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D) $$

In the most popular variant, term frequency is the count of term $$t$$ in document $$d$$ divided by the length of the document:

$$ \text{tf}(t, d)=\frac{n_{t, d}}{\sum_{t' \in d}n_{t', d}} $$

Inverse document frequency is usually calculated as the logarithm of the total number of documents divided by the number of documents containing $$t$$:

$$ \text{idf}(t, D)=\log\frac{|D|}{|{d \in D: t \in d}|} $$

If you find the above notation to be unnecessary cryptic, here's a [pythonic](https://www.youtube.com/watch?v=M_eYSuPKP3Y) example:

```python
>>> doc1 = 'egg bacon sausage and spam'
>>> doc2 = 'spam bacon sausage and spam'
>>> doc3 = 'spam egg spam spam bacon and spam'
>>> D = [doc1, doc2, doc3]
```
Say we want to find the tf-idf weights for all the terms in the second document. Let's start with term frequencies:
```python
>>> from collections import Counter 
>>> count = Counter(doc2.split())
>>> terms = count.keys() 
>>> total = sum(count.values())
>>> tf = {t: count[t]/total for t in terms}
>>> print(tf)

{'sausage': 0.2, 'bacon': 0.2, 'spam': 0.4, 'and': 0.2}
```
As we can see, *spam* scores pretty high on term frequency. Hopefully, inversed document frequency will fix that:
```python
>>> from math import log 
>>> idf = {t: log(len(D)/sum(t in d.split() for d in D)) for t in terms}
>>> tf_idf = {t: round(tf[t]*idf[t], 3) for t in terms}
>>> print(tf_idf)

{'bacon': 0.081, 'and': 0.0, 'spam': 0.0, 'sausage': 0.081}
```
Whoa, looks like it fixed it a bit too much. Because *spam* and *and* appear in all documents, their inverse document frequency reduces to the logarithm of one, also known as zero. A common trick to avoid that is to use smoothing, i.e. add one inside the logarithm:

$$ \text{idf}(t, D)=\log\left(1+\frac{|D|}{|{d \in D: t \in d}|}\right) $$

