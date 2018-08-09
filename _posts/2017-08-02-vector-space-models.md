---
layout:     post
title:      "Vector space models"
date:       2017-08-02
summary:    Representing text documents with vectors.
categories: nlp ml
published: false
typora-root-url: ../../blog
---

## What is a vector space model?
Computers, as the name suggest, are good with numbers. While representing images or one-dimensional signals as lists of numbers is straightforward, text requires a bit more work. That's what vector space models do – they map words, sentences, or whole documents to numerical vectors. There are many ways to do it and the deep learning era brought around a couple of [clever solutions](https://github.com/MaxwellRebo/awesome-2vec) to that problem. In this post we're going to reinvent a classic document-level vector space model called *term frequency - inverse document frequency*. We'll then use it to build a simple search engine!

## First attempt: sets of words
Depending on the application, we might want the document vectors to have different properties. Since we want to build a search engine, an ideal model would map similar documents to similar vectors. Note that we want every vector to have the same length. That way it will be easier to come up with a similarity metric.

I always find it easier to work on an example, so let's say these are our documents:

```python
first = 'egg bacon sausage and spam'
second = 'spam bacon sausage and spam'
third = 'spam egg spam and spam'

documents = [first, second, third]
```

We'll start with a simple idea. Let's represent documents as rows of 1s and 0s, where 1 means a certain word occurs in the document and 0 means it doesn't:

{% include image name="first_attempt.png" width="500" caption=""%}

The matrix of this form is called a term-document matrix. Every document is represented by a document vector – a list of numbers, each corresponding to a term. Of course we can't use every term in the language, so we need to limit ourselves to a pre-defined set called a vocabulary. The easiest solution is to use all the words from all the documents. This is not always feasible, so it sometimes makes sense to limit the vocabulary to N most popular words. Note that the vocabulary size determines the dimensionality of document vectors.

//TODO fix code (`homar`)

To implement our basic model in Python, we first need to build the vocabulary:

```python
def build_vocabulary(documents):
    vocabulary = set()
    for d in documents:
        vocabulary.update(d.split())
    return sorted(vocabulary)

vocabulary = build_vocabulary(documents)
print(vocabulary)
```

```sh
['and', 'bacon', 'egg', 'homar', 'sausage', 'spam']
```

We can now create the term-document matrix (`tdm` for short):
```python
def build_tdm(documents, vocabulary):
    tdm = []
    for d in documents:
        tdm.append([int(t in d) for t in vocabulary])
    return tdm

tdm = build_tdm(documents, vocabulary)
print(tdm)
```
```sh
[[1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1], [1, 0, 1, 0, 0, 1]]
```

If you feel your Python sense tingling and hear a voice in your head whisper "there should be a package for that", you're right! With a little help from scikit-learn, we can do all the above in two lines:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary=True)
tdm = vectorizer.fit_transform(documents).to_array()

print(tdm)
```
```sh
[[1 1 1 0 1 1]
 [1 1 0 1 1 1]
 [1 0 1 0 0 1]]
```
Neat! Of course, we can access the vocabulary as well:

```python
vocabulary = vectorizer.get_feature_names()
print(vocabulary)
```
```sh
['and', 'bacon', 'egg', 'homar', 'sausage', 'spam']
```

Now, a proper vector space model must be able to convert any text document into a vector:
```python
print(vectorizer.transform(['spam spam aand spam']).toarray())
```


## From sets to bags
Our first attempt is very simple, but not completely useless. It lets us run boolean queries like "find all documents that contain *spam* and don't contain *sausage*". However, by reducing every document to a set of words we loose a lot of information. Sure, we can tell if a word occurs in the document, but we don't even know how many times it occured. Take the first document as an example:

{% katex display %}
\text{spam egg sausage and spam}
{% endkatex %}

In our binary model it is simply represented as an unordered set:

{% katex display %}
\text{\{spam, egg, and\}}
{% endkatex %}

Again, each row of the matrix represents a document and each column corresponds to a term. The values are just raw counts of terms in respective documents. In the example above, "Emma" contains one occurence of *aardvark* and two occurences of *aardwolf*, but no occurences of *zulu*, while "Alice in Wonderland" features surprisingly many Zulus and no aard-creatures. 

Now this is a pretty awful idea. First of all, we are using raw counts, so longer documents will result in much larger values. Even if we adjust for the length of document, the most frequent terms will likely be *the*, *be*, *to*, *of*, and the like, so all the vectors will end up pretty similar.


## Counting and discounting
What we need is a cunning way to discount unimportant terms and bump up relevant ones. Intuitively, a term is relevant to a document if it occurs frequently in it, but is rare across all documents. Slightly more formally, given a collection of documents $$D$$, we would like to assign to each term $$t$$ in document $$d \in D$$ a weight $$w_{t, d}$$ that is:

* proportional to the frequency of $$t$$ in $$d$$,
* inversely proportional to the percentage of documents in D containing $$t$$.

Just by formulating the problem, we accidentaly discovered a weighting scheme aptly named *term frequency-inverse document frequency*:

$$ \text{tf-idf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D) $$

In the most popular variant, term frequency is the count of term $$t$$ in document $$d$$ divided by the length of the document:

$$ \text{tf}(t, d)=\frac{n_{t, d}}{\sum_{t' \in d}n_{t', d}}$$

Inverse document frequency is usually calculated as the logarithm of the total number of documents divided by the number of documents containing $$t​$$:

{% katex display %}

\text{idf}(t, D)=\log\frac{|D|}{|{d \in D: t \in d}|}

{% endkatex %}

The notation is a bit cryptic, so let's return to our example:

Say we want to find the tf-idf weights for all the terms in the second document. Let's start with term frequencies:

```python
from collections import Counter 
count = Counter(doc2.split())
terms = count.keys() 
total = sum(count.values())
tf = {t: count[t]/total for t in terms}
print(tf)

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

{% katex display %}
\text{idf}(t, D)=\log\left(1+\frac{|D|}{|{d \in D: t \in d}|}\right)
{% endkatex %}

## Building a search engine
```python
documents = [
  'the reticulated python is a species of python found in southeast asia and the longest snake in the world',
  'the burmese python is a large snake native to tropical southeast asia',
  'python is an interpreted high level programming language for general purpose programming',
  'the green anaconda is a non venomous snake species found in south america',
  'the yellow anaconda is a snake species endemic to south america',
  'anaconda is an open source distribution of the python and r programming languages'
]
```
## Wrapping it up
