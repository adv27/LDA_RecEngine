# Latent Dirichlet Allocation Based Recommender System
This is a prototype of a "Semantics-aware" recommender system built on top of LDA, Latent Dirichelet Allocation. A high-level workflow is described in this [**presentation**](http://www.slideshare.net/EasonChan2/latent-dirichlet-allocation-based-rs). This recommender system gives very nice results within a limited training time.


# Preparation work
First of all you need to web-crawl any news website or have a corpus of documents. Each document should be like the format below:

| Line        | Sample content           |  Require?  |
| ------------- |:-------------:|  ------:|
| 1      | http://www.bbc.com/news/entertainment-arts-31164553 |  Optional | 
| 2      | Baz Luhrmann to make Netflix musical      |  Required |
| 3 | Moulin Rouge director Baz Luhrmann is to make a 13-part musical series for streaming service Netflix.Set in 1970s New York, The Get Down traces how the city gave birth to disco, new wave and hip-hop.It will follow a group of teenagers from downtrodden, crime-ridden South Bronx who find solace in each other and their own musical abilities.Luhrmann will direct the first two episodes, echoing David Fincher's input to Netflix's hit series House Of Cards.      | Required |

A basic corpus of web-crawled BBC News is provided in the repo directory.


# Usage

Make sure you have **gensim** & **nltk** installed in your python path, this repo depeneds on these libraries.

* Build model
* 
```
python /path-to-repo/LDAModel_English.py --dir /path-to-repo/BBCNews
```
* Model building should take minutes to run, depends on how large your corpus is.
* After the model is generated and saved, now try to get some recommendations: 

```
python /path-to-repo/LDAModel_English.py --shell True
```

* You should see something like this image.
<img src=shell.png width=400/>


# Online Demo
Hee is a [**simple demo site**](http://54.183.251.139:8080/) I built upon using this repo and MEAN stack.

<br>