#!/usr/bin/python3

#   Licensed to the Apache Software Foundation (ASF) under one
#   or more contributor license agreements.  See the NOTICE file
#   distributed with this work for additional information
#   regarding copyright ownership.  The ASF licenses this file
#   to you under the Apache License, Version 2.0 (the
#   "License"); you may not use this file except in compliance
#   with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing,
#   software distributed under the License is distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#   KIND, either express or implied.  See the License for the
#   specific language governing permissions and limitations
#   under the License.


# ----------------------------------------------------------------------------

# Setup step - load all our libraries
# These are chosen for speed of development and understanding, not performance!

import json
import pickle
import csv
from collections import namedtuple, defaultdict

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics

import matplotlib.pyplot as plt

# Try to import things for Notebook display/rendering
try:
    from IPython.display import display, HTML
    notebook = True
except ImportError:
    notebook = False

# ----------------------------------------------------------------------------

# Read in the JSON files of the talk abstracts
# Convert these into Panda DataFrames
# If we didn't need to tweak things, could be done with pandas.read_json

keys = ("year","title","level","track","speaker","url","abstract")
years = (2021,2020,2019,2018,2017,2016,2015)

data = {}
for k in keys:
    data[k] = []
for year in years:
    filename = "data/%d/sessions.json" % year
    print("Loading %s" % filename)
    
    with open(filename) as f:
       d = json.load(f)
       for talk in d:
            talk["year"] = year
            for k in keys:
                data[k].append( talk[k] )
                
print("Loaded %d talks from %d years" % (len(data["year"]),len(years)))
talks = pd.DataFrame.from_dict(data)

# Build an "learn" field, by combining the bits we want to learn on
talks["learn"] = talks["title"] + " " + talks["track"] + " " + talks["abstract"]

# ----------------------------------------------------------------------------

# At this point, you might want to explore the data a bit
talks.head(5)

# ----------------------------------------------------------------------------

# Build two different TF-IDF matricies
# One which is word based, one which is character based
# Do n-grams for both
#  - for words this permits pseudo-phrase-queries
#  - for characters this permits pseudo-stemming and typo fixing
# Build these over just the "learn" combined column
# Then generate an inter-document similarity
TFIDF = namedtuple('TFIDF', 'tfidf matrix similarities settings')

tf_settings_word = dict( analyzer="word",    ngram_range=(1,2) )
tf_settings_char = dict( analyzer="char_wb", ngram_range=(3,4) )
tf_settings_base = dict( sublinear_tf=True, min_df=0, stop_words='english' )

def build_tfidf(tf_settings, text_to_process):
    print("Building TFI for: %s" % tf_settings)
    tfs = dict(tf_settings)
    tfs.update(tf_settings_base)

    # Build the TF-IDF over all the talks
    # Use title + category + abstract for our text
    tfidf = TfidfVectorizer(**tfs)
    tfidf_matrix = tfidf.fit_transform(text_to_process)
    tfidf_matrix.shape
    print(" - TF-IDF has %d entries" % tfidf_matrix.shape[1])
    
    # Build the similarities of each talk against every other talk
    # We'll use this for scoring
    tfidf_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Return this bundle ready for re-use
    return TFIDF(tfidf, tfidf_matrix, tfidf_similarities, tfs)

tfidf_word = build_tfidf(tf_settings_word, talks["learn"])
tfidf_char = build_tfidf(tf_settings_char, talks["learn"])

# Let's see what some of our terms are
# Check alphabetically
print(tfidf_word.tfidf.get_feature_names()[:10])
print(tfidf_char.tfidf.get_feature_names()[:10])

# TODO Find the highest values in the TF-IDF matrix, and show
#  the terms at those indexes

# ----------------------------------------------------------------------------

# Now, build a ML model for each
models = []
for tfidf in (tfidf_word, tfidf_char):
    print("Building model for: %s" % tfidf.settings)
    
    # Build a model, using Multinomial Naive Bayse
    # Model the text of the talk, to predict the talk's index
    classifier = MultinomialNB()
    model = make_pipeline(tfidf.tfidf, classifier)
    learn_text = talks["learn"]
    model.fit( list(learn_text), list(learn_text.index) )

    # Save for later predictions
    models.append({
        "model": model,
        "tfidf": tfidf,
        "classifier": classifier
    })
    print(" - model built!")

# ----------------------------------------------------------------------------

# Code to recommend talks based on a query
# Will return the indexes of the talks, ranked
def recommend(query, model_dict, max_hits=10):
    # Ask the model to compare our query against every talk,
    #  then pick the talk it thinks is the most similar
    pred_idx = model_dict["model"].predict([query])

    # The prediction should be the index of that talk
    print("Best match - talk %d" % pred_idx)
    print(talks.iloc[pred_idx,:]["title"])
    
    # Get the pairwise similarity scores of all other talks with that one
    # Filter for ones high enough, and sort so highest scores come first
    similarities = model_dict["tfidf"].similarities[pred_idx]
    sim_scores = list( [idx,s] for idx,s in enumerate(similarities[0]) if s > 0.01 )
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indexes and scores of the x most similar talks
    sim_scores = sim_scores[0:max_hits]
    
    # Grab those talks
    indexes = [i[0] for i in sim_scores]
    return talks.iloc[indexes]

# ----------------------------------------------------------------------------

# Pretty-print our DataFrames
def render(df):
    if notebook:
        display(HTML(df.to_html()))
    else:
        print(df)

# Let's try it!
queries = ("apache tika", "ngram", "nlp", "storm spark", "bm25")
for q in queries:
    print("")
    print(q)
    print("")
    render( recommend(q, models[0],5) )
    print("")
    render( recommend(q, models[1],5) )
    print("")

# ----------------------------------------------------------------------------

# K-Means clustering
# 
# Using the word-based TF-IDF, figure out what's the optimal number of
#  clusters to reduce our 20k-30k terms down into
#
# Note - we can't use lots of the scoring (eg completeness or homogeneity
#  or V-measure as we don't have a ground-truth)

KM = namedtuple('KM', 'k km scoff tfidf')

# Try a range of k-sizes
# A wide range is slow but lets us check what's best
# If we know what's almost right, a narrow range is fine!
krange = range(51, 52)
#krange = range(25, talks.shape[0]-25)

# Try each one with a few different inits, to avoid getting stuck in local-minima
kms = []
for k in krange:
    tfidf = tfidf_word

    print("Building k-means cluster of size %d" % k)
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, 
                n_init=15, verbose=False)
    km.fit(tfidf.matrix)
    # TODO Do we need to include labels / talk indexes?

    scoff = metrics.silhouette_score(tfidf.matrix, km.labels_, sample_size=1000)
    print(" - Silhouette Coefficient: %0.4f" % scoff)

    kms.append( KM(k,km,scoff,tfidf) )

# Save all the scores, so we can draw a graph
with open("outputs/cluster-scores.csv","w") as c:
   cw = csv.writer(c)
   cw.writerow(("k","Silhouette Coefficient"))
   for k in kms:
      cw.writerow((k.k, k.scoff))

# ----------------------------------------------------------------------------

# Visualise the Silhouette Coefficient vs K
fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(111)
ax.plot([k.k for k in kms], [k.scoff for k in kms])
ax.set(title="Silhouette Coefficient vs K", xlabel="Cluster Size (K)")
if len(kms) > 3:
   plt.show()

# ----------------------------------------------------------------------------

# Which one had the best Silhouette Coefficient?
kms.sort(key=lambda x: x.scoff, reverse=True)
best_km = kms[0]
print("")
print("Best K-Means found with a cluster-size (k) of %d" % best_km.k)
print("That had a Silhouette Coefficient: %0.4f" % best_km.scoff)

# ----------------------------------------------------------------------------

# What does our "Best Clustering" look like?
# What ended up where?

# Report some top features for a few clusters
def cluster_terms(km, top_clusters=4):
    # Work out the middle of each cluster
    order_centroids = km.km.cluster_centers_.argsort()[:, ::-1]

    # Print the top 10 features for the first few clusters
    terms = km.tfidf.tfidf.get_feature_names()
    for i in range(top_clusters):
        print(" - Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print('  %s ' % terms[ind], end='')
        print()
    print()
cluster_terms(best_km)

# If you want to visualise the top features of the clusters, rather
# than our simple printing them out, see
# https://buhrmann.github.io/tfidf-analysis.html

# ----------------------------------------------------------------------------

# Predict the best talks for a query based on the clusters

# Identify which cluster each talk belongs to
talk_clusters = best_km.km.predict(best_km.tfidf.matrix)
cluster_talks = defaultdict(list)
for talk_id, cluster_id in enumerate(talk_clusters):
    cluster_talks[cluster_id].append(talk_id)

#  - Work out which cluster our query best maps to
#  - Identify the talks in that cluster
#  - Find the centre of the cluster
#  - Do a regular similarity to work out the "closest" to the centre
def recommend_km(query, km, max_hits=10):
    query_tf = km.tfidf.tfidf.transform([query])
    cluster_id = int(km.km.predict(query_tf))
    print("Best cluster for '%s' is %d" % (query,cluster_id))
    
    c_centre_tfidf = km.km.cluster_centers_[cluster_id]
    c_talk_ids = cluster_talks[cluster_id]
    print(" - That cluster contains talks: %s" % c_talk_ids)

    # Compare all talks with this cluster centre
    similarities = linear_kernel(c_centre_tfidf.reshape(1, -1), km.tfidf.matrix)

    # Order, with a boost for ones in the cluster
    tscores = [[i,s] for i,s in enumerate(similarities[0])]
    for idx, score in tscores:
        if idx in c_talk_ids:
            tscores[idx][1] = score+0.01
    tscores = sorted(tscores, key=lambda x: x[1], reverse=True)

    # Get the indexes and scores of the x most similar talks
    tscores = tscores[0:max_hits]
    
    # Grab those talks
    indexes = [i[0] for i in tscores]
    return talks.iloc[indexes]

# ----------------------------------------------------------------------------

# Let's try it!
for q in queries:
    print("")
    print(q)
    print("")
    render( recommend_km(q, best_km, 10) )
    print("")

# ----------------------------------------------------------------------------

# Try to visualise the ~50 clusters in 2 dimensions

def make_tsne():
   tsne_init = 'pca'  # could also be 'random'
   tsne_perplexity = 20.0
   tsne_early_exaggeration = 4.0
   tsne_learning_rate = 1000
   random_state = 1
   return TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
               early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

def visualise_km_tsne(centroids):
   model = make_tsne()
   transformed_centroids = model.fit_transform(centroids)
   fig = plt.figure(figsize=(15,9))
   plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], marker='x')
   plt.show()

visualise_km_tsne(best_km.km.cluster_centers_)

# ----------------------------------------------------------------------------

# How does that clustering compare to the original TF-IDF data?
def visualise_tfidf_tsne(tfidf_matrix):
   # Build a TSVD reducer, to get us down to ~50 dimensions
   reducer = TruncatedSVD(n_components=50, random_state=0)
   print(tfidf_matrix.shape)
   tfidf_reduced = reducer.fit_transform(tfidf_matrix)
   print(tfidf_reduced.shape)

   # Use t-SNE to get down to 2 dimensions
   tfidf_emb = make_tsne().fit_transform(tfidf_reduced)

   # Plot it
   fig = plt.figure(figsize=(15, 10))
   ax = plt.axes(frameon=False)
   ax.set(title="t-SNE visualisation of raw TF-IDF")
   plt.setp(ax, xticks=(), yticks=())
   plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                       wspace=0.0, hspace=0.0)
   plt.scatter(tfidf_emb[:, 0], tfidf_emb[:, 1], c=talk_clusters, marker="x")
   plt.show()

visualise_tfidf_tsne(tfidf_word.matrix)

# ----------------------------------------------------------------------------

# TODO Try using OPICS (Ordering Points To Identify the Clustering Structure)
#  to try to identify talk clusters
# Will need manual review to help identify the hyperparameters, especially
#  around the cluster difference
# Once clusters, then pull out the top words for each cluster, identify
#  likely talks from clusters etc
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

# ----------------------------------------------------------------------------

# While we could go further with SciKitLearn, it's time to 
#  swap over to Apache MXNet!
