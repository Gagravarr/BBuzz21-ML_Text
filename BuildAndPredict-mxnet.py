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
from mxnet import nd
from mxnet.contrib.text import embedding
import json

# Build a GloVe word embedding for our text
# This will take some time on the first run, as it downloads the
#  pre-trained embedding
# We use a smaller pre-trained model for speed, you may want to use 
#  the larger default by skipping the pretrained_file_name option
print("Loading GloVe embeddings")
glove = embedding.GloVe(pretrained_file_name='glove.6B.50d.txt')
print("GloVe loaded, contains %d terms" % len(glove))
print("")

# ----------------------------------------------------------------------------

# Test the embeddings

# For finding cosine-similar embeddings
def find_nearest(vectors, wanted, num):
    # 1e-9 factor is to avoid zero/negative numbers
    cos = nd.dot(vectors, wanted.reshape((-1,))) / (
            (nd.sum(vectors * vectors, axis=1) + 1e-9).sqrt() * 
            nd.sum(wanted * wanted).sqrt())
    top_n = nd.topk(cos, k=num, ret_typ='indices').asnumpy().astype('int32')
    return top_n, [cos[i].asscalar() for i in top_n]

# Looking up some similar words
def print_similar_tokens(query_token, num, embed):
    top_n, cos = find_nearest(embed.idx_to_vec,
                         embed.get_vecs_by_tokens([query_token]), num+1)
    print("Similar tokens to: %s" % query_token)
    for i, c in zip(top_n[1:], cos[1:]):  # Skip the word itself
        print(' - Cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))

print_similar_tokens("search", 3, glove)
print_similar_tokens("linux", 3, glove)
print_similar_tokens("gpl", 3, glove)
print("")

# ----------------------------------------------------------------------------

# Looking up word relationships, to verify the embeddings are working
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = find_nearest(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]  # Remove unknown words
def print_analogy(token_a, token_b, token_c, embed):
    anal = get_analogy('berlin','germany','paris', embed)
    print("The analogy for %s -> %s of %s is %s" % 
                                (token_a, token_b, token_c, anal))

print_analogy('berlin','germany','paris', glove)
print("")

# ----------------------------------------------------------------------------

# Load our talks
years = (2021,2020,2019,2018,2017,2016,2015)
talks = []
for year in years:
    filename = "data/%d/sessions.json" % year
    print("Loading %s" % filename)
    with open(filename) as f:
       d = json.load(f)
       for talk in d:
            talk["year"] = year
            talks.append(talk)

# For each talk title, what are the key words
#  based on the embedding?
# Project the title through the embedding space,
#  and see what words are by where we end up
# Just run for the first few talks to demo
for talk in talks[:15]:
    title = talk["title"]

    # Get the vectors for each word in the title
    # TODO Handle unknown words better
    # TODO Tokenize better
    title_vectors = glove.get_vecs_by_tokens(title.split(" "))
    num_words = title_vectors.shape[0]

    # Project the title text through the space, and renormalise
    # Is mean the best? Checking some research papers recommended...
    overall = title_vectors.mean(0)

    # What words are near there?
    nearby_count = 5
    topk, cos = find_nearest(glove.idx_to_vec, overall, nearby_count)
    nearby_words = [glove.idx_to_token[idx] for idx in topk]
    print(talk)
    print(nearby_words)
    print("")

# Convert the text into vectors using the embeddings
# Any words not known by the embedding are ignored
# For now, just use the titles
talk_titles = []
# TODO

# ----------------------------------------------------------------------------

# TSNE of talk titles
# TODO

# ----------------------------------------------------------------------------

# Now, you're ready to move onto more advanced things!
# Take a look at
#  ELMo - https://nlp.gluon.ai/examples/sentence_embedding/elmo_sentence_representation.html
#  BERT - https://nlp.gluon.ai/examples/sentence_embedding/bert.html
# And start reading some NLP scientific papers!
