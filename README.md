# Taking AI/ML to the next level - Text!
## Code, Data and Slides for my Berlin Buzzwords 2021 Talk

Berlin Buzzwords 2021, virtual Kesselhaus, 17.06.2021 17:30 – 18:00 CET

https://2021.berlinbuzzwords.de/session/taking-aiml-next-level-text

### Code
The sample / example code, in Python, for the areas covered in the talk. 

These should go into a lot more detail than I had time for in the talk, 
which was limited to 30 minutes! 

The code needs Python 3, but should run fine both in normal Python
and IPython / Jupyter.

#### SkiKitLearn

`BuildAndPredict-SciKitLearn.py` used SciKitLearn to do "simpler"
Machine Learning operations on text. It builds TF-IDFs, classifies and
clusters the talks, recommends based on the different models, and
visualises the models. None of the models are amazing, but they do
work fairly well, and this should guide you through the key concepts.

#### Apache MXNet

`BuildAndPredict-mxnet.py` uses Apache MXNet to build Word Embeddings
(based on GloVe and some pre-trained data taken from Wikipedia), then
uses that for inference, clustering and recommendation. The Word 
Embeddings are used to find words similar to a given talk, then then talks
similar to a given talk.

Extending the mxnet code to work with BERT and ELMO is left as an 
exercise for the reader... You'll want to start at 
https://mxnet.apache.org/versions/1.7/api/python/docs/tutorials/packages/gluon/text/index.html

Ideally you would use a Word Embedding that includes all the words for
your data. We use a cut-down set to keep the size reasonable + speed fast,
but this means some matching is poor because the specialist words used
aren't known about.

### Sample Data - BBuzz Talks
Data on all Berlin Buzzwords talks from 2015-2021, extracted into JSON, can
be found in `data/` indexed by year. The BeautifulSoup powered Python
scripts for the web scraping can also be found there.

Talk data is copyright Newthinking, Plain Schwarz and the Speakers

### Slides
Slides as PDF, once finished...
