# Reddit Mental Health during COVID-19 Pandemic

<h2>Documentation:</h2>

This work is part of a Colab Notebook, so you will need to upload the training_utils.py and dataset to a folder in Google Drive. 

First you will need to mount your Drive. Please refer to the <a href="https://colab.research.google.com/drive/1srw_HFWQ2SMgmWIawucXfusGzrj1_U0q">example notebook</a> provided by Google.

Make sure to set up a path the aforementioned folder like so:

```
sys.path.append('/content/drive/My Drive/reddit_mental_health')
```

You may have to perform some installs.
```
 may have to !pip install transformers
# and then restart notebook

!pip install transformers
```

Warnings were ignored as they clouded readability of some output. 
```
# ignore warnings
warnings.filterwarnings("ignore")
```

<h4>📁 source</h4>

The source code is contained within a Colab notebook which can also be found at <a href="https://colab.research.google.com/drive/1SkrRq0HywnwgoKh-Lzs9qq5UoNL5AdDW?usp=sharing"> this link</a>.


<h2>Other Folders:</h2>
<h4>📁 results</h4>


This folder contains comparison plots and the metrics of the training.


<h4>📁 datasets</h4>

Dataset was collected and used within the paper, <i> How dramatic events can affect emotionality in social posting: the impact of COVID-19 on Reddit</i> by Valerio Basile, Francesco Cauteruccio and Giorgio Terracina. It is available at <a href="https://bitbucket.org/cauteruccio/reddit-dataset/src/master/">this link</a>.
