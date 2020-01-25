# Detecting Sarcasm on Social Media Platforms : Reddit 

As the prominence of online platforms such as Reddit and Stack Overflow increases tremendously, the need to detect sarcasm, toxic culture and passivity is increasing as well. 
Detecting tone from written text is substantially more difficult than from conversations held in person. Lack of body language and tone makes it difficult to decipher the true message of the writer. 
For this project, our goal was to predict whether a given text is sarcastic or not using sentiment analysis. Sentiment analysis is a technique used to organize opinions expressed in textual context to determine the attitude of the author towards a certain topic. 
We used text from popular platform Reddit for sarcasm detection. 

## Getting Started

These instructions will get you a copy of our project file up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


Download the two .ipynb files and the train_balanced_sarcasm.csv file from the uploaded files on Collab:

Required Code/Files: 

1. train_balanced_sarcasm.csv
- required for the first file(Bayesian_final_part1.ipnb) 

2. Bayesian_final_part1.ipynb 
- This file will generate a csv file named "dfsub.csv" which is needed for the second .ipnb file (Bayesian_final_part2.ipnb )

3. Bayesian_final_part2.ipynb 


Please run in order part 1, and then part 2 and please note that all files must be in the same directory. 



### Prerequisites

What packages you need to install the software and how to install them


1. Required packages for the Bayesian_final_part1.ipnb 

```

Please see below for the code to install/run all the required python libraries and packages 

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import pymc3 as pm
import arviz as az
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

```

1. Required packages for the Bayesian_final_part2.ipnb 

```

Please see below for the code to install/run all the required python libraries and packages 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pymc3 as pm
import arviz as az
import os

from scipy import stats

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score



```



