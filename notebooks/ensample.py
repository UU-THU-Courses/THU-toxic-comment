# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
sub1 = pd.read_csv('/kaggle/input/toxic-comments/submission_1_.876.csv')
sub2 = pd.read_csv('/kaggle/input/toxic-comments/submission_2_.8668.csv')
sub3 = pd.read_csv('/kaggle/input/toxic-comments/submission_3_.8835.csv')
sub4 = pd.read_csv('/kaggle/input/toxic-comments/submission_4_.8236.csv')






sub['toxic'] = (0.876*sub1['toxic'] + 0.8668*sub2['toxic'] + 0.8835*sub3['toxic'] + 0.8236*sub4['toxic']) / (0.876+0.8668+0.8853+0.8236) 
sub.to_csv('submission.csv', index=False)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session