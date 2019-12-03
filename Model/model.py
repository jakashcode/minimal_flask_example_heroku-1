import numpy as np
import pandas as pd
import requests
from flashtext.keyword import KeywordProcessor
from nltk.corpus import stopwords

# let's read in a couple of forum posts
forum_posts = pd.read_csv("input/ForumMessages.csv")

# get a smaller sub-set for playing around with
sample_posts = forum_posts.Message[0:3]

# get data from list of top 5000 pypi packages (last 30 days)
url = 'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json'
data = requests.get(url).json()

# get just the list of package names
list_of_packages = [data_item['project'] for data_item in data['rows']]


def create_keywordProcessor(list_of_terms, remove_stopwords=True,
                            custom_stopword_list=[""]):
    """ Creates a new flashtext KeywordProcessor and optionally
    does some lightweight text cleaning to remove stopwords, including
    any provided by the user.
    """
    # create a KeywordProcessor
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(list_of_terms)

    # remove English stopwords if requested
    if remove_stopwords == True:
        keyword_processor.remove_keywords_from_list(stopwords.words('english'))

    # remove custom stopwords
    keyword_processor.remove_keywords_from_list(custom_stopword_list)

    return (keyword_processor)


def apply_keywordProcessor(keywordProcessor, text, span_info=True):
    """ Applies an existing keywordProcessor to a given piece of text.
    Will return spans by default.
    """
    keywords_found = keywordProcessor.extract_keywords(text, span_info=span_info)
    return (keywords_found)


# create a keywordProcessor of python packages
py_package_keywordProcessor = create_keywordProcessor(list_of_packages,
                                                      custom_stopword_list=["kaggle", "http"])

# apply it to some sample posts (with apply_keywordProcessor function, omitting
# span information)
for post in sample_posts:
    text = apply_keywordProcessor(py_package_keywordProcessor, post, span_info=True)
    print(text)
import pickle
# save our file (make sure our file permissions are "wb",
# which will let us _w_rite a _b_inary file)
pickle.dump(py_package_keywordProcessor, open("processor.pkl", "wb"))
