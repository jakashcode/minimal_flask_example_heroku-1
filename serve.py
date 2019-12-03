from flashtext.keyword import KeywordProcessor
import pickle


# Function that takes loads in our pickled word processor
# and defines a function for using it. This makes it easy
# to do these steps together when serving our model.
def get_keywords_api():
    # read in pickled word processor. You could also load in
    # other models as this step.
    keyword_processor = pickle.load(open("Model/processor.pkl", "rb"))

    # Function to apply our model & extract keywords from a 
    # provided bit of text
    def keywords_api(keywordProcessor, text, span_info=True):
        keywords_found = keywordProcessor.extract_keywords(text, span_info=True)
        return keywords_found

    # return the function we just defined
    return keywords_api