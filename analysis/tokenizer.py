
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# lowercase and remove punctuation
def tokenize(sents):
    token_list = []
    tokenizer = TweetTokenizer()
    punctuations = string.punctuation
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        token_list.append([token.lower() for token in tokens if (token not in punctuations) and (token not in stopwords.words('english'))])
    return token_list
#string.to.lower()
