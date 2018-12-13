import numpy as np
from scipy.special import gammaln
import time

class BaseTopicModel(object):
    """
    Attributes
    ----------
    n_doc: int
        the number of total documents in the corpus
    n_voca: int
        the vocabulary size of the corpus
    verbose: boolean
        if True, print each iteration step while inference.
    """
    def __init__(self, n_doc, n_voca, **kwargs):
        self.n_doc = n_doc
        self.n_voca = n_voca
        self.verbose = kwargs.pop('verbose', True)

class BaseGibbsParamTopicModel(BaseTopicModel):
    """ Base class of parametric topic models with Gibbs sampling inference
    Attributes
    ----------
    n_topic: int
        a number of topics to be inferred through the Gibbs sampling
    TW: ndarray, shape (n_voca, n_topic)
        word-topic matrix, keeps the number of assigned word tokens for each word-topic pair
    DT: ndarray, shape (n_doc, n_topic)
        document-topic matrix, keeps the number of assigned word tokens for each document-topic pair
    sum_T: ndarray, shape (n_topic)
        number of word tokens assigned for each topic
    alpha: float
        symmetric parameter of Dirichlet prior for document-topic distribution
    beta: float
        symmetric parameter of Dirichlet prior for topic-word distribution
    """

    def __init__(self, n_doc, n_voca, n_topic, alpha, beta, **kwargs):
        super(BaseGibbsParamTopicModel, self).__init__(n_doc=n_doc, n_voca=n_voca, **kwargs)
        self.n_topic = n_topic
        self.TW = np.zeros([self.n_topic, self.n_voca])
        self.DT = np.zeros([self.n_doc, self.n_topic])
        self.sum_T = np.zeros(self.n_topic)

        self.alpha = alpha
        self.beta = beta

        self.topic_assignment = list()

        self.TW += self.beta
        self.sum_T += self.beta * self.n_voca
        self.DT += self.alpha


class ATM(BaseGibbsParamTopicModel):
    """Author Topic Model
    implementation of `The Author-Topic Model for Authors and Documents` by Rosen-Zvi, et al. (UAI 2004)
    Attributes
    ----------
    vocab:
        vocabulary list
    n_topic:
        number of topics
    n_author:
        number of authors
    alpha:
        author-topic distribution dirichlet parameter
    beta:
        word-topic distribution dirichlet parameter
    docList:
        list of documents, constructed based on the vocab
        format = list(list(words))
        ex) [[0,2,2,3],[1,3,3,4]]
            tokens of 1st document= 0,2,2,3 (note that 2 appears twice becase word 2 used twice in the first document)
    authorList:
        format = list(list(authors))
        at least one author should be exist for each document
        ex) [[0,1],[1,2]]
            authors of 1st doc = 0, 1
    """

    def __init__(self, n_doc, n_voca, n_topic, n_author, alpha=0.1, beta=0.01, **kwargs):
        super(ATM, self).__init__(n_doc, n_voca, n_topic, alpha, beta, **kwargs)
        self.n_author = n_author

        self.AT = np.zeros([self.n_author, self.n_topic]) + self.alpha
        self.topic_assigned = list()
        self.author_assigned = list()
        self.sum_A = np.zeros(self.n_author) + self.alpha * self.n_author

    def fit(self, docs, doc_authors, max_iter=100):
        if type(docs[0][0]) != int:
            _docs = list()
            for doc in docs:
                _doc = list()
                for word in doc:
                    doc.append(int(word))
                _docs.append(doc)
            docs = _docs

        if type(doc_authors[0][0]) != int:
            _doc_authors = list()
            for doc in doc_authors:
                _doc = list()
                for author in doc:
                    _doc.append(int(author))
                _doc_authors.append(_doc)
            doc_authors = _doc_authors

        self.random_init(docs, doc_authors)
        self.gibbs_sampling(docs, doc_authors, max_iter)

    def random_init(self, docs, doc_authors):
        for di in range(self.n_doc):
            self.author_assigned.append(list())
            self.topic_assigned.append(list())
            doc = docs[di]
            authors = doc_authors[di]
            for w in doc:
                # random sampling topic
                z = np.random.choice(self.n_topic, 1)[0]
                # random sampling author
                a = np.random.choice(len(authors), 1)[0]

                # assigning sampled value (sufficient statistics)
                self.TW[z, w] += 1
                self.AT[authors[a], z] += 1
                self.sum_T[z] += 1
                self.sum_A[authors[a]] += 1

                # keep sampled value for future sampling
                self.topic_assigned[di].append(z)
                self.author_assigned[di].append(authors[a])

    def gibbs_sampling(self, docs, doc_authors, max_iter):
        for iter in range(max_iter):
            tic = time.time()
            for di in range(len(docs)):
                doc = docs[di]
                authors = doc_authors[di]

                for wi in range(len(doc)):
                    w = doc[wi]
                    old_z = self.topic_assigned[di][wi]
                    old_a = self.author_assigned[di][wi]

                    self.TW[old_z, w] -= 1
                    self.AT[old_a, old_z] -= 1
                    self.sum_T[old_z] -= 1
                    self.sum_A[old_a] -= 1

                    wt = (self.TW[:, w] + self.beta) / (self.sum_T + self.n_voca * self.beta)
                    at = (self.AT[authors, :] + self.alpha) / (
                        self.sum_A[authors].repeat(self.n_topic).reshape(len(authors),
                                                                         self.n_topic) + self.n_topic * self.alpha)


                    pdf = at * wt
                    pdf = pdf.reshape(len(authors) * self.n_topic)
                    pdf = pdf / pdf.sum()

                    # sampling author and topic
                    idx = np.random.multinomial(1, pdf).argmax()
                    new_ai = int(idx / self.n_topic)
                    new_topic = idx % self.n_topic

                    new_author = authors[new_ai]
                    self.TW[new_topic, w] += 1
                    self.AT[new_author, new_topic] += 1
                    self.sum_T[new_topic] += 1
                    self.sum_A[new_author] += 1
                    self.topic_assigned[di][wi] = new_topic
                    self.author_assigned[di][wi] = new_author

            ll = self.log_likelihood()
            pp = self.perplexity(docs, doc_authors)
            print(' {} elapsed_time: {} log_likelihood: {}'.format(iter, time.time() - tic, ll))
            print('Perplexity',pp)


    def perplexity(self, docs, doc_authors):
        perpl=[]
        for di in range(len(docs)):
            doc = docs[di]
            author = doc_authors[di][0]
            P=1
            for wi in range(len(doc)):
                w = doc[wi]
                wt = (self.TW[:, w] + self.beta) / (self.sum_T + self.n_voca * self.beta)
                at = (self.AT[author, :] + self.alpha) / (self.sum_A[author].repeat(self.n_topic) + self.n_topic * self.alpha)
                P = P*(wt@at)
#             print(P)
#             print('111',-np.log(P))
#             print(len(doc))
            perpl.append(np.exp(-np.log(P)/len(doc)))
        return np.mean(perpl)
    def log_likelihood(self):
        ll = self.n_author * gammaln(self.alpha * self.n_topic)
        ll -= self.n_author * self.n_topic * gammaln(self.alpha)
        ll += self.n_topic * gammaln(self.beta * self.n_voca)
        ll -= self.n_topic * self.n_voca * gammaln(self.beta)

        for ai in range(self.n_author):
            ll += gammaln(self.AT[ai, :]).sum() - gammaln(self.AT[ai, :].sum())
        for ti in range(self.n_topic):
            ll += gammaln(self.TW[ti, :]).sum() - gammaln(self.TW[ti, :].sum())

        return ll

def get_top_words(topic_word_matrix, vocab, topic, n_words=20):
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)
    top_words = vocab[topic_word_matrix[topic].argsort()[::-1][:n_words]]
    return top_words


