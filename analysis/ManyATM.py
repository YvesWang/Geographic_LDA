
# coding: utf-8

# In[1]:


import pandas as pd

import json
from pprint import pprint

#Use gensim, which remove stop words, short words(<3 letters), puncts and do stemming
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import preprocess_string


from collections import Counter
import numpy as np


from gensim.models import AuthorTopicModel
from gensim.corpora import mmcorpus
from gensim.test.utils import common_dictionary, datapath, temporary_file

from gensim import corpora

import pickle
import logging

import numpy as np
import matplotlib
import json
import pickle as pkl
from matplotlib import pyplot as plt



# In[2]:


from lib.ATM import ATM,get_top_words
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[3]:


def NN(good_topics,target):
    scores = normalized_AT[:,good_topics] @ target

    lda_pred=[]
    ranked_pairs = sorted([(i,scores[i]) for i in range(len(scores))],key=lambda pair:pair[1],reverse=True)
    for pair in ranked_pairs:
#         print(author_name[pair[0]])
        lda_pred.append(author_name[pair[0]].lower())
    return lda_pred

def log_reg(good_topics):
    X_red=normalized_AT[red_states_ids,:][:,good_topics]
    Y_red=np.zeros((X_red.shape[0]))+1

    X_blue=normalized_AT[blue_states_ids,:][:,good_topics]
    Y_blue=np.zeros((X_blue.shape[0]))

    Y=np.concatenate((Y_red,Y_blue),axis=0)

    X=np.concatenate((X_red,X_blue),axis=0)

    cl = RandomForestClassifier(n_estimators=1000, max_depth=5,random_state=0)
#     cl = LogisticRegression(C=1e5, solver='lbfgs')

    cl.fit(X, Y)
    score_rank=sorted(list(zip(author_name,cl.predict_proba(normalized_AT[:,good_topics])[:,1])),key=lambda x: x[1],reverse=True)
    lda_pred=[]
    for pair in score_rank:
        lda_pred.append(author_name[pair[0]].lower())
    return lda_pred



def simple_rank_loss(new_truth,lda_pred):
    swing_truth=[]
    for state in new_truth:
        if state in swing_states:
            swing_truth.append(state)

    swing_pred=[]
    for state in lda_pred:
        if state in swing_states:
            swing_pred.append(state)

    new_truth=swing_truth
    lda_pred=swing_pred

    truth_rank={}
    n=0
    for state in new_truth:
        truth_rank[state]=n
        n+=1


    truth_binary_rank={}
    for k in range(len(new_truth)):
        for j in range(len(new_truth)):
            if truth_rank[new_truth[k]]<truth_rank[new_truth[j]]:
                truth_binary_rank[(new_truth[k],new_truth[j])]=1
            else:
                truth_binary_rank[(new_truth[k],new_truth[j])]=0


    pred_rank={}
    n=0
    for state in lda_pred:
        pred_rank[state]=n
        n+=1


    pred_binary_rank={}
    for k in range(len(lda_pred)):
        for j in range(len(lda_pred)):
            if pred_rank[lda_pred[k]]<pred_rank[lda_pred[j]]:
                pred_binary_rank[(lda_pred[k],lda_pred[j])]=1
            else:
                pred_binary_rank[(lda_pred[k],lda_pred[j])]=0

    assert len(truth_binary_rank.values()) == len(pred_binary_rank.values())

    correct=0
    for key in pred_binary_rank.keys():
        if pred_binary_rank[key] == truth_binary_rank[key]:
            correct+=1

    return correct/len(pred_binary_rank.keys())





with open('states.pkl','rb') as file:
    states=pkl.load(file)
for pair in states.items():
    states[pair[0]]=pair[1].lower()


# In[16]:


with open('alldebates.pkl','rb') as file:
    odf=pkl.load(file)



# In[21]:


odf.state=list(map(lambda x: x.lower(),odf.state))


# In[18]:




# ## Filter and extract

# In[122]:
ITER=0
for sample_split in [10,10,10,10,10,5,5,5,5,5,4,4,4,4,4,3,3,3,3,3,2,2,2,2,2,1,1,1,1,1]:
    print(ITER)
    ITER+=1

    odf=odf.sample(len(odf) // sample_split ,replace=False)


    # In[22]:


    number_uid=(odf.groupby('uid').count()['ID'])
    good_uid=set(number_uid[number_uid<5].index.values)
    odf=odf[odf.uid.isin(good_uid)]


    # In[23]:


    with open('truth.pkl','rb') as file:
        truth=pkl.load(file)
    with open('web_pred.pkl','rb') as file:
        web_pred=pkl.load(file)


    # In[24]:


    #drop states with less than 300 tweets
    #number_tweets=odf.groupby('state').count()['ID']
    good_states=set(truth)

    odf=odf[odf.state.isin(good_states)]


    # In[28]:


    assert len(np.unique(odf.state)) == len(good_states)


    # In[29]:


    new_truth=truth
    nre_wpred=web_pred


    # In[30]:


    # Tokenize tweets

    documents=odf['text'].values

    from gensim.parsing.preprocessing import *
    CUSTOM_FILTERS = [lambda x: x.lower(),strip_multiple_whitespaces,remove_stopwords, strip_non_alphanum,strip_numeric,lambda x: strip_short(x,minsize=3)]

    filtered_docs=list(map(lambda t:preprocess_string(t, CUSTOM_FILTERS),documents))

    texts=filtered_docs

    from collections import defaultdict
    frequency = Counter()

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if (frequency[token] > 2) and (frequency[token]<200)] for text in texts]


    # In[31]:


    #add filtered text to odf, but remove empty ones
    odf['filtered']=texts
    odf['filtered_len']=list(map(lambda x:len(x),texts))
    odf_c=odf[odf['filtered_len']>0]
    texts=odf_c['filtered'].values


    # In[37]:



    n=0
    author2doc={}
    for author in states.values():
        author2doc[author]=[]

    for author in odf_c['state'].values:
        author2doc[author].append(int(n))
        n+=1

    author_name={}
    author2id={}
    nstates=np.unique(odf_c['state'].values)
    for i in range(len(nstates)):
        author_name[i]=nstates[i]
        author2id[nstates[i]]=i


    # In[38]:


    import os
    dictionary = corpora.Dictionary(texts)
    dictionary.save(os.path.join('temp.dict'))
    #indexize words and tweets
    corpus = [dictionary.doc2bow(text) for text in texts]

    #generate author2doc dic
    n=0
    author2doc={}
    for author in states.values():
        author2doc[author]=[]

    for author in odf_c['state'].values:
        author2doc[author].append(int(n))
        n+=1


    # In[43]:


    #indexize states
    author_name={}
    author2id={}
    nstates=np.unique(odf_c['state'].values)
    for i in range(len(nstates)):
        author_name[i]=nstates[i]
        author2id[nstates[i]]=i

    #input corpus for our ATM model
    new_corpus=[]
    for tweet in corpus:
        new_data=[]
        for pair in tweet:
            new_data+=[pair[0]]*pair[1]
        new_corpus.append(new_data)

    # A new word dic for our model input
    new_dic={}
    for key in dictionary.keys():
        new_dic[key]=dictionary[key]

    voca=[new_dic[i] for i in range(len(new_dic.keys()))]


    # In[45]:


    #generate docauthor list for our model input
    doc_author=np.arange(len(new_corpus))

    for pair in author2doc.items():
    #     print(pair)
        for doc_id in pair[1]:
            doc_author[doc_id]=author2id[pair[0]]

    doc_author=[[i] for i in doc_author]


    # In[48]:


    #model hyperparameters
    n_doc = len(corpus)
    n_topic = 300
    n_author = len(author_name)
    n_voca = len(voca)
    max_iter = 1


    # In[50]:


    model = ATM(n_doc, n_voca, n_topic, n_author)
    model.fit(new_corpus, doc_author, max_iter=max_iter)


    # In[51]:


    #run
    normalized_AT=[]
    for author_id in range(len(author_name.values())):
        normalized_AT.append((model.AT[author_id,:]/np.sum(model.AT[author_id,:])))

    normalized_AT=np.array(normalized_AT)


    # In[52]:


    #run
    topic_stds=[]
    for topic in range(n_topic):
        topic_stds.append(np.std(normalized_AT[:,topic]))


    # # Bootstrap

    # In[54]:


    with open('three_parts.pkl','rb') as file:
        three_parts=pkl.load(file)


    # In[55]:


    red_states,swing_states,blue_states=three_parts


    # In[56]:


    author2id_lower={}
    for pair in author2id.items():
        author2id_lower[pair[0].lower()]=pair[1]


    # In[58]:


    red_states_ids=[]
    for i in red_states:
        if i in author2id_lower:
            red_states_ids.append(author2id_lower[i])


    # In[59]:


    blue_states_ids=[]
    for i in blue_states:
        if i in author2id_lower:
            blue_states_ids.append(author2id_lower[i])


    # In[61]:


    topic_means=[]
    for topic in range(n_topic):
        topic_means.append(np.mean(normalized_AT[:,topic]))


    # In[62]:


    red_means=[]
    for t in range(n_topic):
        red_means.append(np.mean(normalized_AT[red_states_ids,t]))


    # In[63]:


    blue_means=[]
    for t in range(n_topic):
        blue_means.append(np.mean(normalized_AT[blue_states_ids,t]))


    # In[64]:


    std_cut=0.05
    interest_cross_blue=[]
    for n in np.arange(len(blue_means)):
        if (blue_means[n]>0.001) and (blue_means[n]>topic_means[n]+std_cut*topic_stds[n]) and (red_means[n]>0.001) and (red_means[n]<topic_means[n]-std_cut*topic_stds[n]):
            interest_cross_blue.append(n)


    # In[66]:


    interest_cross_red=[]
    for n in np.arange(len(red_means)):
        if (blue_means[n]>0.001) and (blue_means[n]<topic_means[n]-std_cut*topic_stds[n]) and (red_means[n]>0.001) and (red_means[n]>topic_means[n]+std_cut*topic_stds[n]):
            interest_cross_red.append(n)


    # In[68]:


    good_topics_all=np.arange(normalized_AT.shape[1])

    good_topics_union=np.array(list(set(interest_cross_blue).union(set(interest_cross_red))))

    good_topics_blue=interest_cross_blue

    good_topics_red=interest_cross_red


    # In[ ]:


    #n_doc,n_topic,n_vocab,max_iter,'rf all topics ','NN red all topics ','NN blue all topics ','rf union topics ',
    #'NN red union topics ','NN blue union topics ','rf red topics ','NN red red topics ','NN blue red topics '
    #'rf blue topics ','NN red blue topics ','NN blue blue topics '


    # In[80]:


    temp_param=[n_doc,n_topic,n_voca,max_iter]


    # In[77]:


    temp_res=[]

    topics=good_topics_all
    lda_pred=log_reg(topics)
    target_red=np.array(red_means)[topics]
    target_blue=np.array(blue_means)[topics]
    print('rf all topics ', simple_rank_loss(new_truth,lda_pred))
    temp_res.append(simple_rank_loss(new_truth,lda_pred))
    print('NN red all topics ', simple_rank_loss(new_truth,NN(topics,target_red)))
    temp_res.append( simple_rank_loss(new_truth,NN(topics,target_red)))
    print('NN blue all topics ', simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))
    temp_res.append(simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))

    topics=good_topics_union
    lda_pred=log_reg(topics)
    target_red=np.array(red_means)[topics]
    target_blue=np.array(blue_means)[topics]
    print('rf union topics ', simple_rank_loss(new_truth,lda_pred))
    temp_res.append(simple_rank_loss(new_truth,lda_pred))
    print('NN red union topics ', simple_rank_loss(new_truth,NN(topics,target_red)))
    temp_res.append( simple_rank_loss(new_truth,NN(topics,target_red)))
    print('NN blue union topics ', simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))
    temp_res.append(simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))

    topics=good_topics_red
    lda_pred=log_reg(topics)
    target_red=np.array(red_means)[topics]
    target_blue=np.array(blue_means)[topics]
    print('rf red topics ', simple_rank_loss(new_truth,lda_pred))
    temp_res.append(simple_rank_loss(new_truth,lda_pred))
    print('NN red red topics ', simple_rank_loss(new_truth,NN(topics,target_red)))
    temp_res.append( simple_rank_loss(new_truth,NN(topics,target_red)))
    print('NN blue red topics ', simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))
    temp_res.append(simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))

    topics=good_topics_blue
    lda_pred=log_reg(topics)
    target_red=np.array(red_means)[topics]
    target_blue=np.array(blue_means)[topics]
    print('rf blue topics ', simple_rank_loss(new_truth,lda_pred))
    temp_res.append(simple_rank_loss(new_truth,lda_pred))
    print('NN red blue topics ', simple_rank_loss(new_truth,NN(topics,target_red)))
    temp_res.append( simple_rank_loss(new_truth,NN(topics,target_red)))
    print('NN blue blue topics ', simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))
    temp_res.append(simple_rank_loss(new_truth,NN(topics,target_blue)[::-1]))


    # In[116]:


    this_res=temp_param+temp_res
    with open('ATM_results.pkl','rb') as file:
        last_res=pkl.load(file)


    # In[115]:


    last_res.append(this_res)
    with open('ATM_results.pkl','wb') as file:
        pkl.dump(last_res,file)


# In[72]:



# In[1300]:



# Bootstrap ends

# In[70]:
