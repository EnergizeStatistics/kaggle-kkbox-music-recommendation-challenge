# # Music Recommendation System #

# ## Load Packages ##

# process & visualize data 
import numpy as np
import pandas as pd
import itertools
import gc
import pickle
import seaborn as sns

# split dataset
from sklearn.model_selection import train_test_split

# natural language processing 
from gensim.models import phrases, word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk import word_tokenize
import re

# sparse matrix
import scipy.io
import scipy.sparse

# feature construction
from sklearn.feature_extraction import DictVectorizer

# recommendation system
from lightfm import LightFM

# deployment
from flask import current_app

np.random.seed(0)


# ## Load Data ##
data_dir = "/home/lee/Documents/DatasetsForGitHub/kaggle_kkbox_music_recommendation_challenge/"

df_train = pd.read_csv(data_dir + "train.csv")
df_songs = pd.read_csv(data_dir + 'songs.csv')
df_members = pd.read_csv(data_dir + 'members.csv')
df_test = pd.read_csv(data_dir + "test.csv")

# download NLP files to working directory
nltk.download(['punkt'], download_dir=data_dir)
nltk.data.path.append(data_dir)


# ## Construct Features ##

# **Prepare for tokenization**

# Still using the above example "周杰倫 (Jay Chou)", one option is to keep both versions of the name for better matching, meaning that this artist's name becomes "周杰倫", "Jay", "Chou" after tokenization. Another option is to remove all text in parentheses.
# 
# Fill missing artists. Replace special characters.

def fill_person(df, col):
    df[col] = df_songs[col].fillna('Unknown')
    df[col].replace(to_replace=r'[|/\\;\(\)\"&.\+]', regex=True, value=' ', inplace=True)
    df[col].str.strip()
    
    return df.copy()

df_songs = fill_person(df_songs, 'artist_name')
df_songs = fill_person(df_songs, 'composer')
df_songs = fill_person(df_songs, 'lyricist')


# **Tokenize names**
# 
# Chinese and English are separated in the tokenizing process. Inspired by [this answer on stackoverflow](https://stackoverflow.com/questions/30425877/how-to-not-split-english-into-separate-letters-in-the-stanford-chinese-parser)

# In[ ]:


pat = re.compile("[A-Za-z]+")

def tokenize_names(sentence):
    sent_tokens = []
    prev_end = 0
    sentence = sentence.lower()
    for match in re.finditer(pat, sentence):
        chinese_part = sentence[prev_end:match.start(0)]
        sent_tokens += nltk.word_tokenize(chinese_part)
        sent_tokens.append(match.group(0))
        prev_end = match.end(0)
    last_chinese_part = sentence[prev_end:]
    sent_tokens += nltk.word_tokenize(last_chinese_part)
    
    return sent_tokens


# In[25]:


df_songs['artist_token'] = df_songs['artist_name'].dropna().apply(lambda row: tokenize_names(row))
df_songs['composer_token'] = df_songs['composer'].dropna().apply(lambda row: tokenize_names(row))
df_songs['lyricist_token'] = df_songs['lyricist'].dropna().apply(lambda row: tokenize_names(row))


# **Embed names**

# In[26]:


artist_song_id_lst = df_songs[df_songs['artist_token'].notnull()]['song_id'].tolist()
composer_song_id_lst = df_songs[df_songs['composer_token'].notnull()]['song_id'].tolist()
lyricist_song_id_lst = df_songs[df_songs['lyricist_token'].notnull()]['song_id'].tolist()

def convert_to_tagged_doc(df, col, id_lst):
    lst = df[col].dropna().tolist()
    name_dict = dict(zip(id_lst, lst))
    tagged_name = [TaggedDocument(words=word, tags=[tag]) for tag, word in name_dict.items()]
    return tagged_name


# In[27]:


tagged_artist = convert_to_tagged_doc(df_songs, 'artist_token', artist_song_id_lst)
tagged_composer = convert_to_tagged_doc(df_songs, 'composer_token', composer_song_id_lst)
tagged_lyricist = convert_to_tagged_doc(df_songs, 'lyricist_token', lyricist_song_id_lst)


# In[28]:


del artist_song_id_lst, composer_song_id_lst, lyricist_song_id_lst


# **Train `doc2vec` models**

# In[30]:


# parameters, increase epochs if underfitting
max_epochs = 10
vec_size = 2**5
alpha = 0.025


# In[34]:


def train_d2v(tagged_data):
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('epoch {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
#         # decrease the learning rate
#         model.alpha -= 0.0002
#         # fix the learning rate, no decay
#         model.min_alpha = model.alpha
    
    return model


# **Serialize the models, save to disk**

# In[ ]:


model_artists = train_d2v(tagged_artist)
model_artists.save(data_dir + "model_artists.model")
model_composers = train_d2v(tagged_composer)
model_composers.save(data_dir + "model_composers.model")
model_lyricists = train_d2v(tagged_lyricist)
model_lyricists.save(data_dir + "model_lyricists.model")


# **Retrieve the models from disk**

# In[23]:


model_artists = Doc2Vec.load(data_dir + "model_artists.model")
model_composers = Doc2Vec.load(data_dir + "model_composers.model")
model_lyricists = Doc2Vec.load(data_dir + "model_lyricists.model")


# **Use trained `doc2vec` model to embed artist names**

# In[31]:


df_songs['artist_vector'] = df_songs['artist_token'].apply(model_artists.infer_vector, alpha=alpha)
df_songs['composer_vector'] = df_songs['composer_token'].apply(model_composers.infer_vector, alpha=alpha)
df_songs['lyricist_vector'] = df_songs['lyricist_token'].apply(model_lyricists.infer_vector, alpha=alpha)


# In[206]:


gc.collect()


# ### User Features ###

# **Create user feature matrix for LightFM**

# In[90]:


def binarize_function(val, val_low=0, val_up=100):
    """Function to binarize a given column in slices
    """
    if val_low <= val < val_up:
        return 1
    else:
        return 0
    
def build_user_feature_matrix(ulist):
    """
    ulist: member master list
    """
    # Segment the age
    ulist["0to30"] = ulist["bd"].apply(binarize_function, val_low=0, val_up=30)
    ulist["30to50"] = ulist["bd"].apply(binarize_function, val_low=30, val_up=50)
    ulist["50to120"] = ulist["bd"].apply(binarize_function, val_low=50, val_up=120)

    list_age_bin = ["0to30", "30to50", "50to120"]
    ulist = ulist[["msno",
                   "city",
                   "gender_fill",
                   "registered_via",
                   "account_length"] + list_age_bin]

    ulist = ulist.T.to_dict().values()
    vec = DictVectorizer(sparse=True)
    ulist = vec.fit_transform(ulist)
    
    ulist = scipy.sparse.csr_matrix(ulist, dtype=np.float32)
    
    return ulist


# In[91]:


user_feat_mtrx = build_user_feature_matrix(df_members)

# save the matrix
scipy.io.mmwrite(data_dir + "user_feat_mtrx.mtx", user_feat_mtrx)


# In[209]:


# double-check the generated matrix
print('user feature matrix has the same number of rows as the member master list: {}'      .format(assert(user_feat_mtrx.shape[0] == df_members.shape[0])))
       
print('# of elements that are NaN or Inf: {}'.format(np.sum(np.logical_not(np.isfinite(user_feat_mtrx.todense())))))


# ### Item Features ###

# **Split labeled training dataset to train and validation**

# In[ ]:


train_interact, val_interact = train_test_split(df_train[df_train['song_id'].isin(df_songs['song_id'])],                                                test_size=0.2, random_state=0)


# In[157]:


# remove songs not in master song metadata dataset
train_songs = df_songs[df_songs['song_id'].isin(train_interact['song_id'].unique())]
val_songs = df_songs[df_songs['song_id'].isin(val_interact['song_id'].unique())]
test_songs = df_songs[df_songs['song_id'].isin(df_test['song_id'].unique())]


# **Create item feature matrix for LightFM**

# In[101]:


def build_item_feature_matrix(cplall, cpltr, cplval, cplte):
    
    """ Build item feature matrix
    cplall: dataframe with all items
    cpltr: items in training set
    cplval: items in validation set
    cplte: items in testing set
    
    Note that cpltr, cplval, cplte can and very likely will overlap.
    
    """
    
    # 0 to 8 min, regular pop song
    cplall["song_length_0to8min"] = cplall["song_length"].apply(lambda x: 1 if x<=480000 else 0)
    cpltr["song_length_0to8min"] = cpltr["song_length"].apply(lambda x: 1 if x<=480000 else 0)
    cplval["song_length_0to8min"] = cplval["song_length"].apply(lambda x: 1 if x<=480000 else 0)
    cplte["song_length_0to8min"] = cplte["song_length"].apply(lambda x: 1 if x<=480000 else 0)
    # longer than 8 min, other type of music

    # List of features
    list_feat = ["song_length_0to8min", "language", "genre_id_extract"] 

    # Reduce dataset to features of interest
    cplall = cplall[list_feat]
    cpltr = cpltr[list_feat]
    cplval = cplval[list_feat]
    cplte = cplte[list_feat]


    # encode categorical features
    
    cplall = cplall.T.to_dict().values()
    vec = DictVectorizer(sparse=True)
    cplall = vec.fit_transform(cplall)
    
    cpltr = cpltr.T.to_dict().values()
    cpltr = vec.transform(cpltr)
    
    cplval = cplval.T.to_dict().values()
    cplval = vec.transform(cplval)
    
    cplte = cplte.T.to_dict().values()
    cplte = vec.transform(cplte)

    cplall = scipy.sparse.csr_matrix(cplall)
    cpltr = scipy.sparse.csr_matrix(cpltr)
    cplval = scipy.sparse.csr_matrix(cplval)
    cplte = scipy.sparse.csr_matrix(cplte)

    return cplall, cpltr, cplval, cplte


# Now we have item feature matrics with only song length, language, and genre info. 

# In[161]:


item_feat_mtrx_no_names, item_feat_train_mtrx_no_names, item_feat_val_mtrx_no_names, item_feat_test_mtrx_no_names = build_item_feature_matrix(df_songs, train_songs, val_songs, test_songs)


# **Convert the embedded artist/composer/lyricist names to `NumPy` matrices**

# In[103]:


artist_mtx = np.array(df_songs['artist_vector'].values.tolist())
composer_mtx = np.array(df_songs['composer_vector'].values.tolist())
lyricist_mtx = np.array(df_songs['lyricist_vector'].values.tolist())

artist_train_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(train_interact['song_id']),                                          'artist_vector'].values.tolist())
composer_train_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(train_interact['song_id']),                                            'composer_vector'].values.tolist())
lyricist_train_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(train_interact['song_id']),                                            'lyricist_vector'].values.tolist())

artist_val_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(val_interact['song_id']),                                        'artist_vector'].values.tolist())
composer_val_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(val_interact['song_id']),                                          'composer_vector'].values.tolist())
lyricist_val_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(val_interact['song_id']),                                          'lyricist_vector'].values.tolist())

artist_test_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(df_test['song_id']),                                         'artist_vector'].values.tolist())
composer_test_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(df_test['song_id']),                                           'composer_vector'].values.tolist())
lyricist_test_mtx = np.array(df_songs.loc[df_songs['song_id'].isin(df_test['song_id']),                                           'lyricist_vector'].values.tolist())

# memory failure
# df_songs[['lyricist{}'.format(i) for i in range(vec_size)]] = \
# pd.DataFrame(df_songs['lyricist_vector'].values.tolist(), index=df_songs.index)


# **Stack them row-wise to obtain the complete item feature matrices**

# In[162]:


item_feat_mtrx = scipy.sparse.hstack((item_feat_mtrx_no_names, artist_mtx, composer_mtx, lyricist_mtx))

train_item_feat_mtrx = scipy.sparse.hstack((item_feat_train_mtrx_no_names, artist_train_mtx,                                             composer_train_mtx, lyricist_train_mtx))
val_item_feat_mtrx = scipy.sparse.hstack((item_feat_val_mtrx_no_names, artist_val_mtx,                                           composer_val_mtx, lyricist_val_mtx))
test_item_feat_mtrx = scipy.sparse.hstack((item_feat_test_mtrx_no_names, artist_test_mtx,                                            composer_test_mtx, lyricist_test_mtx))


# In[214]:


print('train item feature matrix has the same # of rows as {}'.format(assert(train_item_feat_mtrx.shape[0] == train_interact.shape[0])))
print('# of data elements that are NaN or Inf: {}'      .format(np.sum(np.logical_not(np.isfinite(item_feat_train_mtrx_no_names.todense())))))


# ### Implicit Feedback User-Item Matrix ###

# **Create user-item matrix for LightFM**

# In[133]:


def  build_user_item_mtrx (ulist, cpltr, cpdtr):
    """ Build user item matrix (for test and train datasets)
    (sparse matrix, Mui[u,i] = 1 if user u has listened to song i again, 0 otherwise)

    ulist: member master list
    cpltr: unique items in the training set
    cpdtr: listen history, all records are recurring listening
     
    """

    # Build a dict with the song index in cpltr
    d_ci_tr = dict(zip(cpltr["song_id"], range(len(cpltr["song_id"]))))

    # Build a dict with the user index in ulist
    d_ui = dict(zip(ulist["msno"], range(len(ulist["msno"]))))

    # Build the user x item matrices using scipy lil_matrix
    Mui_tr = scipy.sparse.lil_matrix((len(ulist), len(cpltr)), dtype=np.int8)

    # Now fill Mui_tr with the info from cpdtr
    for i in range(len(cpdtr)):
        user = cpdtr["msno"].values[i]
        item = cpdtr["song_id"].values[i]
        ui, ci = d_ui[user], d_ci_tr[item]
        Mui_tr[ui, ci] = 1

    return Mui_tr.tocoo(copy=True)


# In[134]:


user_item_train_mtrx = build_user_item_mtrx(df_members, train_songs, train_interact[train_interact['target'] == 1])


# In[135]:


print('User-item train matrix has the same rows as member master dataset: {}'      .format(assert((user_item_train_mtrx.shape[0] == df_members.shape[0])))

print('User-item train matrix has the same columns as training songs dataset: {}'\
      .format(assert((user_item_train_mtrx.shape[1] == train_songs.shape[0])))


# In[ ]:


def fit_model(testdata, itef, lr, ep):
    """ Fit the lightFM model to training data

    testdata, itef = dataframe with user and song ids, user-item matrix of validation/testing data without ids
    lr, ep = (float, int) learning rate, number of epochs

    returns: d_user_pred, list_user, list_coupon
    list_coupon = list of test coupons
    list_user = list of user ID
    d_user_pred : key = user, value = predicted ranking of coupons in list_coupon
    """


    # Load data
    Mui_train = user_item_train_mtrx
    uf = user_feat_mtrx 
        
    itrf = train_item_feat_mtrx

    # Build model
    model = LightFM(learning_rate=lr, loss='warp')
    model.fit(
        interactions=user_item_train_mtrx,
        user_features=user_feat_mtrx,
        item_features=train_item_feat_mtrx,
        epochs=ep,
        num_threads=4, # CPU cores
        verbose=True)

    # dictionaries mapping real ids and integer ids that prediction requires
    uid_array = dict(zip(df_members["msno"].unique(), np.arange(len(df_members["msno"].unique()), dtype=np.int32)))
    pid_array = dict(zip(testdata["song_id"].unique(), np.arange(len(testdata["song_id"].unique()),dtype=np.int32)))

    pred_target_mtx = model.predict(
            list(uid_array.keys()),
            list(pid_array.keys()),
            user_features=uf,
            item_features=itef,
            num_threads=4)
    
    # output recommendation score
    testdata['pred_target'] = testdata.apply(lambda row:                                              pred_target_mtx(uid_array[row['msno']], pid_array[row['song_id']]))
        
    return testdata # testdata[['msno', 'song_id', 'pred_target']]


# In[ ]:


# no_comp, lr, ep = 10, 0.01, 5
lr, ep = 0.01, 5
val = fit_model(val_interact, val_item_feat_mtrx, lr, ep)

    
with open(data_dir+'recommendation_score_validation','wb') as outfile:
    pickle.dump(val, outfile)      