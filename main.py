#importing libraries
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


#loading dataset
movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(1)#displays first 5 rows for()


# In[4]:


#genres means which type of movie
# homepage means movie website
#id means tmdb id it is movies database ,so in that tmdb what is the id of the particular movie 
#keywords means movie description means what is there in movie like in avtar movie we have space,aliens 
#orglang means in which language movie is made like avtar made in english and dubbed to other lang
#org title that part lang title
#overview means brief desc about movie like summary
#pop is jus a num A measure of how well-liked and widely seen a movie is
#prodcomp means like suresh prodcutions
#prod count means in which counts movie shooting is done
#release date on which day released
#revenue how much money earned 
#runtime duration of movie in minutes
#spoken lang in movies how many
#status -released,about to release
#tagline movie name downside
#title english title of movie
#vote avg  average rating a movie has received from users.
#vote count the total number of users who have submitted a rating for the movie.


# In[5]:


credits.head(1)


# In[6]:


#cast=actors
#crew=behind camera directors,editorsetc
credits.head(1)['cast'].values
#credits.head(1)['crew'].values


# In[7]:


#merging
#reassign dataframe name as we merged 2to1
Movies = movies.merge(credits,on='title')


# In[8]:


Movies.head(1)


# In[9]:


movies.merge(credits,on='title').shape
#shape gives noof rows n cols


# In[10]:


#individual shapes
movies.shape


# In[11]:


credits.shape
#we got 4 but total 23cols 20in mov n 3in cred nah coz we merged based on title so it counted only once title col


# In[12]:


movies['original_language'].value_counts()


# In[13]:


Movies.info()


# In[14]:


#remove cols which r not useful in our analysis of proj
#useful cols:
#genres
#id when we make website atlast we need movie posters so from id we can fetch that
#keywords they r like tags but tags r diff they r created by us acc to user search for specific movie
#title
#overview coz if recom should do based on same content if2 movs content same it comps both overview rec both
#cast
#crew


# In[15]:


Movies = Movies[['movie_id','genres','keywords','title','overview','cast','crew']]


# In[16]:


movies.columns


# In[17]:


Movies.head(1)


# In[18]:


#so now from this dataframe we should create new dataframe which contains movid,title,tags
#tags=overv+genres+cast+crew
#so we will merge all these to make tag col but in cast will take top 3 actors of mov,crew director in keywords ids are removed
#datapreprocessing remove missingval,duplicates


# In[19]:


#missing values
Movies.isnull().sum()


# In[20]:


#in overview 3rows we dont know values it has missingval
Movies.dropna(inplace=True)


# In[21]:


Movies.isnull().sum()


# In[22]:


#duplic
Movies.duplicated().sum()


# In[23]:


Movies.iloc[0]


# In[24]:


Movies.iloc[0].genres


# In[25]:


#{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}
#it is in the form of list of dicts so we should bring them in format:
#['Action','Adventure','Fantasy','Science Fiction']
#so we can create a helper func for this


# In[26]:


#helper func
def convert(obj):
    L=[]
    for i in obj:
        L.append(i['name'])
        return L


# In[27]:


#if we call conv func and pass list 
convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
#it is form of string of list ,so conv to list 
#to do this we have module in python 'ast' in ast we have lit_eval func


# In[28]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[29]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[30]:


#Movies['genres'].apply(convert) and store it in same col
Movies['genres']=Movies['genres'].apply(convert)


# In[31]:


Movies.head()
#here genres changed to list with only names ids removed


# In[32]:


Movies['keywords'] = Movies['keywords'].apply(convert)


# In[33]:


Movies.head()


# In[34]:


#now i want top3 cast of every movie so take first3 dicts 
Movies['cast'][0]


# In[35]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
           L.append(i['name'])
           counter+=1
        else:
            break
    return L


# In[36]:


#Movies['cast'].apply(convert3)
Movies['cast'] = Movies['cast'].apply(convert3)


# In[37]:


Movies.head()
#top 3 cast of every mov


# In[38]:


#now in crew i want dict where job=director only
Movies['crew'][0]


# In[39]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[40]:


Movies['crew'] = Movies['crew'].apply(fetch_director)


# In[41]:


Movies.head()


# In[42]:


#now overview is in form of string conv to list
Movies['overview'][0]


# In[43]:


Movies['overview'] = Movies['overview'].apply(lambda x:x.split())
#The function lambda itself is very simple: 
    #it splits the string (x) using the split() method and returns the resulting list of words.


# In[44]:


Movies.head()


# In[45]:


#now we concatenate all cols which r in form of lists ,that becomes a big list 
#then we conv list to string which becomes a big paragraph that finaly bcoms our tag col
#now we have to removes spaces bw words in keyw,genr,cast,crew
#eg sam worthington - samworthington 
#we need to do this coz if there is a space sam and worthington bcoms 2 seperate words 2tags
#we have sam mendes also so if we search sam wort movies recom sys shows sam mend movies


# In[46]:


Movies['genres'] = Movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
Movies['keywords'] = Movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
Movies['cast'] = Movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
Movies['crew'] = Movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[47]:


Movies.head()


# In[48]:


Movies['tags'] = Movies['overview']+Movies['genres']+Movies['keywords']+Movies['cast']+Movies['crew']


# In[49]:


Movies.head()


# In[50]:


#remove rem cols kept movid,tit,tags
#so create new df
new_df=Movies[['movie_id','title','tags']]


# In[51]:


new_df.head()


# In[52]:


#conv list in tags to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[53]:


new_df.head()


# In[54]:


new_df['tags'][0]


# In[55]:


#converting tot matter in tags to lowercase coz we want to rec based on this
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[56]:


new_df.head()


# In[57]:


import nltk


# In[58]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[59]:


#helperfunc
def stem(text):
    y = []
    
    for i in text.split():
       y.append(ps.stem(i))
    return " ".join(y)


# In[60]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[61]:


new_df.head()


# In[62]:


#now we need to do text vectorization
#prob stmnt= create a website where user says a movname and from that we should give 5 similar mov to user.
#out of 5000mov how we will know which 5movs r similar? based on tags


# In[63]:


new_df['tags'][0]


# In[64]:


new_df['tags'][1]


# In[65]:


#from these 2tags we should calc similarity bw those 2texts
#so here concept of vectorization will be used
#conv text to vector
#every mov bcoms 1vector
#if someone likes some mov and asks more which movies should i see
#in 5000vect u should take closest vect for above case
#techniques under textvect:bagofwords,tfidf,wordtovec
#bagofwor: concatente all tags then it becomes large text under string now i need 5000 mostcommon words
#calc frequency of every word;no of times word there we will get a table or df word1(action)how many times came in movie1
#    w1 w2 w3
# m1  1  2  3 #123 is vector 
#stop words is,are,and,to,from useful in sentence formation but wont contribute to sentence meaning
#while performing stop words we shouldnt consider stopwords remove them
#manually also we can do by join all strings & extract most comm words and do vect
#but we have libs like sckitlearn under we have func class count_vectorizer which makes our work easier


# In[66]:


#cv.fit_transform(new_df['tags']):
#This part uses the CountVectorizer object (cv) to process the text data in the 'tags' column.
#The fit_transform method performs two actions:
#fit: Analyzes the text data in new_df['tags'] to learn the vocabulary of unique words or phrases (depending on the CountVectorizer configuration).
#transform: Converts the text data into a numerical representation based on the learned vocabulary. This numerical representation is typically a sparse matrix where most elements are zero.
#Output and shape:

#.toarray(): This method converts the sparse matrix output from cv.fit_transform into a dense NumPy array. While this can be useful for visualization or debugging, it's important to note that dense arrays can consume more memory compared to sparse matrices.

#.shape: This attribute retrieves the shape of the resulting NumPy array. The shape is a tuple containing two integers representing the number of rows and columns in the array.#
#Overall shape explanation:

#The final shape of the output depends on the dimensions of the original data (new_df['tags']) and the number of unique words/phrases learned by the CountVectorizer (cv). Here's a breakdown of the possible interpretations:

##Number of rows: This typically corresponds to the number of data points (documents or entries) in the 'tags' column of new_df.
#Number of columns: This represents the size of the vocabulary learned by the CountVectorizer. It reflects the number of unique words or phrases identified in the 'tags' data.
#For example, if new_df has 100 rows (data points) and the CountVectorizer learned 500 unique words/phrases from the 'tags' column, the resulting array's shape would be (100, 500). This means there are 100 rows (representing the 100 data points) and 500 columns (representing the 500 unique words/phrases).
#feature extraction
#Feature extraction in text data processing refers to the process of transforming textual information into numerical features that can be used by machine learning algorithms. Text data itself cannot be directly processed by algorithms, so feature extraction acts as a bridge between the textual world and the numerical world of machine learning.
#There are various techniques for feature extraction from text data. Some common methods include:
#Machine learning algorithms primarily operate on numerical data.
#Bag-of-Words (BoW): This method represents a document as a collection of words, ignoring the order and grammar. Each unique word becomes a feature, and its value could be the frequency of its occurrence in the document.
#TF-IDF (Term Frequency-Inverse Document Frequency): This method goes beyond simple word frequency and considers the importance of a word. It balances the weight of a word based on how often it appears in a specific document (TF) and how rare it is across the entire dataset (IDF).
#N-grams: This technique captures sequences of words (phrases) instead of single words. By considering groups of words, it can capture more context and meaning from the text data.
#Word Embeddings: This is a more advanced technique that represents words as vectors in a high-dimensional space. Words with similar meanings tend to have similar vector representations, allowing the model to capture semantic relationships between words.


# In[77]:


#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features=5000,stop_words='english')
#cv=obj


# In[78]:


#fit_transform(new_df['tags'] in these all vals bcoms 0 
#bydef cv returns obj i.e scipy sparsematrix so conv them to numpy array
#cv.fit_transform(new_df['tags']).toarray()


# In[79]:


#cv.fit_transform(new_df['tags']).toarray().shape
#from 4806mov it learned 5000words


# In[70]:


#There are likely 4806 movies (or data points) represented in the 'tags' column.
#The CountVectorizer (cv) identified 5000 unique words or phrases used to describe these movies in the tags.
#The resulting 4806x500 array would then have a value of 1 in a specific position (i, j)
#if the movie (row i) has the corresponding word/phrase (column j) in its tags. 
#For example, if the first movie (row 0) has the tags "action, comedy", the array would have a 1 at positions (0, 0)
#for "action" and (0, 1) for "comedy", and 0s elsewhere in that row for words not present in those tags.


# In[80]:


#vectors=cv.fit_transform(new_df['tags']).toarray()


# In[81]:


#vectors


# In[82]:


##now all movies conv to vects
#vectors[0]
#1st movie 
#in singlemov 5000 commwords wont there so most will be 0 it will be sparse matrix 


# In[83]:


#cv.get_feature_names()
#here action,actions,activity,activities r similar words but we have 2 here
#apply stemming technique used in text processing it conv [love,loves,loving] to [love,love,love]
#for this  we need nltk lib intsall and apply stm on tags data


# In[84]:


#len(cv.get_feature_names())


# In[85]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[86]:


cv.fit_transform(new_df['tags']).toarray()


# In[87]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[88]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[89]:


vectors


# In[90]:


vectors[0]


# In[91]:


cv.get_feature_names()


# In[92]:


len(cv.get_feature_names())


# In[93]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[94]:


#unique became uniqu


# In[95]:


#4806 vect ,everyvec has 5000nums
#in 5000vec space cal dist bw one mov(vector) to other dist low= similarity more
#here euclideandist is not calc insted calc cosine dist ie angle bw them vec
#angle 0,5,10 short dist 90 dissimilar movs,180 oppmov
#this is 5000dim space Jupyter Notebook
#always rememb in high dim space eucldist fails its not a reliable measure(curse of dimensionality)
#u can use cosine dist thatis theta bw 2vec,small the teta the mov r similar
#dist invprop to simlarity


# In[96]:


from sklearn.metrics.pairwise import cosine_similarity


# In[97]:


cosine_similarity(vectors)


# In[98]:


cosine_similarity(vectors).shape
#1st vec is compared with 4806vec,2nd is 4806.....4806thvec 4806


# In[99]:


similarity=cosine_similarity(vectors)


# In[100]:


similarity[0]
#sarray aying similarity bw 1stmov to other
#1 we got coz 1st mov wd 1stmov sim will be 1 


# In[101]:


#when i give a mov it should recom 5simlmov 
#fst we need movie index eg avtar index is 0 in our data
#sort the dist like more sim movs come fst less will go back and fetch top5sim movs so fetch index


# In[102]:


#fetching index
new_df[new_df['title']== 'Batman Begins'].index


# In[103]:


new_df[new_df['title']== 'Batman Begins'].index[0]


# In[127]:


new_df['title']=='Avatar'


# In[128]:


new_df[new_df['title']=='Avatar'].index
#creates a new DataFrame containing only the rows from new_df where the movie title is "Avatar". 
#Then, it extracts the index labels (row numbers) of those filtered rows.


# In[129]:


new_df[new_df['title']=='Avatar'].index[0]
#index[0] position 0


# In[130]:


sorted(similarity[0])


# In[131]:


sorted(similarity[0])[-10:-1]


# In[110]:


sorted(similarity[0],reverse=True) 
#here desc order we got but index val changes moresim mov to 0thmov
#so use enum


# In[133]:


#enum
enumerate(similarity[0])
#o/p is enum obj


# In[134]:


list(enumerate(similarity[0]))
#0 index val 1.00 so fixed positions
#oth mov to 0th mov dist 1.0
#list of tuples


# In[137]:


sorted(list(enumerate(similarity[0])),reverse=True)
#sorting over but based on index it over but we want based on 2nd one i.e similarity


# In[138]:


sorted(list(enumerate(similarity[0])),reverse=True ,key=lambda x:x[1])
#0 is avtar movie to that similar is 1216,2409,...


# In[139]:


#5mov sim 
sorted(list(enumerate(similarity[0])),reverse=True ,key=lambda x:x[1])[1:6]
#rev=true argument reverses the sorting order
#lambda x: x[1]: This lambda function takes one argument x, which represents each tuple in the list of tuples.
#It simply returns the second element of the tuple (x[1]), which corresponds to the similarity score in this case.
# [1:6]slicing operation selects a specific part of the sorted list.
#It retrieves elements from index 1 (second element) up to, but not including, index 6 (sixth element).
#Similarity measures how alike two data points are. Higher similarity scores indicate a closer relationship between the data points.
#n text analysis, similarity scores might reflect how similar the content or meaning of two documents are
#Distance measures how far apart two data points are. Lower distances indicate a closer relationship between the data points, while higher distances suggest greater dissimilarity.
#In machine learning algorithms like K-Nearest Neighbors (KNN), distance metrics are used to identify the k nearest neighbors (most similar data points) for a new data point


# In[ ]:


Similarity:

Identifying similar movies: Similarity measures are used to compare movies based on various features. These features can include:

Genre: Movies belonging to the same genre (e.g., comedy, action) are likely to be similar in terms of content and style.
Cast and crew: Movies featuring the same actors, directors, or other crew members might share thematic elements or production styles.
User ratings: Movies that receive similar ratings from other users, especially those with similar tastes, are considered potentially similar for a specific user.
Content-based features: Techniques like natural language processing can analyze movie descriptions, reviews, or dialogue to identify thematic similarities.
Recommendation based on similarity: By calculating the similarity between a user's watched or liked movies and other movies in the system, the recommendation system can suggest movies that share similar characteristics. This approach assumes that users who enjoy a particular movie might also enjoy similar movies.

Distance:

Ranking similar movies: While similarity helps identify potentially relevant movies, distance metrics help refine the recommendations further. By calculating the distance between a user's preferred movies and other similar movies, the system can prioritize recommendations.
Common distance metrics include Euclidean distance or cosine similarity. A lower distance between a user's movie and a recommended movie suggests a closer match.
Combined approach:

Most recommendation systems leverage a combination of both similarity and distance. Here's a simplified example:

Identify similar movies: Based on genres, actors, or user ratings, the system identifies movies that share similarities with a user's watched or liked movies.
Rank similar movies: The system calculates the distance between each identified movie and the user's preferred movies.
Recommend closest matches: The system prioritizes movies with the lowest distance (highest similarity) and presents them as recommendations to the user.


# In[143]:


new_df['title']=='Avatar'


# In[146]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True ,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(i[0])
        


# In[147]:


recommend('Avatar')
#these r 5sim mov to avatar
#now with index we need to fetch those movie names


# In[148]:


new_df.iloc[1216]


# In[149]:


new_df.iloc[1216].title


# In[150]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True ,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
        


# In[151]:


recommend('Avatar')


# In[ ]:


#model is built now we should convert into website
#do in pycharm


# In[152]:


#so we need to bring list of movies fromhere that is useful in webpage selectbar to display list of movs there 
#so use pickle
#he pickle module provides functions like pickle.dumps to convert a Python object into a byte stream.
import pickle 


# In[ ]:


Pickle, in the context of Python programming, refers to the process of converting a Python object (like a list, dictionary, or even a custom class) into a byte stream. This byte stream can then be stored on disk or transmitted over a network. The opposite process, unpickling, involves converting the byte stream back into the original Python object.

Here's a breakdown of the key aspects of pickling:

Purpose:

Serialization: Pickle allows you to serialize Python objects. Serialization means converting an object into a format that can be stored or transmitted. This is useful for:
Saving the state of your program or data for later use.
Sharing data between different Python processes or applications.
How it Works:

Pickling:

The pickle module provides functions like pickle.dumps to convert a Python object into a byte stream.
This process involves encoding information about the object's data type, structure, and any contained objects.
Unpickling:

The pickle.loads function takes the pickled byte stream and converts it back into the original Python object.
This process decodes the information stored during pickling and recreates the object in memory.
Limitations:

Security: Pickle is not secure for untrusted data. Unpickling data from an untrusted source can lead to security vulnerabilities. Only unpickle data from sources you trust.
Compatibility: Pickled data might not be compatible between different Python versions due to potential changes in object structures or modules.
Alternatives:

JSON: For simple data structures, JSON (JavaScript Object Notation) is a more human-readable and secure alternative for serialization.
YAML: YAML (YAML Ain't Markup Language) is another option that offers a more readable format compared to pickle.
When to Use Pickle:

When you need to serialize complex Python objects for storage or transmission within a trusted environment.
When compatibility between different Python versions is not a major concern.


# In[153]:


pickle.dump(new_df,open('movies.pkl','wb')) 
#open cdrive,users,there u find movies.pkl copy it and paste in pycharm


# In[154]:


new_df['title'].values


# In[155]:


#at the end our dataframe will be dictionary
new_df.to_dict()


# In[156]:


#so when we sending df to website getting error so conv to dict and send dict and paste m-d.pkl in pycharm
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb')) 


# In[157]:


#sim func not there in pych code dump from here
pickle.dump(similarity,open('similarity.pkl','wb'))
