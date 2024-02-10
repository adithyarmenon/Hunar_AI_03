#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


data = pd.read_csv('C:/Users/user/Downloads/mail_data.csv')


# In[6]:


print(data)


# In[7]:


data.dropna(inplace=True)
data.drop_duplicates(inplace=True)


# In[8]:


vectorizer = CountVectorizer()


# In[12]:


x = data['Message']
y = data['Category']


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 42)


# In[14]:


vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# In[15]:


nb_classifier = MultinomialNB()
nb_classifier.fit(x_train_vectorized,y_train)


# In[16]:


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(x_train_vectorized,y_train)


# In[17]:


nb_predictions = nb_classifier.predict(x_test_vectorized)
svm_predictions = svm_classifier.predict(x_test_vectorized)


# In[18]:


nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions, pos_label='spam')
nb_recall = recall_score(y_test, nb_predictions, pos_label='spam')
nb_f1 = f1_score(y_test, nb_predictions, pos_label='spam')
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, pos_label='spam')
svm_recall = recall_score(y_test, svm_predictions, pos_label='spam')
svm_f1 = f1_score(y_test, svm_predictions, pos_label='spam')


# In[19]:


print("Naive bayes metrics: ")
print("accuracy:  ",nb_accuracy)
print("Precision: ",nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
print("\nSupport Vector Machine Metrics:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)


# In[20]:


def print_predictions(classifier, X_test, predictions):
    print(f"Predictions for {classifier}:")
    for text, label in zip(X_test, predictions):
        print(f"Email: WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.")
        print(f"Predicted Label: {'spam' if label == 'spam' else 'not spam'}")


# In[24]:


nb_predictions = nb_classifier.predict(x_test_vectorized)
print_predictions("Naive Bayes", x_test, nb_predictions)


# In[ ]:




