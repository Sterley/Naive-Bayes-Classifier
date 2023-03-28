import numpy as np
import re
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

filename = 'smsspamcollection/SMSSpamCollection'
regex = re.compile('[^a-zA-Z0-9\s]')

# Read the data from the file and clean it
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    data = []
    labels = []

    for line in lines:
        parts = line.split('\t')
        label = parts[0]
        message = parts[1]

        # clean the message
        message = regex.sub('', message)

        # separate the words
        words = message.split()
        data.append(words)
        labels.append(label)

# convert the data to numpy arrays
data = np.array(data)
labels = np.array(labels)

# divide the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)



class BayesianSpamClassifier:
    def __init__(self, load_data):
        self.epsilon = 1e-300
        self.load_data =load_data

    def train(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        if self.load_data:
            # loading parameters
            with open("class_prob_dico.pkl", "rb") as file:
                self.class_prob_dico= pickle.load(file)
            with open("words_occ_dico.pkl", "rb") as file:
                self.words_occ_dico = pickle.load(file)
            with open("words_occ_dico_byClass.pkl", "rb") as file:
                self.words_occ_dico_byClass = pickle.load(file)
        else:
            # training parameters
            self.class_prob_dico = {}
            self.words_occ_dico = {}
            self.words_occ_dico_byClass = {}
            unique_labels, label_counts = np.unique(train_labels, return_counts=True)
            for lab in unique_labels:
                self.class_prob_dico[lab] = label_counts[unique_labels == lab][0] / len(train_labels)
            for i in range(len(train_data)):
                for word in train_data[i]:
                    if word not in self.words_occ_dico:
                        self.words_occ_dico[word] = 1
                    else:
                        self.words_occ_dico[word] += 1
                    if train_labels[i] not in self.words_occ_dico_byClass:
                        self.words_occ_dico_byClass[train_labels[i]] = {}
                    if word not in self.words_occ_dico_byClass[train_labels[i]]:
                        self.words_occ_dico_byClass[train_labels[i]][word] = 1
                    else:
                        self.words_occ_dico_byClass[train_labels[i]][word] += 1
            # saving parameters
            with open("class_prob_dico.pkl", "wb") as file:
                pickle.dump(self.class_prob_dico, file)
            with open("words_occ_dico.pkl", "wb") as file:
                pickle.dump(self.words_occ_dico, file)
            with open("words_occ_dico_byClass.pkl", "wb") as file:
                pickle.dump(self.words_occ_dico_byClass, file)
    
    def get_prob_byClass(self, label):
        # get the probability of a class
        return self.class_prob_dico[label]
    
    def get_prob_word_byClass(self, word, label):
        # get the probability of a word given a class
        if word in self.words_occ_dico_byClass[label]:
            return self.words_occ_dico_byClass[label][word]/sum(self.words_occ_dico_byClass[label].values())
        else:
            return self.epsilon
        
    def get_prob_word(self, word):
        # get the probability of a word
        if word in self.words_occ_dico:
            return self.words_occ_dico[word]/sum(self.words_occ_dico.values())
        else:
            return self.epsilon
    
    def predict(self, x):
        # predict the class of a message
        keys_list = list(self.class_prob_dico.keys())
        ret_prob_class = {}
        for k in keys_list:
            ret_prob_class[k] = 0     
        for k in keys_list:
            prob_num = np.log(self.get_prob_byClass(k))
            prob_den = 0
            for xi in x:
                prob_num += np.log(self.get_prob_word_byClass(xi, k))
                prob_den += np.log(self.get_prob_word(xi))
            prob = prob_num - prob_den
            ret_prob_class[k] = prob
        return max(ret_prob_class, key=ret_prob_class.get)

    def accuracy(self, desc_set, label_set):
        # compute the accuracy of the model
        count=0
        for i in range(len(label_set)):
          if self.predict(desc_set[i]) == label_set[i]:
            count+=1
        return count/len(label_set)


bayes = BayesianSpamClassifier(True)
bayes.train(train_data, train_labels)
print("len test_data", len(test_data))
print(bayes.accuracy(test_data, test_labels))

print("Predicting...")
y_true = []
y_pred = []
for i in range(len(test_data)):
    y_true.append(test_labels[i])
    y_pred.append(bayes.predict(test_data[i]))
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm, cmap='binary')
plt.show()

# get the 5 most probable words for spam
spam_word_prob = {}
for word in bayes.words_occ_dico:
    spam_word_prob[word] = bayes.get_prob_word_byClass(word, 'spam')
sorted_spam_word_prob = sorted(spam_word_prob.items(), key=lambda x: x[1], reverse=True)
for i in range(5):
    print(sorted_spam_word_prob[i][0], ":", sorted_spam_word_prob[i][1])