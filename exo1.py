import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import pickle
from sklearn.metrics import confusion_matrix


#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# merge the list of pixels of the list of images
train_X = train_X.reshape((train_X.shape[0], 28 * 28))
# merge the list of pixels of the list of images
test_X = test_X.reshape((test_X.shape[0], 28 * 28))

print("\n----------------------------------------------------------")
#printing the shapes of the vectors 
print('Dataset type :', type(train_X))
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
print("----------------------------------------------------------\n")


class BayesianNumberClassifier:
    def __init__(self, load_list):
        self.epsilon = 1e-10
        self.load_list =load_list

    def train(self, desc_set, label_set):
        self.desc_set = desc_set
        self.label_set = label_set
        self.classes = np.unique(train_y)
        if self.load_list:
            with open('my_histogram.pkl', 'rb') as f:
                self.histogram = pickle.load(f)
            with open('my_prob_class_tab.pkl', 'rb') as f:
                self.prob_class_tab = pickle.load(f)
            with open('my_prob_pixel_i_sg_tab.pkl', 'rb') as f:
                self.prob_pixel_i_sg_tab = pickle.load(f)
        else:
            print("Creating the tables...")
            self.histogram = self.generate_histo(self.desc_set, self.label_set, self.classes)
            self.prob_class_tab = self.get_prob_class_tab()
            self.prob_pixel_i_sg_tab = self.get_prob_pixel_i_sg_tab()
            with open('my_histogram.pkl', 'wb') as f:
                pickle.dump(self.histogram, f)
            with open('my_prob_class_tab.pkl', 'wb') as f:
                pickle.dump(self.prob_class_tab, f)
            with open('my_prob_pixel_i_sg_tab.pkl', 'wb') as f:
                pickle.dump(self.prob_pixel_i_sg_tab, f)
    def generate_histo(self, train_X, train_y, classes):
        # list histograms of each class of the training set
        histo_niv_class = []
        for c in classes: 
            # filter the training set to keep only the images of the class c
            class_filtered_train_X = train_X[train_y == c]
            histo_niv_class_i = []
            for i in range(784):
                pixel_filtered_train_X_tmp = class_filtered_train_X[:, i] # by column
                # generate the histogram of the merged list of images
                histo_niv_pixel = (np.histogram(pixel_filtered_train_X_tmp, bins=np.arange(257), density=True))[0]
                histo_niv_class_i.append(histo_niv_pixel)
            histo_niv_class.append(np.array(histo_niv_class_i))
        return np.array(histo_niv_class)
    
    def get_prob_class_c(self, c):
        # the number of exemples of class c / total exemples in the dataset
        return len(self.desc_set[self.label_set == c])/len(self.desc_set)

    def get_prob_pixel_i_sg(self, i, sg):
        # the number of exemples of pixel i with intensity sg / total exemples in the dataset
        return len(self.desc_set[self.desc_set[:, i] == sg])/len(self.desc_set)
    
    def get_prob_class_tab(self,):
        prob_class_tab = []
        for c in self.classes:
            prob_class_tab.append(self.get_prob_class_c(c))
        return prob_class_tab

    def get_prob_pixel_i_sg_tab(self,):
        prob_pixel_i_sg_tab = []
        for i in range(784):
            prob_pixel_i_sg = []
            for sg in range(256):
                prob_pixel_i_sg.append(self.get_prob_pixel_i_sg(i, sg))
            prob_pixel_i_sg_tab.append(prob_pixel_i_sg)
        return prob_pixel_i_sg_tab
    
    def plot(self, classe, pixel):
        intensity = [i for i in range(256)]
        plt.plot(intensity, self.histogram[classe][pixel]) 
        plt.title("histogram") 
        plt.xlabel("Intensité de la couleur")
        plt.ylabel("Densité")
        plt.show()

    def predict(self, x):
        prob_class = []
        for c in self.classes:
            prob_num = np.log(self.prob_class_tab[c])
            for i in range(len(x)):
                prob_num += np.log(self.histogram[c][i][x[i]] + self.epsilon)
            prob_den = 0
            for i in range(len(x)):
                prob_den += np.log(self.prob_pixel_i_sg_tab[i][x[i]] + self.epsilon)
            prob = prob_num - prob_den
            prob_class.append(prob)
        return np.argmax(prob_class)

    def accuracy(self, desc_set, label_set):
        count=0
        for i in range(len(label_set)):
          if self.predict(desc_set[i]) == label_set[i]:
            count+=1
        return count/len(label_set)
        

bayes = BayesianNumberClassifier(True)
bayes.train(train_X, train_y)
num_samples = 100
random_indices = np.random.choice(len(test_X), num_samples, replace=False)
random_samples = test_X[random_indices]
random_labels = test_y[random_indices]
print("Predicting...")
y_true = []
y_pred = []
for i in range(num_samples):
    y_true.append(random_labels[i])
    y_pred.append(bayes.predict(random_samples[i]))
cm = confusion_matrix(y_true, y_pred)
plt.xlabel(('Accuracy: '+str(bayes.accuracy(random_samples, random_labels))))
plt.imshow(cm, cmap='binary')
plt.show()

# get the histogram of each class
histogram = bayes.histogram
# get the number of images in the training set
total_images = len(train_X)
normalized_histogram = histogram / total_images
p_x_yk = np.vstack(normalized_histogram)
# print the matrix of parameters p(x|yk) for each class
print("Matrice de paramètres p(x|yk) pour chaque classe : \n", p_x_yk)