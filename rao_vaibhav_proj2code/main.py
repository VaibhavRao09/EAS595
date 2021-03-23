import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import cv2

import os 

DATASET_ROOT = "./dataset_files/"

#Function to return Path where our Dataset related files are located
def dataset_input(path):
    return os.path.join(DATASET_ROOT, path)

#Load the csv file with training data
train_data = pd.read_csv(dataset_input("Train.csv"))
#print ("DEBUG: Train CSV Head data :\n" ,train_data.head())
#print("DEBUG: Train data sample per Class:\n" , train_data.groupby('ClassId')['ClassId'].count())

#Load the csv file with the test data
test_data =  pd.read_csv(dataset_input("Test.csv"))
#print ("DEBUG: Test CSV Head data :\n" , test_data.head())
#print("DEBUG: Test data sample per Class:\n" , test_data.groupby('ClassId')['ClassId'].count())

#Take out "Paths" Field and "Class ID" from the csv data loaded 
paths_train = train_data['Path'].values
Y = train_data['ClassId'].values

paths_test = test_data['Path'].values
y_test = test_data['ClassId'].values
  
#The above TRAINING DATA from the CSV is grouped and data of similar classes is adjacent
#Lets Shuffle for a better learning!!!
indices = np.arange(Y.shape[0])
np.random.seed(43) #Seed so that the same sequence is generated accross runs
np.random.shuffle(indices)
paths_train = paths_train[indices]
Y = Y[indices]
'''
print ("DEBUG:: Training data head for Path and y_train to ensure proper shuffling::\n", 
       paths_train, "\n", Y)
# Just for testing out loaded images and figure out parameters::
temp = cv2.imread("./dataset_files/Train/20/00020_00000_00000.png")
temp = cv2.resize(temp,(40, 40))
temp = temp/255.0
plt.imshow(temp)
'''
def load_image(Path):
    data = []
    for x in Path:
        img = cv2.imread(dataset_input(x))
        img = cv2.resize(img,(40, 40))
        img = img/255.0 # NORMALIZE
        data.append(img)
    print ("DEBUG DATASET LOADED FOR PATH", Path)
    return np.array(data)


print("===========================================================================")
print("LOADING TRAIN DATA.... This may take a while")
#load data for training from image files
X = load_image(paths_train)

print("===========================================================================")
print("LOADING TEST DATA.... This may take a while")
#load data for testing from image files
X_test = load_image(paths_test)

#Lets Split the Train data into Train and Validate in ratio 80:20
(X_train,X_val)=X[(int)(0.2*len(Y)):],X[:(int)(0.2*len(Y))]
(y_train,y_val)=Y[(int)(0.2*len(Y)):],Y[:(int)(0.2*len(Y))]


#Convert Y (Test and Train) to categorical class representation.
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=43)
y_test = to_categorical(y_test, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)

#plt.imshow(X_train[0])
#plt.imshow(X_train[1])
#print(X_train.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]),
    MaxPool2D(pool_size=(2, 2)),
    #Dropout(rate=0.25),
    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    #Dropout(rate=0.25),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    #Dropout(rate=0.5),
    Dense(43, activation='softmax')
])

'''
model = Sequential([
    Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]),
    MaxPool2D(pool_size=(2, 2)),
    #Dropout(rate=0.25),
    Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    #Dropout(rate=0.25),
    Conv2D(filters=256, kernel_size=(3,3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    #Dropout(rate=0.5),
    Dense(43, activation='softmax')
])
'''
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.summary()

from keras.preprocessing.image import ImageDataGenerator
from timeit import default_timer as timer

# creating datagenerator for augmenting images
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

epochs = 20
batch_size = 32 ### A smaller batch size had better accuracy that a larger.
VERBOSE = 1  # Change this value to "0" if you dont want details for every epoch on console

train_datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

start_time = timer()
history = model.fit_generator(train_datagen, epochs=epochs, verbose=VERBOSE,
                              validation_data=(X_val, y_val),
                              steps_per_epoch= round(X_train.shape[0] / batch_size))
end_time = timer()

print("===========================================================================")
print("DEBUG:: Trained in {:.2f} minutes".format((end_time - start_time) / 60))
print("DEBUG:: history values after model.fit\n", history.history)


plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

acc=history.history['accuracy'][-1] #Final Value of accuracy
val_acc=history.history['val_accuracy'][-1] #Final Value of val_accuracy
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(X_test,
                                    y_test,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    seed=17)

print("============================================================================")
test_acc = model.evaluate_generator(test_generator)[1]
print("Results at the end of training and using model.evaluate on test data: acc={:.02f}%, test_acc={:.02f}%"
      .format(acc*100, test_acc*100))

#We can Use both Evaluate as above or Predict to check accuracy. 
#Example for predict below:
from sklearn.metrics import accuracy_score, confusion_matrix
pred = model.predict_classes(X_test)
pred = to_categorical(pred, num_classes=43)
test_acc_predict = accuracy_score(y_test, pred)
confusionmatrix = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
print("Accuracy Score after using the model to Predict: {:.2f}%".format(test_acc_predict*100))
print("=========Confusion Matrix :===========\n",confusionmatrix)
