import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
#%%

# Check if GPU is available
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
#%%
participant_list=["Participant"+str(d) for d in range(1,11)]
participant_list
df_collection={}

#%%
path='data/Participant_'
for i in range(1,11):
    df_collection["Participant"+str(i)]= pd.read_csv(path + str(i) + '.csv', header=1)


#%%


for i in range(10):
    df_collection[participant_list[i]].columns.values[69]='Activity'
            

#%%

N_TIME_STEPS = 100 #sliding window length
STEP = 50 #Sliding window step size
N_FEATURES = 9

def generate_sequence(x, y, n_time_steps, step):
    
    segments = []
    labels = []
    for i in range(0, len(x) - n_time_steps, step):
        ax = x['Ax'].values[i: i + n_time_steps]
        ay = x['Ay'].values[i: i + n_time_steps]
        az = x['Az'].values[i: i + n_time_steps]

        lx = x['Lx'].values[i: i + n_time_steps]
        ly = x['Ly'].values[i: i + n_time_steps]
        lz = x['Lz'].values[i: i + n_time_steps]
        
        gx = x['Gx'].values[i: i + n_time_steps]
        gy = x['Gy'].values[i: i + n_time_steps]
        gz = x['Gz'].values[i: i + n_time_steps]

        label = stats.mode(y['Activity'][i: i + n_time_steps])[0][0]
        segments.append([ax, ay, az, lx, ly, lz, gx, gy, gz])
        labels.append(label)
        
    return segments, labels
#%%
def make_frames(data):
    
    # Select left pocket data
    left_pocket = data.iloc[:,1:10]

    # Select right pocket data
    right_pocket = data.iloc[:,15:24]
    right_pocket.columns=['Ax', 'Ay', 'Az', 'Lx', 'Ly', 'Lz', 'Gx', 'Gy', 'Gz']
    
    

    
    # Extract labels 
    a_labels = data.iloc[:, 69] 
    a_labels = a_labels.to_frame()
    
    #replace typo 'upsatirs' with upstairs! 
    a_labels.loc[(a_labels['Activity'] == 'upsatirs')] = 'upstairs'
    
    left_frames,left_labels=generate_sequence(left_pocket, a_labels, N_TIME_STEPS , STEP)
    right_frames,right_labels=generate_sequence(right_pocket, a_labels, N_TIME_STEPS , STEP)
    frames=left_frames+right_frames
    labels=left_labels+right_labels
    return frames,labels

#%%
frames={}
labels={}
for i in participant_list:
    frames[i],labels[i]= make_frames(df_collection[i])

#%%



train_users= [0,2,3,5,6,7,9]
test_users= [1,4,8]

train_X=[]
train_y=[]
test_X=[]
test_y=[]

for i in train_users:
    train_X += frames[participant_list[i]]
    train_y += labels [participant_list[i]]
for i in test_users:
    test_X += frames[participant_list[i]]
    test_y += labels [participant_list[i]]


#%%


# reshape input segments and one-hot encode labels
def reshape_segments(x, y, n_time_steps, n_features):
    
    x_reshaped = np.asarray(x, dtype= np.float32).reshape(-1, n_time_steps, n_features)
    y_reshaped = np.asarray(pd.get_dummies(y), dtype = np.float32)
    return x_reshaped, y_reshaped

X_train, y_train = reshape_segments(train_X, train_y, N_TIME_STEPS, N_FEATURES)
X_test, y_test = reshape_segments(test_X, test_y, N_TIME_STEPS, N_FEATURES)

#%%


from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout

N_CLASSES = 7
N_HIDDEN_UNITS = 32
L2 = 0.000001


def get_model1():

      model = Sequential() 
      model.add(Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(N_TIME_STEPS,N_FEATURES)))
      model.add(Dropout(0.2))
      model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
      model.add(Dropout(0.2))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Flatten())
      model.add(Dense(200, activation='relu'))
      model.add(Dropout(0.3))
      model.add(BatchNormalization())
      model.add(Dense(7, activation='softmax'))

      opt = optimizers.RMSprop(lr=0.0001)

      model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
      return model
def get_model2():

      model = Sequential() 
      model.add(Conv1D(filters=64, kernel_size=3, activation='relu',input_shape=(N_TIME_STEPS,N_FEATURES)))
      model.add(Dropout(0.1))
      model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
      model.add(Dropout(0.1))
      model.add(MaxPooling1D(pool_size=2))
      model.add(LSTM(100))
      model.add(Flatten())
      model.add(Dense(100, activation='relu'))
      model.add(Dropout(0.2))
      model.add(BatchNormalization())
      model.add(Dense(7, activation='softmax'))

      opt = optimizers.RMSprop(lr=0.0001)

      model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
      return model


#%%

model=get_model1()
model.summary()
#%%

BATCH_SIZE = 64
N_EPOCHS = 100
es= EarlyStopping(monitor='val_loss',
                              patience=10,
                              mode='auto')
history=model.fit(X_train, y_train,
          batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          validation_data=(X_test, y_test), callbacks=[es])

#%%

plt.title('Accuracies vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.plot(history.history['accuracy'], label='Training Fold 1')
plt.legend()
plt.grid(False)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show()

#%%
plt.title("Training session's progress over iterations")
plt.xlabel('Training iteration')
plt.ylabel('Training Progress(Accuracy values)')
plt.plot(history.history['accuracy'], label='Train accuracies', color='blue')
plt.plot(history.history['val_accuracy'], label='Test accuracies', color='red')
plt.legend()
plt.ylim(.7, 1)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show()


#%%


score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%


plt.title("Training session's progress over iterations")
plt.xlabel('Training iteration')
plt.ylabel('Training progress(Loss values)')
plt.plot(history.history['loss'], label='Train losses', color='blue')
plt.plot(history.history['val_loss'], label='Test losses', color='red')
plt.legend()
plt.ylim(0, 1.5)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show()

#%%

import seaborn as sns
y_pred_ohe = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)

LABELS = ['Biking' ,' Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

plt.figure(figsize=(16, 14))
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show();


#%%
