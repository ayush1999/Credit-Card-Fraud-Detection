from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.models import Sequential
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')

x=[]
y=[]
for element in data[1:]:
    y.append(element[-1].tolist())
    x.append(element[1:29].tolist())

def baseline_model():
    
    model = Sequential()
    model.add(Dense(100, input_dim=28, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


model = baseline_model()
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'], optimizer='adam')
model.fit(x, y, epochs=2, batch_size=32)

predictions = model.predict(x)

rounded = []
for  i in predictions:
    rounded.append(round(i[0]))
count =0 
for i in range(len(x)):
    if rounded[i]!= y[i]:
        count= count +1

print("Number of wrong prediction {}".format(count))