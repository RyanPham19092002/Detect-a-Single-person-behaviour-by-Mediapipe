import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# Đọc dữ liệu
clapping_df = pd.read_csv("CLAPPING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")
nothing_dk = pd.read_csv("NOTHING.txt")


X = []
y = []

#ghép 10 frame lại để train model và dự đoán
no_of_timesteps = 10

dataset = handswing_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(0)

dataset = clapping_df.iloc[:,1:].values
n_sample = len(dataset)
#lấy 10 time_step 1 lần
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])  #lấy từ i-10 tới i
    y.append(1)


dataset = nothing_dk.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append(2)

X, y = np.array(X), np.array(y)
y = to_categorical(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model  = Sequential()   #xây dựng model từ lớp sequntial
model.add(LSTM(units = 128, return_sequences = True, input_shape = (X.shape[1], X.shape[2]))) #X1 là sl timestep, X2 là sl đặc trưng
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))        #rs cờ trả về chuỗi hay kh
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=3, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
model.save("swing_clap_nothing.h5")
