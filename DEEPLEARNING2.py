#WE ONLY SHOW FOR DAILY CASES IN FRANCE. SAME PROCEDURE IS USED FOR OTHER COUNTRIES
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
df=pd.read_excel("FRANCEBEGINNING.xlsx")
print('Number of rows and columns:', df.shape)
df.head(5)
#LSTM
#Splitting the dataset
training_set = df.iloc[:446, 1:2].values
test_set = df.iloc[446:, 1:2].values
##### Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure from training set with 30 time-steps and 1 output
X_train, y_train = [], []
for i in range(30,446):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#X_train.shape
model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

"Compiling the RNN using adam as the optimizer as it is found to slightly outperform other learning algorithms ..........(1)"
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Getting the predicted 
def data_trans():
    dataset_train = df.iloc[:446, 1:2]
    dataset_test = df.iloc[446:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 30:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(30, 141):
        X_test.append(inputs[i-30:i, 0])
    X_test = np.array(X_test)
    #X_test.shape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)
    return dataset_train, dataset_test, X_test
dataset_train, dataset_test, X_test = data_trans()
predicted_cases = model.predict(X_test)
predicted_cases = sc.inverse_transform(predicted_cases)
predicted_cases.shape
mse = mean_squared_error(dataset_test,predicted_cases)
relative_rmse = np.sqrt(np.mean((predicted_cases - dataset_test)**2) ) / ( np.max(dataset_test)-np.min(dataset_test))
mape = mean_absolute_percentage_error(dataset_test,predicted_cases)
print("Mean Square Error = ", mse)
print("Relative rmse = ", relative_rmse.values)
print("MAPE = ", mape)

#GRU
regressorGRU = Sequential()
# layers with Dropout regularisation
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))# layer1
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))# layer2
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))# layer3
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50, activation='tanh'))# layer4
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer='adam',loss='mean_squared_error')
regressorGRU.summary()
# Fitting to the training set
regressorGRU.fit(X_train,y_train,epochs=100,batch_size=32)
dataset_train, dataset_test, X_test = data_trans()

GRU_predicted_cases = regressorGRU.predict(X_test)
GRU_predicted_cases = sc.inverse_transform(GRU_predicted_cases)
mse = mean_squared_error(dataset_test,GRU_predicted_cases)
relative_rmse = np.sqrt(np.mean((GRU_predicted_cases - dataset_test)**2) ) / ( np.max(dataset_test)-np.min(dataset_test))
mape = mean_absolute_percentage_error(dataset_test,GRU_predicted_cases)
print("Mean Square Error = ", mse)
print("Relative rmse = ", relative_rmse.values)
print("MAPE = ", mape)
#CNN
from keras.layers.convolutional import Conv1D, MaxPooling1D
Convmodel =Sequential()
Convmodel.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1)))
Convmodel.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1)))
Convmodel.add(MaxPooling1D(pool_size=2))
Convmodel.add(Flatten())
Convmodel.add(Dense(16, activation='relu'))
Convmodel.add(Dense(1, activation='sigmoid'))
Convmodel.compile(optimizer='adam',loss='mean_squared_error')
Convmodel.summary()
#Fitting the training set
checkpoint = callbacks.ModelCheckpoint('New_case-network.h5', monitor='val_loss', verbose=0,save_best_only=True,save_weights_only=True,
                                       mode='auto',period=1)
callback = [checkpoint]
json = 'New_case-network.h5'
model_json = model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
Convmodel.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,verbose=2,callbacks=callback)
dataset_train, dataset_test, X_test = data_trans()


CNN_predicted_cases = Convmodel.predict(X_test)
CNN_predicted_cases = sc.inverse_transform(CNN_predicted_cases)
mse = mean_squared_error(dataset_test,CNN_predicted_cases)
relative_rmse = np.sqrt(np.mean((CNN_predicted_cases - dataset_test)**2) ) / ( np.max(dataset_test)-np.min(dataset_test))
mape = mean_absolute_percentage_error(dataset_test,CNN_predicted_cases)
print("Mean Square Error = ", mse)
print("Relative rmse = ", relative_rmse.values)
print("MAPE = ", mape)
#DNN
from tensorflow.keras import layers
Dnnmodels = Sequential()

Dnnmodels.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],1)))
Dnnmodels.add(layers.BatchNormalization())
Dnnmodels.add(Flatten())
Dnnmodels.add(layers.Dense(8, activation='relu'))
Dnnmodels.add(layers.BatchNormalization())
Dnnmodels.add(layers.Dense(1, activation='sigmoid'))
  
Dnnmodels.compile(optimizer='adam',loss='mean_squared_error')
Dnnmodels.summary()
#Fitting the training set

checkpoint = callbacks.ModelCheckpoint('New_case-network.h5', monitor='val_loss', verbose=0,save_best_only=True,save_weights_only=True,
                                       mode='auto',period=1)
callback = [checkpoint]
json = 'New_case-network.h5'
model_json = model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
    
    
Dnnmodels.fit(X_train, y_train,batch_size=32, epochs=100,validation_split=0.2, verbose=2,callbacks=callback)
dataset_train, dataset_test, X_test = data_trans()

DNN_predicted_cases = Dnnmodels.predict(X_test)
DNN_predicted_cases = sc.inverse_transform(DNN_predicted_cases)
mse = mean_squared_error(dataset_test,DNN_predicted_cases)
relative_rmse = np.sqrt(np.mean((DNN_predicted_cases - dataset_test)**2) ) / ( np.max(dataset_test)-np.min(dataset_test))
mape = mean_absolute_percentage_error(dataset_test,DNN_predicted_cases)
print("Mean Square Error = ", mse)
print("Relative rmse = ", relative_rmse.values)
print("MAPE = ", mape)

# Visualising the results
plt.rcParams['figure.figsize'] = [10, 5]

plt.plot(df.loc[446:, 'date'],dataset_test.values,label = 'Real Cases')
plt.plot(df.loc[446:, 'date'],predicted_cases, label = 'Predicted CasesLSTM')
plt.plot(df.loc[446:, 'date'],GRU_predicted_cases,  label = 'Predicted CasesGRU')
plt.plot(df.loc[446:, 'date'],CNN_predicted_cases,  label = 'Predicted CasesCNN')
plt.plot(df.loc[446:, 'date'],DNN_predicted_cases,  label = 'Predicted CasesDNN')
plt.xticks(np.arange(0,127,20))
plt.title('France Covid Cases Prediction')
plt.xlabel('Time')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()

