import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle

df=pd.read_csv("Churn_Modelling.csv")
print(df.head())

## preprocess the data

## drop irrelevant features
data=df.drop(['RowNumber','CustomerId','Surname'],axis=1)

print(data)

print(data.info())

print(data['Gender'].unique())
## encode the categorical variable
label_encoder_gender=LabelEncoder()
data['Gender']=label_encoder_gender.fit_transform(data['Gender'])
print(data['Gender'])

print(data['Geography'].unique())

##One hot encode geography column
from sklearn.preprocessing import OneHotEncoder
onehot_encoder_geo=OneHotEncoder(sparse_output=False)
geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']])

print(geo_encoder)

print(onehot_encoder_geo.get_feature_names_out(['Geography']))

geo_encoded_df=pd.DataFrame(geo_encoder,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

print(geo_encoded_df)

## combine one hot encoder columns with the original data
data=pd.concat([data.drop('Geography',axis=1),geo_encoded_df],axis=1)
print(data.head())


## save the encoders and scaler

with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl','wb') as file:
    pickle.dump(onehot_encoder_geo,file)

## divide the dataset into independednt and dependednt features
x=data.drop('Exited',axis=1)
y=data['Exited']


##split the data in training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

## scale down the features
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

with open ('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)




##ANN
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping,TensorBoard
import datetime
import warnings
warnings.filterwarnings('ignore')

## Build our ANN Model

print(x_train.shape)


model=Sequential([
    Dense(64,activation='relu',input_shape=(x_train.shape[1],)), ## HL1 connected with input layer
    Dense(32,activation='relu'),  ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]
)

print(model.summary())


## compile the model

opt=keras.optimizers.Adam(learning_rate=0.01)
loss=keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])


## setup the tensorflow
from keras.callbacks import EarlyStopping,TensorBoard
log_dir="logs/fit"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

## Setup early stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

## training the model
history=model.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.keras')

