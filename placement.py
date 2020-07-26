# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:14:10 2020

@author: Shikhar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Placement_Data_Full_Class.csv")
df.isnull().sum()
df.set_index(['sl_no'], inplace = True) 
df['salary']=df['salary'].fillna(value = df['salary'].mean())
from sklearn.preprocessing import LabelEncoder  
label_encoder_df= LabelEncoder()  
df['status']= label_encoder_df.fit_transform(df['status'])  
df['specialisation'].value_counts()
x = df.iloc[:,:13].values
y = df.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
lb_1 = LabelEncoder()
lb_2 = LabelEncoder()
lb_3 = LabelEncoder()
lb_4 = LabelEncoder()
lb_5 = LabelEncoder()
lb_6 = LabelEncoder()
lb_7 = LabelEncoder()
x[:,0] = lb_1.fit_transform(x[:,0])
x[:,2] = lb_2.fit_transform(x[:,2])
x[:,4] = lb_3.fit_transform(x[:,4])
x[:,5] = lb_4.fit_transform(x[:,5])
x[:,7] = lb_5.fit_transform(x[:,7])
x[:,8] = lb_5.fit_transform(x[:,8])
x[:,10] = lb_5.fit_transform(x[:,10])
#y[:,0] = lb_5.fit_transform(y[:,0])
ct = ColumnTransformer([("hsc_s", OneHotEncoder(), [5])], remainder = 'passthrough')
ct2 = ColumnTransformer([("degree_t", OneHotEncoder(), [7])], remainder = 'passthrough')
x = ct.fit_transform(x)
x = ct2.fit_transform(x)
x[:,1:]


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2 ,random_state = 56)

from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout
sc= StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.transform(test_x)




from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform',input_dim = 16))
classifier.add(Dropout(rate =0.1))

classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform'))
classifier.add(Dropout(rate =0.1))


classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))


classifier.compile(optimizer='adam', loss = 'binary_crossentropy',metrics= ['accuracy'])

classifier.fit(train_x,train_y,batch_size =25,epochs = 100)



y_pred = classifier.predict(test_x)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,y_pred )


#def classifier_build(optimizer):
 #   classifier = Sequential()
#
 #   classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform',input_dim = 14))
#
 #   classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform'))
#
 #   classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))
#

#    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy',metrics= ['accuracy'])
 #   return classifier    
#classifier = KerasClassifier(build_fn=classifier_build)


#para = {'batch_size':[25,32],'epochs': [100,500],
 #       'optimizer':['adam','rmsprop']}

#gscv = GridSearchCV(estimator = classifier,
 #                   param_grid= para,
  #                  scoring ='accuracy',
   #                 cv = 10)

#gscv = gscv.fit(train_x,train_y)

#best_para =gscv.best_params_ 
#best_acc = gscv.best_score_


from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



def classifier_build(optimizer):
    classifier = Sequential()

    classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform',input_dim = 16))

    classifier.add(Dense(units = 7,activation= 'relu',kernel_initializer ='uniform'))

    classifier.add(Dense(units = 1,activation='sigmoid',kernel_initializer= 'uniform'))


    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy',metrics= ['accuracy'])
    return classifier    
classifier = KerasClassifier(build_fn=classifier_build('rmsprop'))


para = {'batch_size':[25,32],'epochs': [100,500],
        'optimizer':['adam','rmsprop']}

gscv = GridSearchCV(estimator = classifier,
                    param_grid= para,
                    scoring ='accuracy',
                    cv = 10)

gscv = gscv.fit(train_x,train_y)

best_para =gscv.best_params_ 
best_acc = gscv.best_score_

