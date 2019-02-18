
# coding: utf-8

# # TensorFlow Classification

# In[1]:


import pandas as pd


# In[2]:


diabetes = pd.read_csv('diabetes.csv')


# In[3]:


diabetes.head()


# ### Normalize the dataset

# In[5]:


cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']


# In[6]:


diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# ### Create the Feature Columns to be accessed in the model

# In[7]:


import tensorflow as tf


# In[10]:


num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#Categorical column
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# Alternative
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])


# ### Putting them together

# In[11]:


feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]


# ### Train Test Split

# In[18]:


diabetes.head()


# In[15]:


#Drop the Class column because it will be predicted by model and will be provided as a label to train test split
x_data = diabetes.drop('Class',axis=1)


# In[12]:


labels = diabetes['Class']


# In[13]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)


# ### Input Function

# In[259]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# ### Creating the Model

# In[260]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[261]:


model.train(input_fn=input_func,steps=1000)


# In[262]:


# Useful link ofr your own data
# https://stackoverflow.com/questions/44664285/what-are-the-contraints-for-tensorflow-scope-names


# ## Evaluation

# In[289]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[281]:


results = model.evaluate(eval_input_func)


# In[290]:


results


# ## Predictions

# In[293]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[304]:


# Predictions is a generator! 
predictions = model.predict(pred_input_func)


# In[305]:


list(predictions)


# # DNN Classifier

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)


# In[21]:


#Assigned group is the numeric feature column for 'Group'
embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
featureColumns = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree,
             embedded_group_column, age_buckets]


# In[22]:


inputFunction = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=1000,shuffle=True)


# In[25]:


dnnClassifierModel = tf.estimator.DNNClassifier(hidden_units=[512, 256, 128],
                                                feature_columns=featureColumns,
                                                n_classes=2,
                                                activation_fn=tf.nn.tanh,
                                                optimizer=lambda: tf.train.AdamOptimizer(
                                                    learning_rate=tf.train.exponential_decay(learning_rate=0.001,
                                                    global_step=tf.train.get_global_step(),
                                                    decay_steps=1000,
                                                    decay_rate=0.96)))

dnnClassifierModel.train(input_fn=inputFunction,steps=1000)


# In[27]:


dnnClassifierModel.train(input_fn=inputFunction,steps=1000)


# In[28]:


evaluateInputFunction = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)
dnnClassifierModel.evaluate(evaluateInputFunction)


# # Linear Classificatication with TensorFlow and Dense Neural Nets
