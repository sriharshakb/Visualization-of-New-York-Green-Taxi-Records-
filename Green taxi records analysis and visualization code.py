#!/usr/bin/env python
# coding: utf-8

# In[132]:


import os
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

bucket = 'fianlprojectcc'# enter your s3 bucket where you will copy data and model artifacts
prefix = 'sagemaker/Greentaxi_predictions' # place to upload training files within the bucket

import pandas as pd
from pandas import Grouper
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import sagemaker.amazon.common as smac


# In[133]:


data = pd.read_csv('https://fianlprojectcc.s3-us-west-1.amazonaws.com/2014-01.csv')

data.columns = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Store_and_fwd_flag', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Passenger_count', 'Trip_distance', 'Fare_amount', 'Extra', 'tax', 'Tip_amount', 'Tolls_amount', 'Ehail_fee', 'Total_amount', 'Payment_type', 'Trip_type'] 

data.to_csv("data.csv", sep=',', index=False)


# In[158]:



columns = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Total_amount', 'Trip_distance']

df = pd.read_csv("data.csv", usecols = columns)
df=df.dropna()
#print(df)

df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'],format='%m/%d/%Y %H:%M')#2015-01-31 23:00:09
df['Lpep_dropoff_datetime'] = pd.to_datetime(df['Lpep_dropoff_datetime'],format='%m/%d/%Y %H:%M')
df['trip_duration']= df['Lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
data = df[~(df['lpep_pickup_datetime'] < '2014-01-01')]
data = data[~(df['lpep_pickup_datetime'] > '2014-02-01')]
data['count']=1
#print(data)
#print(data)

reqcolumns = ['Trip_distance', 'Total_amount']
reqcolumns2 = ['trip_duration', 'Total_amount']
data_distance=data[reqcolumns]
data_time = data[reqcolumns2]
#print(data_time)

by_days=data.groupby(Grouper(key='lpep_pickup_datetime', freq='d')).sum()
by_hour=data.groupby(Grouper(key='lpep_pickup_datetime', freq='H')).sum()

print(by_days)

# data atribute has non group data


# In[130]:


X = data.iloc[:, 0].values
y = data.iloc[:, 1].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


# In[131]:


X


# In[111]:


Y


# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[113]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[115]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Amount vs Distance (Training set)')
plt.xlabel('Distance(miles)')
plt.ylabel('Amount($)')
plt.show()


# In[116]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Amount vs Distance (Test set)')
plt.xlabel('Distance(miles)')
plt.ylabel('Amount($)')
plt.show()


# In[120]:


X1 = data_time.iloc[:, 0].values
y1 = data_time.iloc[:, 1].values
X1 = X.reshape(-1, 1)
y1 = y.reshape(-1, 1)


# In[121]:


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)


# In[122]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train1, y_train1)
y_pred_time = regressor.predict(X_test1)


# In[123]:


# Visualising the Training set results
plt.scatter(X_train1, y_train1, color = 'red')
plt.plot(X_train1, regressor.predict(X_train1), color = 'blue')
plt.title('Amount vs Time (Training set)')
plt.xlabel('Time(min)')
plt.ylabel('Amount($)')
plt.show()


# In[125]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X, X1, Y, c='red', cmap='Greens', marker= 'o')

ax.set_xlabel("Distance")
ax.set_ylabel("Time")
ax.set_zlabel("Amount")
plt.plot(X_train1, regressor.predict(X_train1), color = 'blue')
plt.show()


# In[126]:


# Visualising the Test set results
plt.plot(y_test, color= 'red')
plt.plot(y_pred, color = 'blue')
plt.title('Amount vs Distance (Test set)')
plt.xlabel('Distance(miles)')
plt.ylabel('Amount($)')
plt.show()


# In[127]:


# Visualising the Test set results
plt.plot(y_test1, color= 'red')
plt.plot(y_pred_time, color = 'blue')
plt.title('Amount vs Distance (Test set)')
plt.xlabel('Distance(miles)')
plt.ylabel('Amount($)')
plt.show()


# In[128]:


y_test


# In[129]:


y_pred


# In[249]:


test_pred = np.array([r['score'] for r in result['predictions']])

test_pred_class = (y_pred > 0.5)+0;
test_pred_baseline = np.repeat(np.median(y_train), len(y_test))

prediction_accuracy = np.mean((y_test == test_pred_class))*100
baseline_accuracy = np.mean((y_test == test_pred_baseline))*100

print("Prediction Accuracy:", round(prediction_accuracy,1), "%")
print("Baseline Accuracy:", round(baseline_accuracy,1), "%")


# In[196]:


by_days['day'] = by_days.index
by_days['daycount'] = by_days['day'].dt.day
print(by_days)


# In[209]:


X2 = by_days.iloc[:, 4].values
Y2 = by_days.iloc[:, 2].values
X2 = X2.reshape(-1, 1)
Y2 = Y2.reshape(-1, 1)


# In[211]:


X2


# In[229]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.plot(Y2, color='red')
plt.xlabel('day')
plt.ylabel('trip records count')
plt.show()
plt.show()


# In[216]:


X2


# In[217]:


Y2


# In[218]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.2, random_state = 0)


# In[219]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X2_train, Y2_train)
Y2_pred = regressor.predict(X2_test)


# In[220]:


# Visualising the Training set results
plt.scatter(X2_train, Y2_train, color = 'red')
plt.plot(X2_train, regressor.predict(X2_train), color = 'blue')
plt.title('Amount vs Distance (Training set)')
plt.xlabel('Distance(miles)')
plt.ylabel('Amount($)')
plt.show()


# In[224]:


X3 = by_days.iloc[:, 4].values
Y3 = by_days.iloc[:, 1].values
X3 = X3.reshape(-1, 1)
Y3 = Y3.reshape(-1, 1)


# In[225]:


Y3


# In[228]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.plot(Y3, color='red')
plt.xlabel('day')
plt.ylabel('REVENUE')
plt.show()


# In[252]:


role = get_execution_role()
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# specify columns extracted from wbdc.names
data.columns = ["id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"] 


# In[253]:


rand_split = np.random.rand(len(data))
train_list = rand_split < 0.8
val_list = (rand_split >= 0.8) & (rand_split < 0.9)
test_list = rand_split >= 0.9

data_train = data[train_list]
data_val = data[val_list]
data_test = data[test_list]

train_y = ((data_train.iloc[:,1] == 'M') +0).as_matrix();
train_X = data_train.iloc[:,2:].as_matrix();

val_y = ((data_val.iloc[:,1] == 'M') +0).as_matrix();
val_X = data_val.iloc[:,2:].as_matrix();

test_y = ((data_test.iloc[:,1] == 'M') +0).as_matrix();
test_X = data_test.iloc[:,2:].as_matrix();


# In[254]:


train_file = 'linear_train.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, train_X.astype('float32'), train_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', train_file)).upload_fileobj(f)


# In[255]:


validation_file = 'linear_validation.data'

f = io.BytesIO()
smac.write_numpy_to_dense_tensor(f, val_X.astype('float32'), val_y.astype('float32'))
f.seek(0)

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation', validation_file)).upload_fileobj(f)


# In[256]:


# See 'Algorithms Provided by Amazon SageMaker: Common Parameters' in the SageMaker documentation for an explanation of these values.
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner')


# In[261]:


linear_job = 'Greentaxi-projectcc' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())



print("Job name is:", linear_job)

linear_training_params = {
    "RoleArn": role,
    "TrainingJobName": linear_job,
    "AlgorithmSpecification": {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    },
    "ResourceConfig": {
        "InstanceCount": 1,
        "InstanceType": "ml.c4.2xlarge",
        "VolumeSizeInGB": 10
    },
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/train/".format(bucket, prefix),
                    "S3DataDistributionType": "ShardedByS3Key"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/validation/".format(bucket, prefix),
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "CompressionType": "None",
            "RecordWrapperType": "None"
        }

    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://{}/{}/".format(bucket, prefix)
    },
    "HyperParameters": {
        "feature_dim": "30",
        "mini_batch_size": "100",
        "predictor_type": "regressor",
        "epochs": "10",
        "num_models": "32",
        "loss": "absolute_loss"
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 60 * 60
    }
}


# In[262]:


get_ipython().run_cell_magic('time', '', "\nregion = boto3.Session().region_name\nsm = boto3.client('sagemaker')\n\nsm.create_training_job(**linear_training_params)\n\nstatus = sm.describe_training_job(TrainingJobName=linear_job)['TrainingJobStatus']\nprint(status)\nsm.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=linear_job)\nif status == 'Failed':\n    message = sm.describe_training_job(TrainingJobName=linear_job)['FailureReason']\n    print('Training failed with the following error: {}'.format(message))\n    raise Exception('Training job failed')")


# In[264]:


linear_endpoint_config = 'Greentaxifinalprojectcc' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
print(linear_endpoint_config)
create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=linear_endpoint_config,
    ProductionVariants=[{
        'InstanceType': 'ml.m4.xlarge',
        'InitialInstanceCount': 1,
        'ModelName': linear_job,
        'VariantName': 'AllTraffic'}])

print("Endpoint Config Arn: " + create_endpoint_config_response['EndpointConfigArn'])


# In[ ]:




