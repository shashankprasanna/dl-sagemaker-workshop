---
title: "Image Classification with built-in algorithms"
weight: 1
---

## Step 1 - Launch SageMaker Studio Notebook
Once you launch SageMaker Studio follow the instructions below to start a notebook.

#### A. Go to File-> New Launcher
![newlauncherImg](https://aiml-data.s3.amazonaws.com/workshop/new-launcher.png)

#### B. Select MXNET Optimized for GPU
![optimizedGPUImg](https://aiml-data.s3.amazonaws.com/workshop/Choose-mxnet.png)

#### C. Select Python 3
![python3Img](https://aiml-data.s3.amazonaws.com/workshop/select-python3.png)

## Step 2 - Setting up Permissions and environment variables
In this step we setup some prequisites so our notebook can authenticate and link to other AWS Services.
#### There are 3 parts to this:
- Define the S3 bucket that you want to use for training and model data.
- Create a role for permissions
- Specify the built-in algorithm that you want to use.

Add the following code to your notebook:
```
%%time
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
print(role)

sess = sagemaker.Session()
bucket=sess.default_bucket()
prefix = 'ic-fulltraining'
```

```
from sagemaker.amazon.amazon_estimator import get_image_uri

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
print (training_image)
```

![preReqImg](https://aiml-data.s3.amazonaws.com/workshop/prereqs.png)

## Step 3 Download the dataset and upload it to Amazon S3
In this step we'll be downloading the CIFAR-10 training dataset and upload that to Amazon S3. As mentioned above this dataset consists of 10 classes with 40,000 images for training, 10,000 for validation and another 10,000 for testing. We can download all of these images from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and then convert them into to the recommended input format for image classification which is Apache MXNet RecordIO.

We can also just download the CIFAR-10 dataset already in the RecordIO format, which is what we'll be doing for this workshop.You can add the code below to your notebook to get the dataset and then upload it to our S3 bucket.

```
import os
import urllib.request
import boto3

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)


def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)

#Cifar-10
download('http://data.mxnet.io/data/cifar10/cifar10_train.rec')
download('http://data.mxnet.io/data/cifar10/cifar10_val.rec')
```

```
s3train = 's3://{}/{}/train/'.format(bucket, prefix)
s3validation = 's3://{}/{}/validation/'.format(bucket, prefix)

# upload the lst files to train and validation channels
!aws s3 cp cifar10_train.rec $s3train --quiet
!aws s3 cp cifar10_val.rec $s3validation --quiet
```

![datasetImg](https://aiml-data.s3.amazonaws.com/workshop/download-dataset.png)

Once we have the data available in the correct format for training, the next step is to actually train the model using the data. After setting training parameters, we kick off training, and poll for status until training is completed.

## Step 4 - Training the model
Now that we are done with all the setup that is needed, we are ready to train our model. To begin, we create a sageMaker.estimator.Estimator object. This estimator will launch the training job.

#### Set Training Parameters

- **Training instance count**: This is the number of instances on which to run the training. When the number of instances is greater than one, then the image classification algorithm will run in distributed settings.
- **Training instance type**: This indicates the type of machine on which to run the training. Typically, we use GPU instances for these training
- **Output path**: This the s3 folder in which the training output is stored`

Add the following code below:

```
s3_output_location = 's3://{}/{}/output'.format(bucket, prefix)
ic = sagemaker.estimator.Estimator(training_image,
                                         role,
                                         train_instance_count=1,
                                         train_instance_type='ml.p2.xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         input_mode= 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)
```

#### Set Hyperparameters

Apart from the above set of parameters, there are hyperparameters that are specific to the algorithm. These are:

- **num_layers**: The number of layers (depth) for the network. We use 18 in this samples but other values such as 50, 152 can be used.
- **image_shape**: The input image dimensions,'num_channels, height, width', for the network. It should be no larger than the actual image size. The number of channels should be same as the actual image.
- **num_classes**: This is the number of output classes for the CIFAR-10 dataset. We set this value to 10 since there are 10 classes.  
- **num_training_samples**: This is the total number of training samples.
- **mini_batch_size**: The number of training samples used for each mini batch. In distributed training, the number of training samples used per batch will be N * mini_batch_size where N is the number of hosts on which training is run.
- **epochs**: Number of training epochs.
- **learning_rate**: Learning rate for training.
- **top_k**: Report the top-k accuracy during training.
- **precision_dtype**: Training datatype precision (default: float32). If set to 'float16', the training will be done in mixed_precision mode and will be faster than float32 mode

Add the following below:

```
ic.set_hyperparameters(num_layers=18,
                             image_shape = "3,28,28",
                             num_classes=10,
                             num_training_samples=50000,
                             mini_batch_size=128,
                             epochs=5,
                             learning_rate=0.01,
                             top_k=2,
                             precision_dtype='float32')
```

#### Specify data used for training
Add the following:

```
train_data = sagemaker.session.s3_input(s3train, distribution='FullyReplicated',
                        content_type='application/x-recordio', s3_data_type='S3Prefix')
validation_data = sagemaker.session.s3_input(s3validation, distribution='FullyReplicated',
                             content_type='application/x-recordio', s3_data_type='S3Prefix')

data_channels = {'train': train_data, 'validation': validation_data}
```

#### Start the training

```
ic.fit(inputs=data_channels, logs=True)
```

#### Deploy model
predictor = ic.deploy(initial_instance_count = 1, instance_type = 'ml.c5.large')
