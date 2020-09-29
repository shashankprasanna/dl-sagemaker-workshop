---
title: "Image Classification with TensorFlow"
weight: 3
---

## Download the dataset and upload it to Amazon S3

In this step you&#39;ll download the training dataset and upload it to Amazon S3. You&#39;ll download the CIFAR-10 dataset to the Amazon SageMaker Studio Notebook, convert it into TensorFlow supported [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format and upload it to Amazon S3. Converting to TFRecord format is optional, but recommended when dealing with larger datasets.

1. In the Amazon SageMaker Studio homepage, click the dropdown menu under **Select a SageMaker image to launch your activity** and select **TensorFlow (optimized for CPU)**
![](/images/image009.png)
**Note:** In this step you&#39;re choosing a CPU instance which is used to run the SageMaker Notebook which is used to download the dataset, build your training scripts and submit Amazon SageMaker training jobs and visualize results. The training job itself will run on a separate instance type that you can specify, such as a GPU instance.

1. Under **Notebook** click on **Python 3**
![](/images/image010.png)

1. Within the new Jupyter notebook, paste the following code to first download the generate\_cifar10\_tfrecords.py script, which will then download the CIFAR-10 dataset and convert it into TFRecord format. To run each code cell, simply select the cell and click on the play button show in the screenshot or hit shift + enter on your keyboard. You can put all the code into a single cell or separate them for better readability as shown in the screenshot. If you put the code into separate cells, you&#39;ll need to select each cell in order and click the play button to run them.
```
!wget https://raw.githubusercontent.com/awslabs/amazon-sagemaker-examples/master/advanced_functionality/tensorflow_bring_your_own/utils/generate_cifar10_tfrecords.py
```
```
!pip install ipywidgets
!python generate_cifar10_tfrecords.py --data-dir cifar10
```
![](/images/image011.png)

1. To upload the dataset to your default Amazon SageMaker Amazon S3 bucket, paste and run the following code. You should see the Amazon S3 location for your dataset as the output.

```
import time, os, sys
import sagemaker, boto3
import numpy as np
import pandas as pd

sess = boto3.Session()
sm   = sess.client('sagemaker')
role = sagemaker.get_execution_role()
sagemaker_session = sagemaker.Session(boto_session=sess)

datasets = sagemaker_session.upload_data(path='cifar10', key_prefix='datasets/cifar10-dataset')
datasets
```
![](/images/image012.png)

## Create an Amazon SageMaker Experiment to track and manage training jobs

In this step you&#39;ll create an Amazon SageMaker Experiment which lets you organize, track, and compare your machine learning training jobs.

Copy and paste the following code into a new code cell and click run.
```
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent

training_experiment = Experiment.create(
                                experiment_name = "sagemaker-training-experiments",
                                description     = "Experiment to track cifar10 training trials",
                                sagemaker_boto_client=sm)
```
![](/images/image013.png)
 The code uses the smexperiments python package to create an experiment named &quot;sagemaker-training-experiments&quot;. This package comes pre-installed on Amazon SageMaker Studio Notebooks. You can customize the experiment name and description.

After running the code cell, click on the flask symbol on the left navigation menu of Amazon SageMaker Studio. This will open the Experiments management pane. Here you should see a new Experiment with the name **sagemaker-training-experiments** you specified
![](/images/image014.png)

## Create the trial and training script

To train a classifier on the CIFAR-10 dataset, you need a training script. In this step, you create your trial and training script for the TensorFlow training job. Each trial is an iteration of your end-to-end training job. In addition to the training job, the trial can also track preprocessing, post processing jobs as well as datasets and other metadata. A single experiment can include multiple trials which makes it easy for you to track multiple iterations over time within the Amazon SageMaker Studio Experiments pane.
Complete the following steps to create a new trial and training scipt for the TensorFlow training job.


1. Create a new Trial and associate it with the Experiment you created earlier. Copy and paste the following code into a new code cell and run it.

```
single_gpu_trial = Trial.create(
    trial_name = 'sagemaker-single-gpu-training',
    experiment_name = training_experiment.experiment_name,
    sagemaker_boto_client = sm,
)

trial_comp_name = 'single-gpu-training-job'
experiment_config = {"ExperimentName": training_experiment.experiment_name,
                       "TrialName": single_gpu_trial.trial_name,
                       "TrialComponentDisplayName": trial_comp_name}
```
Each trial, as the name suggests, is an iteration of your end-to-end training job. In addition to the training job, it can also track preprocessing, post processing jobs as well as datasets and other metadata. A single experiment can include multiple trials which makes it easy for you to track multiple iterations over time within the Amazon SageMaker Studio Experiments pane.

1. Under the Experiment pane on the left, double-click on the experiment **sagemaker-training-experiments** and you should see a new Trial named **sagemaker-single-gpu-training**
![](/images/image015.png)

1. Click File > New > Text File to open up a new code editor, and paste the following TensorFlow code into the newly created file.
![](/images/image016.png)
This script implements TensorFlow code to read the CIFAR-10 dataset and train a resnet50 model. To train a different model, feel free to customize the **get\_model** function in the script below to include your custom model.
The text editor should look something like this:
![](/images/image017.png)
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
import argparse
import os
import re
import time

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10

def single_example_parser(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      # Dimensions of the images in the CIFAR-10 dataset.
      # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
      # input format.
      features = tf.io.parse_single_example(
          serialized_example,
          features={
              'image': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.int64),
          })
      image = tf.decode_raw(features['image'], tf.uint8)
      image.set_shape([DEPTH * HEIGHT * WIDTH])

      # Reshape from [depth * height * width] to [depth, height, width].
      image = tf.cast(
          tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
          tf.float32)
      label = tf.cast(features['label'], tf.int32)

      image = train_preprocess_fn(image)
      label = tf.one_hot(label, NUM_CLASSES)

      return image, label

def train_preprocess_fn(image):
      # Resize the image to add four extra pixels on each side.
      image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

      # Randomly crop a [HEIGHT, WIDTH] section of the image.
      image = tf.image.random_crop(image, [HEIGHT, WIDTH, DEPTH])

      # Randomly flip the image horizontally.
      image = tf.image.random_flip_left_right(image)
      return image

def get_dataset(filenames, batch_size):
      """Read the images and labels from 'filenames'."""
      # Repeat infinitely.
      dataset = tf.data.TFRecordDataset(filenames).repeat().shuffle(10000)

      # Parse records.
      dataset = dataset.map(single_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

      # Batch it up.
      dataset = dataset.batch(batch_size, drop_remainder=True)
      return dataset

def get_model(input_shape, learning_rate, weight_decay, optimizer, momentum):
      input_tensor = Input(shape=input_shape)
      base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                            weights='imagenet',
                                                            input_tensor=input_tensor,
                                                            input_shape=input_shape,
                                                            classes=None)
      x = Flatten()(base_model.output)
      predictions = Dense(NUM_CLASSES, activation='softmax')(x)
      model = Model(inputs=base_model.input, outputs=predictions)
      return model

def main(args):
      # Hyper-parameters
      epochs       = args.epochs
      lr           = args.learning_rate
      batch_size   = args.batch_size
      momentum     = args.momentum
      weight_decay = args.weight_decay
      optimizer    = args.optimizer

      # SageMaker options
      training_dir   = args.training
      validation_dir = args.validation
      eval_dir       = args.eval

      train_dataset = get_dataset(training_dir+'/train.tfrecords',  batch_size)
      val_dataset   = get_dataset(validation_dir+'/validation.tfrecords', batch_size)
      eval_dataset  = get_dataset(eval_dir+'/eval.tfrecords', batch_size)

      input_shape = (HEIGHT, WIDTH, DEPTH)
      model = get_model(input_shape, lr, weight_decay, optimizer, momentum)

      # Optimizer
      if optimizer.lower() == 'sgd':
          opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
      else:
          opt = Adam(lr=lr, decay=weight_decay)

      # Compile model
      model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      # Train model
      history = model.fit(train_dataset, steps_per_epoch=40000 // batch_size,
                          validation_data=val_dataset,
                          validation_steps=10000 // batch_size,
                          epochs=epochs)


      # Evaluate model performance
      score = model.evaluate(eval_dataset, steps=10000 // batch_size, verbose=1)
      print('Test loss    :', score[0])
      print('Test accuracy:', score[1])

      # Save model to model directory
      model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')


#%%
if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      # Hyper-parameters
      parser.add_argument('--epochs',        type=int,   default=10)
      parser.add_argument('--learning-rate', type=float, default=0.01)
      parser.add_argument('--batch-size',    type=int,   default=128)
      parser.add_argument('--weight-decay',  type=float, default=2e-4)
      parser.add_argument('--momentum',      type=float, default='0.9')
      parser.add_argument('--optimizer',     type=str,   default='sgd')

      # SageMaker parameters
      parser.add_argument('--model_dir',        type=str)
      parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
      parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
      parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])

      args = parser.parse_args()
      main(args)
```

1. Rename the file to include a python file extension by clicking on File > Rename File … and pasting **cifar10-training-sagemaker.py** under New Name and click Rename. Make sure that the new extension is &quot;.py&quot; and not &quot;.txt&quot;. After renaming the file, click File > Save File or hit Ctrl + S on your keyboard when the file is selected and open.
![](/images/image018.png)
![](/images/image019.png)

## Run the TensorFlow training job

In this step, you run a TensorFlow training job using Amazon SageMaker. Training models is easy with Amazon SageMaker. You specify the location of your dataset in Amazon S3 and type of training instance, and then Amazon SageMaker manages the training infrastructure for you. 

Complete the following steps to run the TensorFlow training job and then visualize the results.

In your Jupyter Notebook, copy and paste the following code block into the code cell and select Run. Then, take a closer look at the code.
Note: If a ResourceLimitExceeded appears, change the instance type to ml.c5.xlarge.
```
from sagemaker.tensorflow import TensorFlow

hyperparams={'epochs'       : 30,
             'learning-rate': 0.01,
             'batch-size'   : 256,
             'weight-decay' : 2e-4,
             'momentum'     : 0.9,
             'optimizer'    : 'adam'}

bucket_name = sagemaker_session.default_bucket()
output_path = f's3://{bucket_name}/jobs'
metric_definitions = [{'Name': 'val_acc', 'Regex': 'val_acc: ([0-9\\.]+)'}]

tf_estimator = TensorFlow(entry_point          = 'cifar10-training-sagemaker.py',
                          output_path          = f'{output_path}/',
                          code_location        = output_path,
                          role                 = role,
                          train_instance_count = 1,
                          train_instance_type  = 'ml.g4dn.xlarge',
                          framework_version    = '1.15.2',
                          py_version           = 'py3',
                          script_mode          = True,
                          metric_definitions   = metric_definitions,
                          sagemaker_session    = sagemaker_session,
                          hyperparameters      = hyperparams)

job_name=f'tensorflow-single-gpu-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}'
tf_estimator.fit({'training'  : datasets,
                  'validation': datasets,
                  'eval'      : datasets},
                 job_name = job_name,
                 experiment_config=experiment_config)
```
This code includes three parts:

- Specifies training job hyperparameters
- Calls an Amazon SageMaker Estimator function and provides training job details (name of the training script, what instance type to train on, framework version, etc.)
- Calls the fit function to initiate the training job

After training is complete, you should see final accuracy results, training time and billable time
![](/images/image020.png)

Amazon SageMaker automatically provisions the requested instances, downloads the dataset, pulls the TensorFlow container, downloads the training script, and starts training.

In this example, you submit an Amazon SageMaker training job to run on ml.g4dn.xlarge which is a GPU instance. Deep learning training is computationally intensive and GPU instances are recommended for getting results faster.   

tutorial-sagemaker-training-import-tensorflow
After training is complete, you should see final accuracy results, training time, and billable time.

![](/images/image021.png)

## Visualize training results

To view training summary in Studio click on the Experiment pane on the left, navigate to **sagemaker-training-experiments > sagemaker-single-gpu-training** and double-click on the newly created Trial Component for your training job.

![](/images/image022.png)

To visualize training performance using charts in Amazon SageMaker Studio, click on Charts > Add chart. On the right-hand side CHART PROPERTIES pane select the following:

  1. Chart type: Line
  2. X-axis dimension: Epochs
  3. Y-axis: val\_acc\_EVAL\_average

You should see a graph showing the change in evaluation accuracy as training progresses, ending with the final accuracy in Step 5.6

![](/images/image023.png)

You can also monitor the training job on Amazon Console. Navigate to Amazon Console > Amazon SageMaker > Training Jobs
![](/images/image024.png)

## Improve accuracy by running an Amazon SageMaker Automatic Model Tuning job to find the best model hyperparameters

In this step you&#39;ll run an Amazon SageMaker automatic model tuning job to find the best hyperparameters and improve upon the training accuracy obtained in Step 5. To run a model tuning job, you&#39;ll need to provide Amazon SageMaker with Hyperparameter ranges rather than fixed values, so that it can explore the hyperparameter space and automatically find the best values for you.

In a new code cell, paste the following code and run it.

```
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

hyperparameter_ranges = {
    'epochs'        : IntegerParameter(5, 30),
    'learning-rate' : ContinuousParameter(0.001, 0.1, scaling_type='Logarithmic'),
    'batch-size'    : CategoricalParameter(['128', '256', '512']),
    'momentum'      : ContinuousParameter(0.9, 0.99),
    'optimizer'     : CategoricalParameter(['sgd', 'adam'])
}

objective_metric_name = 'val_acc'
objective_type = 'Maximize'
metric_definitions = [{'Name': 'val_acc', 'Regex': 'val_acc: ([0-9\\.]+)'}]

tf_estimator = TensorFlow(entry_point          = 'cifar10-training-sagemaker.py',
                          output_path          = f'{output_path}/',
                          code_location        = output_path,
                          role                 = role,
                          train_instance_count = 1,
                          train_instance_type  = 'ml.g4dn.xlarge',
                          framework_version    = '1.15',
                          py_version           = 'py3',
                          script_mode          = True,
                          metric_definitions   = metric_definitions,
                          sagemaker_session    = sagemaker_session)

tuner = HyperparameterTuner(estimator             = tf_estimator,
                            objective_metric_name = objective_metric_name,
                            hyperparameter_ranges = hyperparameter_ranges,
                            metric_definitions    = metric_definitions,
                            max_jobs              = 4,
                            max_parallel_jobs     = 4,
                            objective_type        = objective_type)

job_name=f'tf-hpo-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}'
tuner.fit({'training'  : datasets,
           'validation': datasets,
           'eval'      : datasets},
            job_name = job_name)
```

This code includes four parts

  1. Specify range of values for hyperparameters. These could be integer ranges (eg. Epoch numbers), continuous ranges (eg. Learning rate) or categorical values (eg. Optimizer type sgd or adam).
  2. Call an Estimator function similar to the one in Step 5
  3. Create a HyperparameterTuner object with hyperparameter ranges, maximum number of jobs and number of parallel jobs to run
  4. Call the fit function to initiate hyperparameter tuning job

Note: Reduce max\_jobs variable to save tuning job cost. However, by reducing the number of tuning jobs, you reduce the chances of finding a better model. You must also reduce the max\_parallel\_jobs variable to a number less than or equal to max\_jobs. You can get results faster when max\_parallel\_jobs is equal to max\_jobs. You must ensure that max\_parallel\_jobs is lower than the instance limits of your AWS account to avoid running into resource errors.

![](/images/image025.png)

After hyperparameter training job is complete you can view the best hyperparameters in the Amazon SageMaker Console. Navigate to Amazon SageMaker Console > Training > Hyperparameter tuning jobs > Best training job. You&#39;ll see an improvement in the training accuracy (80%) compared to results in Step 5 (70%). Note: Your results may vary. You can further improve your results by increasing max\_jobs, relaxing the hyperparameter ranges and exploring other model architectures.

![](/images/image026.png)

## Deploy and test your model
You can deploy the best model with a single line of code:

```
tuner.deploy(initial_instance_count=1,
             instance_type='ml.c5.xlarge')
```
![](/images/deploy0.png)
After the model is deployed you can test it against the CIFAR10 test dataset.
In a new cell paste the following

```
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

%%time
from keras.preprocessing.image import ImageDataGenerator

def predict(data):
    predictions = predictor.predict(data)['predictions']
    return predictions

predicted = []
actual = []
batches = 0
batch_size = 128

datagen = ImageDataGenerator()
for data in datagen.flow(x_test, y_test, batch_size=batch_size):
    for i, prediction in enumerate(predict(data[0])):
        predicted.append(np.argmax(prediction))
        actual.append(data[1][i][0])

    batches += 1
    if batches >= len(x_test) / batch_size:
        break
```

After it's done predicting on the test dataset, you can check it's accuracy by
running the following code

```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred=predicted, y_true=actual)
display('Average accuracy: {}%'.format(round(accuracy * 100, 2)))
```
![](/images/deploy1.png)

You can plot the confusion matrix to visualize the accuracy of the deployed model:
First let's install the following packages for visualization:
```
!pip install matplotlib
!pip install seaborn
```
Now run the following code to generate the confusion matrix.

```
%matplotlib inline

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred=predicted, y_true=actual)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sn.set(rc={'figure.figsize': (11.7,8.27)})
sn.set(font_scale=1.4)  # for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 10})  # font size
```
![](/images/deploy2.png)

## Congratulations!

You have learned how to run training jobs and deploy models with TensorFlow using Amazon SageMaker. You also learnt how to find the best model hyperparameters using Amazon SageMaker automatic model tuning and deploy models easily. You can now use this guide as a reference to train models on Amazon SageMaker using other deep learning frameworks such as PyTorch and Apache MXNet.
