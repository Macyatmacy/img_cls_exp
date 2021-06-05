# !{sys.executable} -m pip install --upgrade stepfunctions

import boto3
import sagemaker 
from sagemaker.tensorflow import TensorFlow
# from sagemaker.amazon.amazon_estimator import image_uris
# from sagemaker.inputs import TrainingInput
# from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader
import random
import shutil
import tensorflow as tf
import numpy as np

session = sagemaker.Session() 
region = boto3.Session().region_name
bucket = session.default_bucket() 

project_name = "pn_deploy"

train_prefix = "train"
val_prefix = "validation"

train_data = "s3://{}/{}/{}/".format(bucket, project_name, train_prefix)
validation_data = "s3://{}/{}/{}/".format(bucket, project_name, val_prefix)

def get_file_list(bucket_name, prefix):
    s3 = boto3.resource('s3')
    bucket=bucket_name
    my_bucket = s3.Bucket(bucket)
    location_list = []
    for (bucket_name, key) in map(lambda x: (x.bucket_name, x.key), my_bucket.objects.filter(Prefix=prefix)):
        data_location = "s3://{}/{}".format(bucket_name, key)
        location_list.append(data_location)
    # Remove the root folder path
    if "s3://{}/{}/".format(bucket_name, prefix) in location_list:
        location_list.remove("s3://{}/{}/".format(bucket_name, prefix))
    return location_list

list_normal = get_file_list(bucket,"pn_deploy/normal_1000")
list_pneumonia = get_file_list(bucket,"pn_deploy/pneumonia_1000")

#download data
for l in list_normal:
    data_source1 = S3Downloader.download(
    local_path="/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/normal/",
    s3_uri=l,
    sagemaker_session=session,
    )

for l in list_pneumonia:
    data_source1 = S3Downloader.download(
    local_path="/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/pneumonia/",
    s3_uri=l,
    sagemaker_session=session,
    )
    
# generate annotations
import os
if not os.path.isdir('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations'):
    os.mkdir('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations')

filePath = '/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/normal'
l = os.listdir(filePath)
ant={}
for n in l:
    name,_ = n.split('.')
    ant[name] = 'normal'
with open('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations/normal.txt', 'w') as f:
    for n, c in ant.items():
        f.write(str(n)+" "+str(c)+"\n")

filePath = '/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/pneumonia'
l = os.listdir(filePath)
ant={}
for n in l:
    name,_ = n.split('.')
    ant[name] = 'pneumonia'
with open('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations/pneumonia.txt', 'w') as f:
    for n, c in ant.items():
        f.write(str(n)+" "+str(c)+"\n")
        
# read annotations
def get_annotations(file_path, annotations={}):
    
    with open(file_path, 'r') as f:
        rows = f.read().splitlines()

    for _, row in enumerate(rows):
        image_name, class_name = row.split(' ')
        image_name = image_name + '.jpeg'
        
        annotations[image_name] = class_name
    
    return annotations

# read annotations
annotations_normal = get_annotations('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations/normal.txt')
annotations_pneumonia = get_annotations('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/annotations/pneumonia.txt')

total_count = len(annotations_normal.keys())
print('Total normal examples', total_count)
total_count = len(annotations_pneumonia.keys())
print('Total pneumonia examples', total_count)

print(next(iter(annotations_normal.items())))
print(next(iter(annotations_pneumonia.items())))

# split and copy file
classes = ['normal', 'pneumonia']
sets = ['train', 'validation']
root_dir = '/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/custom_data'

if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
    
for set_name in sets:
    if not os.path.isdir(os.path.join(root_dir, set_name)):
        os.mkdir(os.path.join(root_dir, set_name))
    for class_name in classes:
        folder = os.path.join(root_dir, set_name, class_name)
        if not os.path.isdir(folder):
            os.mkdir(folder)
            
for image, class_name in annotations_normal.items():
    target_set = 'validation' if random.randint(0, 99) < 20 else 'train'
    target_path = os.path.join(root_dir, target_set, class_name, image)
    shutil.copy(os.path.join('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/normal', image), target_path)

for image, class_name in annotations_pneumonia.items():
    target_set = 'validation' if random.randint(0, 99) < 20 else 'train'
    target_path = os.path.join(root_dir, target_set, class_name, image)
    shutil.copy(os.path.join('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/data/pneumonia', image), target_path)
    
sets_counts = {
    'train': 0,
    'validation': 0
}

for set_name in sets:
    for class_name in classes:
        path = os.path.join(root_dir, set_name, class_name)
        count = len(os.listdir(path))
        print(path, 'has', count, 'images')
        sets_counts[set_name] += count

print(sets_counts)

print('Uploading to S3..')
s3_data_path = session.upload_data(path=root_dir, bucket=bucket, key_prefix='pn_deploy')

print('Uploaded to', s3_data_path)

role = sagemaker.get_execution_role()
estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    instance_count=2,
    instance_type='ml.m5.4xlarge', # you can use any instance_type within your quotas
    framework_version='2.1.0',
    py_version='py3',
    output_path="s3://{}/{}".format(bucket, project_name),
)

s3_data_path = 's3://sagemaker-us-east-2-179199196742/pn_deploy'
estimator.fit(s3_data_path)

predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
print('\nModel Deployed!')

def get_pred(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    results = predictor.predict(img)
    class_id = int(np.squeeze(results['predictions']) > 0.5)
    return classes[class_id]

# task link and list
list_task = get_file_list(bucket, "pn_deploy/task/data")

for l in list_task:
    data_source = S3Downloader.download(
    local_path='/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/task/data/',
    s3_uri=l,
    )
    
image_path = []
for l in list_task:
    image_path.append('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/task/' + l[53:])
image_path

with open('/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/task/prediction_output.txt', 'w') as f:
    for i in image_path:
        f.write(i[70:] + " " + get_pred(i)+"\n")
        
print('Uploading to S3..')
s3_data_path = session.upload_data(path='/home/ec2-user/SageMaker/img_cls_exp/MedicalImage/Pneumonia/task/prediction_output.txt', bucket=bucket, key_prefix='pn_deploy/task/pred_output')
print('Uploaded to', s3_data_path)

