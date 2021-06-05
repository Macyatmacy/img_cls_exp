import sagemaker 
import boto3
from sagemaker.s3 import S3Uploader
from sagemaker.s3 import S3Downloader
import tensorflow as tf
import numpy as np

session = sagemaker.Session() 
bucket = session.default_bucket() 


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

def get_pred(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    results = predictor.predict(img)
    class_id = int(np.squeeze(results['predictions']) > 0.5)
    return classes[class_id]

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