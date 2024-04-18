import uuid
import boto3
#from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME
import os
from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME=os.getenv('S3_BUCKET_NAME')
AWS_REGION=os.getenv('AWS_REGION')


s3 = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY, 
        aws_secret_access_key=AWS_SECRET_KEY, 
        region_name=AWS_REGION
    )

ALLOWED_FILE_TYPES = {'png', 'jpg', 'jpeg'}
S3_BUCKET_NAME = S3_BUCKET_NAME
S3_EXPIRES_IN_SECONDS = 1000

def get_file_type(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower()

def is_file_type_allowed(filename):
    return get_file_type(filename) in ALLOWED_FILE_TYPES

def upload_file_to_s3(file, provided_file_name,folder):
    stored_file_name = f'{folder}/{provided_file_name}'
    s3.upload_fileobj(file, S3_BUCKET_NAME, stored_file_name)
    return stored_file_name

def get_presigned_file_url(unique_folder, file_name):
    
    stored_file_name = f'{unique_folder}/{file_name}'
    return s3.generate_presigned_url('get_object',
                                                    Params={'Bucket': S3_BUCKET_NAME,
                                                            'Key': stored_file_name},
                                                    ExpiresIn=S3_EXPIRES_IN_SECONDS)

def delete_folder(bucket_name, folder_prefix):
    # List objects in the specified folder
    objects_to_delete = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    for obj in response.get('Contents', []):
        objects_to_delete.append({'Key': obj['Key']})
    
    # Delete objects in the folder
    if objects_to_delete:
        s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects_to_delete})