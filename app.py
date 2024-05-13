from flask import Flask,request,make_response,jsonify
import boto3
from werkzeug.utils import secure_filename
import uuid
#from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME
from filehandler import upload_file_to_s3,get_presigned_file_url,delete_folder
from model.predict import classify_image_urls
from model_unet.segmenter import segment
import os
from dotenv import load_dotenv
from kmeans import score
load_dotenv()
# Create an instance of the Flask class
app = Flask(__name__)

AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME=os.getenv('S3_BUCKET_NAME')





s3=boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,

)
# Define a route and a view function
@app.route('/',methods=["POST"])
def get_leaf_type():
    

    
    leaf_type = request.json['leaf_type']
    print(type(leaf_type))
    # Set cookie with leaf_type value
    resp = make_response(jsonify({"leaf_type":leaf_type}))

    # Construct response data
    resp.set_cookie('leaf_type', leaf_type)
    return resp

@app.route('/severity',methods=["POST"])
def get_severity():
    unique_folder=str(uuid.uuid4())
    uploaded_files=[]
    for key, file in request.files.items():
        
        if file.filename != '':
            
            file_name=secure_filename(file.filename)
            
            #s3.upload_fileobj(file,S3_BUCKET_NAME,s3_key,ExtraArgs={'ContentType': file.content_type})
            upload_file_to_s3(file,file_name,unique_folder)
            
            uploaded_files.append(get_presigned_file_url(unique_folder, file_name))
    
    # Process uploaded images (for demonstration, just returning the count)
    num_images = len(uploaded_files)
    disease_level=score(uploaded_files)

    return disease_level




# Define another route and view function
@app.route('/<type_leaf>',methods=["POST"])
def classify_disease(type_leaf):
    unique_folder=str(uuid.uuid4())
    uploaded_files=[]
    for key, file in request.files.items():
        
        if file.filename != '':
            
            file_name=secure_filename(file.filename)
            
            #s3.upload_fileobj(file,S3_BUCKET_NAME,s3_key,ExtraArgs={'ContentType': file.content_type})
            upload_file_to_s3(file,file_name,unique_folder)
            
            uploaded_files.append(get_presigned_file_url(unique_folder, file_name))
    
    # Process uploaded images (for demonstration, just returning the count)
    num_images = len(uploaded_files)
    
    leaves,diseases=classify_image_urls(uploaded_files)
    leaves=set(leaves)
    leaves=list(leaves)
    severity=segment(uploaded_files)
    delete_folder(S3_BUCKET_NAME,unique_folder)
    #print(predictions)
    
    return jsonify({"leaves":leaves,"diseases": diseases,"area affected unet":severity[1],"severity level":severity[0]})

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
