import boto3
import os
import json
class s3htpasswd:
    # Initialize S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )
    
    # Bucket name and object key
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    object_key = 'server.htpasswd'
    
    @classmethod
    def read_htpasswd(cls):
        response = cls.s3.get_object(Bucket=cls.bucket_name, Key=cls.object_key)
        htpasswd_content = response['Body'].read().decode('utf-8')
        return htpasswd_content
    
    @classmethod
    def write_htpasswd(cls, content):
        cls.s3.put_object(Body=content, Bucket=cls.bucket_name, Key=cls.object_key)

class s3userportmap:

    # Initialize the S3 client
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
    FILE_KEY = "user_port_map.json"

    @classmethod
    def read_user_port_map(cls):
        try:
            response = cls.s3.get_object(Bucket=cls.BUCKET_NAME, Key=cls.FILE_KEY)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except Exception as e:
            print(f"Error reading {cls.FILE_KEY} from S3: {e}")
            return {}

    @classmethod
    def write_user_port_map(cls, data):
        try:
            cls.s3.put_object(Body=json.dumps(data, indent=4), Bucket=cls.BUCKET_NAME, Key=cls.FILE_KEY)
        except Exception as e:
            print(f"Error writing {cls.FILE_KEY} to S3: {e}")

class S3:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')

    def upload_to_s3(self, file_obj, s3_file_name):
        try:
            file_obj.seek(0)
            #st.write("File object position:", file_obj.tell())
            self.client.upload_fileobj(file_obj,  self.bucket_name, s3_file_name, 
                              ExtraArgs={'ContentType': 'application/pdf', 'ACL': 'public-read'})
            print("Upload Successful")
            return True
        except Exception as e:  
            print(f"An error occurred: {e}")
            return False

    def delete_document_objects(self, s3_urls):
        for s3_url in s3_urls:
            s3_key = s3_url.split(f"{self.bucket_name}/")[1]
            self.client.delete_object(Bucket=self.bucket_name, Key=s3_key)

    def delete_objects_in_collection(self, username, actual_collection_name):
        collection_prefix = f"{username}/{actual_collection_name}/"
        
        # List all objects under the collection prefix
        result = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=collection_prefix)
        
        # If the collection exists in S3
        if result.get('KeyCount') > 0:
            # Prepare the list of objects to delete
            objects_to_delete = [{'Key': obj['Key']} for obj in result['Contents']]
            
            # Delete the objects
            self.client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects_to_delete})

    def rename_objects_in_collection(self, username, old_collection_name, new_collection_name):
        old_prefix = f"{username}/{old_collection_name}/"
        new_prefix = f"{username}/{new_collection_name}/"
        
        # List all objects under the old collection prefix
        result = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=old_prefix)
        
        # If the collection exists in S3
        if result.get('KeyCount') > 0:
            for obj in result['Contents']:
                old_key = obj['Key']
                new_key = old_key.replace(old_prefix, new_prefix)
                
                # Copy the object to the new key
                copy_source = {'Bucket': self.bucket_name, 'Key': old_key}
                self.client.copy_object(CopySource=copy_source, Bucket=self.bucket_name, Key=new_key, ACL='public-read')
                
                # Delete the old object
                self.client.delete_object(Bucket=self.bucket_name, Key=old_key)


