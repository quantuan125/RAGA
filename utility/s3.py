import streamlit as st
import boto3
import os


class s3:
    def delete_s3_object():

        s3 = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

    bucket_name = os.getenv("AWS_BUCKET_NAME")

