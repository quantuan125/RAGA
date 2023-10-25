import os 
import json
import subprocess
import streamlit as st
import boto3
from passlib.apache import HtpasswdFile
from passlib.hash import bcrypt
from utility.s3 import s3htpasswd,s3userportmap



class Login:

    #htpasswd_path = os.path.join("user", "server.htpasswd")

    @classmethod
    def create_server_htpasswd(cls, username, password):
        # Assuming you have docker available
        htpasswd_content = s3htpasswd.read_htpasswd()
        hashed_password = bcrypt.using(rounds=12).hash(password)
    
        # Format the new entry
        new_entry = f"{username}:{hashed_password}"

        # Only add a newline if the existing content is not empty
        separator = "\n" if htpasswd_content else ""
        updated_htpasswd_content = f"{htpasswd_content}{separator}{new_entry}"

        s3htpasswd.write_htpasswd(updated_htpasswd_content)

    def start_new_server(port):
        server_number = port - 8000  # This will start from 1 for port 8001, 2 for port 8002, and so on
        if server_number <= 0:  # Handle the case for admin or any other special cases
            server_number = 1  # or set it to whatever number you prefer for admin
        
        container_name = f"server-{server_number}"

        # The command to run the Docker container with a specified name
        command = f'docker run -d --name {container_name} -p {port}:8000 -e "ALLOW_RESET=TRUE" server:latest'
        subprocess.run(command, shell=True)


    def get_next_server_port():
        highest_port = 8000  # start with a default port

    # Fetch the user-port map from S3
        user_port_map = s3userportmap.read_user_port_map()
        if user_port_map:
            highest_port = max(user_port_map.values())

            return highest_port + 1

    @classmethod
    def verify_credentials(cls, username, password):
        htpasswd_content = s3htpasswd.read_htpasswd()

        # Create an HtpasswdFile instance from the content
        htpasswd_file = HtpasswdFile()
        htpasswd_file.load_string(htpasswd_content)
        
        # Use the verify method to check the credentials
        return htpasswd_file.verify(username, password)

    def save_user_port_mapping(username, port):
        user_port_map = s3userportmap.read_user_port_map()

        # Update with the new user-port mapping
        user_port_map[username] = port

        # Save updated mappings to S3
        s3userportmap.write_user_port_map(user_port_map)


    def get_port_for_user(username):
        DEFAULT_ADMIN_PORT = 8000
        # Load existing mappings

        if username == "admin":
            return DEFAULT_ADMIN_PORT
        

        user_port_map = s3userportmap.read_user_port_map()
    
        # Check if the mapping is a valid dictionary
        if not isinstance(user_port_map, dict):
            user_port_map = {}
        
        return user_port_map.get(username, None)

    @classmethod
    def delete_user_account(cls, username):
        
        # 1. Remove the user from the htpasswd file stored in S3
        htpasswd_content = s3htpasswd.read_htpasswd()
        lines = htpasswd_content.split("\n")
        updated_lines = [line for line in lines if not line.startswith(username)]
        updated_htpasswd_content = "\n".join(updated_lines)
        s3htpasswd.write_htpasswd(updated_htpasswd_content)
        
        # 2. Delete all objects from the user's folder in S3
        s3 = boto3.client('s3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                        region_name=os.getenv('AWS_DEFAULT_REGION'))

        bucket_name = os.getenv('AWS_BUCKET_NAME')
        folder_path = username  # Assuming the folder name in S3 is the username

        # List all objects under the folder
        result = s3.list_objects(Bucket=bucket_name, Prefix=folder_path)

        # Check if the folder exists
        if 'Contents' in result:
            # Prepare the list of objects to delete
            objects_to_delete = [{'Key': obj['Key']} for obj in result['Contents']]

            # Delete the objects
            s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects_to_delete})

            print(f"Deleted folder {folder_path} and all its contents from S3.")
        else:
            print(f"Folder {folder_path} does not exist in S3. Nothing to delete.")

    def sign_in_process(username, password):
        if Login.verify_credentials(username, password):
            st.session_state.username = username
            st.success(f"Successfully signed in as {username}!")
            st.session_state.authentication = True  # Update the session state
        else:
            st.error("Authentication failed. Please check your username and password.")
            st.session_state.authentication = False

    @staticmethod
    def sign_up_process(signup_username, signup_password):
        if Login.username_exists(signup_username):
            st.error("This username already exists. Please choose a different username.")
            return

        Login.create_server_htpasswd(signup_username, signup_password)
        st.success(f"Successfully signed up as {signup_username}!")

    @classmethod
    def username_exists(cls, username):
        try:
            htpasswd_content = s3htpasswd.read_htpasswd()
            if not htpasswd_content:
                return False
            lines = htpasswd_content.split("\n")
            existing_usernames = [line.split(":")[0] for line in lines]
            return username in existing_usernames
        except Exception as e:  # Replace this with the specific exception for a missing S3 object, if applicable
            return False
