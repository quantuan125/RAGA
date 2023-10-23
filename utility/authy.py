import os 
import json
import subprocess
import streamlit as st
import boto3
import tempfile

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



class Login:

    #htpasswd_path = os.path.join("user", "server.htpasswd")

    @classmethod
    def create_server_htpasswd(cls, username, password):
        # Assuming you have docker available
        htpasswd_content = s3htpasswd.read_htpasswd()
        command = f'docker run --rm --entrypoint htpasswd httpd:2 -Bbn {username} {password}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        new_entry = result.stdout.strip()

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
        MAPPING_FILE_PATH = "./user/user_port_map.json"
        highest_port = 8000  # start with a default port

        # If the mapping file exists, find the highest port in it
        if os.path.exists(MAPPING_FILE_PATH):
            with open(MAPPING_FILE_PATH, "r") as file:
                content = file.read()
                if content:
                    try:
                        user_port_map = json.loads(content)
                        if user_port_map:
                            highest_port = max(user_port_map.values())
                    except json.JSONDecodeError:
                        # Handle an empty or malformed JSON file
                        pass

        return highest_port + 1

    @classmethod
    def verify_credentials(cls, username, password):
        htpasswd_content = s3htpasswd.read_htpasswd()

        # Use the local htpasswd tool for verification
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            temp.write(htpasswd_content)
            temp_path = temp.name

        command = f'htpasswd -vb {temp_path} {username} {password}'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        os.unlink(temp_path)
        
        # Check if the output contains the success message
        success_message = f"Password for user {username} correct."
        if success_message in result.stdout or success_message in result.stderr:
            return True
        return False

    def save_user_port_mapping(username, port):
        MAPPING_FILE_PATH = "./user/user_port_map.json"
        
        # Load existing mappings
        user_port_map = {}
        if os.path.exists(MAPPING_FILE_PATH):
            with open(MAPPING_FILE_PATH, "r") as file:
                content = file.read()
                if content:
                    try:
                        user_port_map = json.loads(content)
                    except json.JSONDecodeError:
                        # Handle an empty or malformed JSON file
                        pass

        # Update with the new user-port mapping
        user_port_map[username] = port

        # Save updated mappings
        with open(MAPPING_FILE_PATH, "w") as file:
            json.dump(user_port_map, file)


    def get_port_for_user(username):
        MAPPING_FILE_PATH = "./user/user_port_map.json"
        DEFAULT_ADMIN_PORT = 8000
        # Load existing mappings

        if username == "admin":
            return DEFAULT_ADMIN_PORT
        

        if os.path.exists(MAPPING_FILE_PATH):
            with open(MAPPING_FILE_PATH, "r") as file:
                try:
                    user_port_map = json.load(file)
                except json.JSONDecodeError:
                    # Handle an empty or malformed JSON file
                    user_port_map = {}
                return user_port_map.get(username, None)
        return None

    @classmethod
    def delete_user_account(cls, username):
        
        # 1. Remove the user from the htpasswd file stored in S3
        htpasswd_content = s3htpasswd.read_htpasswd()
        lines = htpasswd_content.split("\n")
        updated_lines = [line for line in lines if not line.startswith(username)]
        updated_htpasswd_content = "\n".join(updated_lines)
        s3htpasswd.write_htpasswd(updated_htpasswd_content)

        # 2. Remove the user's mapping from the JSON file
        MAPPING_FILE_PATH = "./user/user_port_map.json"
        user_port_map = {}
        if os.path.exists(MAPPING_FILE_PATH):
            with open(MAPPING_FILE_PATH, "r") as file:
                try:
                    user_port_map = json.load(file)
                except json.JSONDecodeError:
                    pass

            if username in user_port_map:
                port = user_port_map[username]
                del user_port_map[username]

                with open(MAPPING_FILE_PATH, "w") as file:
                    json.dump(user_port_map, file)

                # 3. Stop and remove the Docker container associated with the user
                server_number = port - 8000 + 1
                container_name = f"server-{server_number}"
                command = f'docker stop {container_name} && docker rm {container_name}'
                subprocess.run(command, shell=True)

            else:
                print(f"No mapping found for user {username} in {MAPPING_FILE_PATH}")
        else:
            print(f"{MAPPING_FILE_PATH} not found!")
        
        # 4. Delete all objects from the user's folder in S3
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
            user_port = Login.get_port_for_user(username)
            if user_port:
                st.success(f"Successfully signed in on localhost:{user_port}!")
                st.session_state.authentication = True  # Update the session state
            else:
                st.error("Username not found!")
                st.session_state.authentication = False
        else:
            st.error("Authentication failed. Please check your username and password.")
            st.session_state.authentication = False


    def sign_up_process(signup_username, signup_password):
        if Login.username_exists(signup_username):
            st.error("This username already exists. Please choose a different username.")
            return
        
        Login.create_server_htpasswd(signup_username, signup_password)
        assigned_port = Login.get_next_server_port()
        Login.start_new_server(assigned_port)
        Login.save_user_port_mapping(signup_username, assigned_port)
        st.success(f"Successfully signed up as {signup_username}! Your server URL is localhost:{assigned_port}.")

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
