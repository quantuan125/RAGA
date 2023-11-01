import os 
import json
import subprocess
import streamlit as st
from passlib.apache import HtpasswdFile
from passlib.hash import bcrypt




class Login:

    htpasswd_path = os.path.join("user", "server.htpasswd")
    MAPPING_FILE_PATH = "./user/user_port_map.json"

    @classmethod
    def create_server_htpasswd(cls, username, password):
        hashed_password = bcrypt.using(rounds=12).hash(password)
        new_entry = f"{username}:{hashed_password}"
        with open(cls.htpasswd_path, 'a') as file:
            file.write(f"{new_entry}\n")

    @staticmethod
    def start_new_server(port):
        server_number = port - 8000  # This will start from 1 for port 8001, 2 for port 8002, and so on
        if server_number <= 0:  # Handle the case for admin or any other special cases
            server_number = 1  # or set it to whatever number you prefer for admin
        
        container_name = f"server-{server_number}"

        # The command to run the Docker container with a specified name
        command = f'docker run -d --name {container_name} -p {port}:8000 -e "ALLOW_RESET=TRUE" server:latest'
        subprocess.run(command, shell=True)

    staticmethod
    def get_next_server_port():
        highest_port = 8000  # start with a default port

        if os.path.exists(Login.MAPPING_FILE_PATH):
                with open(Login.MAPPING_FILE_PATH, "r") as file:
                    user_port_map = json.load(file)
                    highest_port = max(user_port_map.values())

        return highest_port + 1

    @classmethod
    def verify_credentials(cls, username, password): #DONE
        # Load the htpasswd file content into an HtpasswdFile instance
        htpasswd_file = HtpasswdFile(cls.htpasswd_path)
        
        # Use the verify method to check the credentials
        return htpasswd_file.verify(username, password)
    
    @staticmethod
    def save_user_port_mapping(username, port):
        if os.path.exists(Login.MAPPING_FILE_PATH):
            with open(Login.MAPPING_FILE_PATH, "r") as file:
                user_port_map = json.load(file)
        else:
            user_port_map = {}

        user_port_map[username] = port
        with open(Login.MAPPING_FILE_PATH, "w") as file:
            json.dump(user_port_map, file)

    @staticmethod
    def get_port_for_user(username):
        DEFAULT_ADMIN_PORT = 8000

        if username == "admin":
            return DEFAULT_ADMIN_PORT

        if os.path.exists(Login.MAPPING_FILE_PATH):
            with open(Login.MAPPING_FILE_PATH, "r") as file:
                user_port_map = json.load(file)
                return user_port_map.get(username, None)
        return None

    @classmethod #DONE
    def delete_user_account(cls, username):
        # 1. Remove the user from the htpasswd file
        with open(cls.htpasswd_path, 'r') as file:
            lines = file.readlines()
        lines = [line for line in lines if not line.startswith(username)]
        with open(cls.htpasswd_path, 'w') as file:
            file.writelines(lines)

        # 2. Remove the user's mapping from the JSON file
        if os.path.exists(cls.MAPPING_FILE_PATH):
            with open(cls.MAPPING_FILE_PATH, "r") as file:
                user_port_map = json.load(file)
            if username in user_port_map:
                port = user_port_map[username]
                del user_port_map[username]
                with open(cls.MAPPING_FILE_PATH, "w") as file:
                    json.dump(user_port_map, file)

                # 3. Stop and remove the Docker container associated with the user
                server_number = port - 8000 
                container_name = f"server-{server_number}"
                command = f'docker stop {container_name} && docker rm {container_name}'
                subprocess.run(command, shell=True)
            else:
                print(f"No mapping found for user {username} in {cls.MAPPING_FILE_PATH}")
        else:
            print(f"{cls.MAPPING_FILE_PATH} not found!")

    @staticmethod
    def sign_in_process(username, password):
        if Login.verify_credentials(username, password):
            st.session_state.username = username
            user_port = Login.get_port_for_user(username)
            if user_port:
                st.success(f"Successfully signed in on localhost:{user_port}!")
                st.session_state.authentication = True
            else:
                st.error("Username not found!")
                st.session_state.authentication = False
        else:
            st.error("Authentication failed. Please check your username and password.")
            st.session_state.authentication = False

    @staticmethod
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
        if os.path.exists(cls.htpasswd_path):
            with open(cls.htpasswd_path, 'r') as file:
                lines = file.readlines()
            existing_usernames = [line.split(":")[0] for line in lines]
            return username in existing_usernames
        return False
