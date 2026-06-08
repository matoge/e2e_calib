Preparation

Go to the internal ClearML page:
http://172.16.200.185:8082/settings/workspace-configuration

Create credentials by clicking Create new credential.

Register the credentials on your local machine:

$ clearml-init

When prompted, enter the following configuration:

api { api_server: http://172.16.200.185:8083 web_server: http://172.16.200.185:8082 files_server: http://172.16.200.185:8084 credentials { "access_key": "YOUR ACCESS KEY", "secret_key": "YOUR SECRET KEY" } }

Run the script

$ ./hoge.py

The model will be automatically downloaded from ClearML and used to predict the corrected positions of the points in the image.

Data

The required data is located under:

sample_script_with_clearml/data

If the data is not present, download it using Git LFS:

$ git lfs pull