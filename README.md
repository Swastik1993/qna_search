# COVID-19 Q&A deployment source code


[![docker build status](https://img.shields.io/badge/docker_build-passing-emerald.svg)](#) [![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://13.92.197.252/) 

___

## Languages used
![Python 3.8](https://img.shields.io/badge/python-v3.8-blue.svg) (app running on FastAPI backend)


#### This version has been modified slightly from the original version so that it can run without using GPU acceleration.
* __If you would like to view the actual code with GPU acceleration that uses both Pytorch and Tensorflow please visit the master branch of the [bitbucket repo](https://bitbucket.org/bridgei2idev/covid_qna/src/master/). This code is similar to the no_gpu branch of the [repo](https://bitbucket.org/bridgei2idev/covid_qna/src/no_gpu/).__
* Please make sure to check the all files and comments and make necessary changes if required before building with docker.
* This is the python code for deployment purpose only. Once the indexes are built, the lucene index folder should be updated. *This code will NOT build the lucene indexes*.

___

## Steps for building and executing
* __Before starting make sure you have a high speed internet connection and enough data because the code will download almost 10GB of data altogether while it is being built. Also make sure you have enough disk space because once the application/images are built the total size of both the application/images running in the system would be almost 20GB. (In case of docker the intermediate images are also built in this process that may be of even larger sizes, so make sure to clean the system at the end once the images are built.)__
* In case you are building with docker (recommended approach) please follow the below instructions else you can skip this section.

___

### Docker build guide
___Before going ahead with the docker build please remember that you will have to edit the module.py file once before the docker build instruction. Simply open the file and uncomment line 71 and comment out line 72 and save the file. That's all!___
##### For Windows systems
* If you are using windows chances are you might have to get in touch with the IT team to get Docker installed. This README is written considering you have admin rights or you are installing on a Windows server.
* Please visit this official [docker url](https://docs.docker.com/get-docker/) and then click on *'Docker Desktop for Windows'* followed by *'Download from Docker Hub'*. Finally click on *'Get Stable'* to download Docker Desktop Community Edition (However if you are installing on a windows server using Enterprise edition is recommended). After that simply install docker by follow the on screen instructions.
* Then unzip the archive and extract the files.
* Open terminal and change your path to the covid_qna directory. Once inside the directory you will need to download the lucene indexed data.
```
cd covid_qna/
```
* In order to download the lucene indexed data for covid-19 articles navigate to the *data* folder and execute the python script.
```
cd data/
pip install tqdm
python data_download.py
```
On running these commands the lucene index data in a .tar.gz file (4.5GB approx) will be downloaded.
* Navigate back to the root directory and run the following commands in sequence in order to build and run the docker images. *(Please check if you have edited the module.py file)*
```
cd ..
docker-compose build
docker-compose up -d
```
* The first command will build the docker images. It will take sometime (about 20-30mins depending on your internet speed and system processing speed.)*
* Once everything is done the application should start. In order to access the UI for the python APIs then please visit ```http://localhost:5000/docs/```.


##### For Linux systems
Open your terminal and type the following commanda to get docker and docker-compose installed.
```
# For installing Docker
sudo apt install docker.io
docker --version

# If you are unable to see the docker version then use the below command to add the current user to docker group and logout and log back in and then re-run the command to check the version.
sudo usermod -aG docker $USER

# For installing docker-compose
sudo apt install docker-compose
docker-compose version
```
* Then unzip the archive and extract the files.
* Open terminal and change your path to the covid_qna directory. Once inside the directory you will need to download the lucene indexed data.
```
cd covid_qna/
```
* In order to download the lucene indexed data for covid-19 articles navigate to the *data* folder and execute the python script.
```
cd data/
pip install tqdm
python data_download.py
```
On running these commands the lucene index data in a .tar.gz file (4.5GB approx) will be downloaded.
* Navigate back to the root directory and run the following commands in sequence in order to build and run the docker images. *(Please check if you have edited the module.py file)*
```
cd ..
docker-compose build
docker-compose up -d
```
* The first command will build the docker images. It will take sometime (about 20-30mins depending on your internet speed and system processing speed.)*
* Once everything is done the application should start. In order to access the UI for the python APIs then please visit ```http://localhost:5000/docs/```.

##### Removing any temporary images that were created while building
Once the docker images are built you can choose to run the prune command to remove any intermediate/unused images that were created in the process of building the images. In order to do that please use the following command.
```
docker system prune
```
On running this command it will prompt to confirm by [y/N], press y to clean the intermediate/unused images
___

### Without using docker
___As of now the application does not work on windows because the Java virtual environment does not gets initialized. Hence the only approach is using Linux systems.___
##### For Linux systems
Open your terminal and type the following commanda to get the latest package updates and install Java, Python virtual environment and a few other necessary packages.
```
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk
sudo apt-get install -y curl build-essential gcc python3 python3-venv
```
* Now create a new virtual environment and activate it. Then unzip the archive and extract the files.
```
python3 -m venv ml_env
source ml_env/bin/activate
unzip covid_qna.zip
```
* Now change your path to the covid_qna directory. Once inside the directory you will need to install the required python packages and then download the lucene indexed data.
```
cd covid_qna/
pip install -r requirements.txt
```
* In order to download the lucene indexed data for covid-19 articles navigate to the *data* folder and execute the python script.
```
cd data/
python data_download.py
```
On running these commands the lucene index data in a .tar.gz file (4.5GB approx) will be downloaded.
* Then you need to untar the compressed file for the python app.
```
tar -xvzf lucene-index-cord19.tar.gz
```
* Navigate back to the root directory and change to the src directory. After that run the commands to download all the models and run the application.
```
cd ../src/
python init_models.py
uvicorn --host 0.0.0.0 app:app
```
* The first command will download all the required models and prep them for the application.
* Once everything is done the application should start. In order to access the UI for the python APIs then please visit ```http://localhost:8000/docs/```.

___

*In case of any issues please reach out to [me](mailto:swastik.biswas@bridgei2i.com).*
