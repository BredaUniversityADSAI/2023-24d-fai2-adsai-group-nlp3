FROM mcr.microsoft.com/azureml/curated/tensorflow-2.12-cuda11:28

RUN apt-get update

RUN apt-get install -y ffmpeg
RUN find / -type d -name bin -print | grep ffmpeg | while read BIN_DIR; do echo "export PATH=\$PATH:$BIN_DIR" >> /etc/profile; done

COPY requirements.txt .

RUN pip install -r requirements.txt
