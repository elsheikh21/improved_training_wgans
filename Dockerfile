FROM tensorflow/tensorflow:1.14.0-gpu-py3

# update ubuntu packages
RUN apt update && apt upgrade -y

# install packages
RUN apt install -y --no-install-recommends nano graphviz wget unzip ca-certificates

# Set the working directory
WORKDIR /home/

# Download dataset
RUN wget -q --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_-nOsg952fVeNpWWPfUMEgDi6xm1s3BW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_-nOsg952fVeNpWWPfUMEgDi6xm1s3BW" -O img_align_celeba.zip && rm -rf /tmp/cookies.txt
RUN mkdir data && mkdir data/train
RUN unzip -q img_align_celeba.zip -d data/train

# Copy your app's source code from your host to your image filesystem
COPY . .

# Install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt