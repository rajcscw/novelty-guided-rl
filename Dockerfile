# base image
FROM ubuntu18.04-cuda10.1

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install basic stuff
RUN apt-get update
RUN apt-get install -y libzmq3-dev python3-pip
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip3 install --upgrade pip

# install the required modules
RUN pip3 install -e .