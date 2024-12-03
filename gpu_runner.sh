# running shell script
conda init
conda activate pytorch
# install awscli
curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
sudo ./aws/install --update
sudo wget https://github.com/bcicen/ctop/releases/download/v0.7.7/ctop-0.7.7-linux-amd64 -O /usr/local/bin/ctop
sudo chmod +x /usr/local/bin/ctop

# install nvidia docker for gpu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update

aws ecr get-login-password --region us-east-1  --profile emlo | docker login --username AWS --password-stdin 575108919357.dkr.ecr.us-east-1.amazonaws.com
docker pull 575108919357.dkr.ecr.us-east-1.amazonaws.com/cr/emlo-docker-plt:latest

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

docker run --gpus all \
    --env AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} --env AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}  \
    -it --entrypoint /bin/bash \
    575108919357.dkr.ecr.us-east-1.amazonaws.com/cr/emlo-docker-plt:latest \
    -c "sh start_script.sh"

# docker run --gpus all  -it  -w /app  --entrypoint   /bin/bash 575108919357.dkr.ecr.us-east-1.amazonaws.com/cr/emlo-docker-plt:latest -c "sh start_script.sh"

