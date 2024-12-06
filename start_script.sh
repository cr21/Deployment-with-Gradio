nvidia-smi
# Inside the container
python3 -c "import torch; print(torch.cuda.is_available())"
dvc pull -r s3_store
ls data
apt update
apt install curl -y
apt install unzip -y
apt install vim -y
curl https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip
unzip awscliv2.zip
./aws/install --update
nvidia-smi
echo "Start Training!!!!"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
#python3 src/train.py experiment=bird_classifier_ex  trainer=cpu +trainer.log_every_n_steps=1
python3 src/train.py experiment=food_classifier_vit_small_patch_16_224  trainer=gpu +trainer.log_every_n_steps=5
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
aws s3 ls s3://pytorch-model-gradio/food_101_vit_small/ --recursive
#aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "RUNNING aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/  --recursive"
aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/ --recursive
#echo "RUNNING aws s3 ls pytorch-model-emlov4-predictions/bird_classification --recursive"
#python3 src/infer.py data=birddata inference=bird_infer_aws
python3 src/infer.py  data=food100data inference=food_101_vit_small
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "START INFERNCING"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "RUNNING aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/  --recursive"
aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/ --recursive
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echo "START TRACING"
python3 src/script.py inference=food_101_vit_small ++IMG_W=224 ++IMG_H=224

mkdir -p .gradio/traced_models/food_101_vit_small
echo "cp traced_models/food_101_vit_small/model.pt  .gradio/traced_models/food_101_vit_small/"
cp traced_models/food_101_vit_small/model.pt  .gradio/traced_models/food_101_vit_small/

echo "START DEPLOYING"
python3 -c "import huggingface_hub; huggingface_hub.login(token='$HF_TOKEN')"
cd .gradio
gradio deploy
echo "DEPLOYMENT DONE!!!!"

# s3://pytorch-model-gradio/food_101_vit_tiny/
#aws s3 ls s3://pytorch-model-gradio/food_101_vit_tiny/ --recursive
# python3 src/train.py experiment=food_classifier_vit_tiny_patch_224 trainer=gpu +trainer.log_every_n_steps=10
#python3 src/train.py experiment=food_classifier_vit_small_patch_16_224  trainer=gpu +trainer.log_every_n_steps=10
# aws s3 ls s3://pytorch-model-gradio/food_101_vit_small/ --recursive

# python3 src/infer.py  data=food100data inference=food_101_vit_tiny
# python3 src/infer.py  data=food100data inference=food_101_vit_small

# aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_tiny/ --recursive

# aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/ --recursive

#  python3 src/script.py inference=food_101_vit_small ++IMG_W=224 ++IMG_H=224 

#  python3 src/script.py inference=food_101_vit_small ++IMG_W=224 ++IMG_H=224

# mkdir -p .gradio/traced_models/food_101_vit_small
# cp traced_models/food_101_vit_small/model.pt  .gradio/traced_models/food_101_vit_small/
# cd .gradio
# gradio deploy




# python3 src/train.py experiment=food_classifier_vit_small_patch_16_224  trainer=gpu +trainer.log_every_n_steps=10
# aws s3 ls s3://pytorch-model-gradio/food_101_vit_small/ --recursive

# aws s3 ls s3://pytorch-gradio-predictions/food_101_vit_small/ --recursive


# python3 src/infer.py  data=food100data inference=food_101_vit_small

# python3 src/script.py inference=food_101_vit_small ++IMG_W=224 ++IMG_H=224
# cp traced_models/food_101_vit_small/model.pt  .gradio/traced_models/food_101_vit_small/