export DATA_FOLDER="/home/yans2/videollama2_cache"

git clone https://huggingface.co/openai/clip-vit-large-patch14-336

wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/pytorch_model.bin?download=true \
    -O ${DATA_FOLDER}/clip-vit-large-patch14-336/pytorch_model.bin \

wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/tf_model.h5?download=true \
    -O ${DATA_FOLDER}/clip-vit-large-patch14-336/tf_model.h5 \




