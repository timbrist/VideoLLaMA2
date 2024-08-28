
# Download Large file without VideoLLaMA2-7B-Base
export DATA_FOLDER="/home/yans2/videollama2_cache"

git clone https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B-Base ${DATA_FOLDER}
wget https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2-7B-Base/resolve/main/mm_projector.bin?download=true \
    -O ${DATA_FOLDER}/VideoLLaMA2-7B-Base/mm_projector.bin 



