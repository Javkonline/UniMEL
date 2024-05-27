echo "Device is ${1}. Running on dataset ${2}";
CUDA_VISIBLE_DEVICES=${1} python -u ./code/main.py --config "./config/${2}.yaml"