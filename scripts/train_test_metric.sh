
#TRIAL=${1}
#NET=${2}
#mkdir checkpoints
#mkdir checkpoints/alexnet_2
#python ./train.py --use_gpu --net alex --name alexnet_adv_linf_p0
python ./test_dataset_model.py --use_gpu --net alex --model_path ./checkpoints/alexnet_adv_linf_p0/latest_net_.pth
