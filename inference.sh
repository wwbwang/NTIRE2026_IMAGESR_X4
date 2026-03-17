# for validation
GPUS="0"
CUDA_VISIBLE_DEVICES=$GPUS python test.py --valid_dir "your_validation_directory"

# for test
GPUS="1"
CUDA_VISIBLE_DEVICES=$GPUS python test.py --test_dir "your_test_directory"
