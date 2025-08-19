export THIRD_PARTY=/home/ubuntu/ThirdParty
export PATH=/usr/local/cuda/bin:$PATH
# export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$THIRD_PARTY/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib:$THIRD_PARTY/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$THIRD_PARTY/FFmpeg-n6.0/install/lib:$THIRD_PARTY/opencv-4.12.0/install_opencv/lib:$THIRD_PARTY/TensorRT-10.12.0.36/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ubuntu/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib:/usr/local/cuda/lib64:$THIRD_PARTY/FFmpeg-n6.0/install/lib:$THIRD_PARTY/opencv-4.5.4/build/install/lib:$THIRD_PARTY/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH

./build/main
