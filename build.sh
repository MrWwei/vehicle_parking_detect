set -e


export PATH=/usr/local/cuda/bin:$PATH
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ..
make -j4

export THIRD_PARTY=/home/ubuntu/ThirdParty
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive/lib:/usr/local/cuda/lib64:$THIRD_PARTY/FFmpeg-n6.0/install/lib:$THIRD_PARTY/opencv-4.5.4/build/install/lib:$THIRD_PARTY/TensorRT-8.5.1.7/lib:$LD_LIBRARY_PATH

echo "编译完成！"
echo ""

# 询问用户是否要运行程序
read -p "是否要运行程序？(y/n): " run_choice

if [[ "$run_choice" == "y" || "$run_choice" == "Y" ]]; then
    cd ${ROOT_PWD}
    ./build/main
else
    echo "编译完成，程序已就绪。"
fi