set -e
TARGET_SOC="rk356x"
GCC_COMPILER=aarch64-linux-gnu
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$(cd "$(dirname $0)" && pwd)
BUILD_DIR=${ROOT_PWD}/build

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake .. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4
make install
