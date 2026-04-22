FROM nvcr.io/nvidia/tritonserver:25.02-py3

LABEL description="Triton Server backend with other dependencies for traccc-as-a-Service (ROCm/AMD)"
LABEL version="2.0"

# Install dependencies
RUN apt-get update -y && apt-get install -y \
    build-essential curl git freeglut3-dev libfreetype6-dev libpcre3-dev\
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
    libtbb-dev ninja-build time tree \
    python3 python3-dev python3-pip python3-numpy \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig \
    rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev ccache libb64-dev \
    libsuitesparse-dev libhwloc-dev libsuperlu-dev tmux \
  && apt-get clean -y

RUN apt-get update -y && apt-get install -y git-lfs \
  && git lfs install \
  && apt-get clean -y

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install -U pandas matplotlib seaborn tritonclient[all] mplhep \
    ipykernel numpy

# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:/opt/rocm/lib:/opt/rocm/hip/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV PYTHONNOUSERSITE=True
# ROCm/AMD GPU support
ENV ROCM_PATH="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV PATH="$PATH:/opt/rocm/bin:/opt/rocm/llvm/bin"

# Install GCC 13.3.0
RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update -y \
    && apt-get install -y gcc-13 g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-13 \
    && apt-get clean -y \
    && gcc --version

# Manual builds for specific packages
# Install CMake v3.29.4
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v3.29.4/cmake-3.29.4-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src

# Install xxHash v0.7.3
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Cyan4973/xxHash/archive/v0.8.2.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src/cmake_unofficial -GNinja\
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  && cmake --build build -- install -j20\
  && cd /tmp && rm -rf src build

# Install full ROCm development stack (HIP compiler, headers, runtime)
RUN apt-get update -y && apt-get install -y wget gnupg2 ca-certificates \
  && mkdir -p --mode=0755 /etc/apt/keyrings \
  && wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
    | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null \
  && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 jammy main" \
    | tee /etc/apt/sources.list.d/rocm.list \
  && echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | tee /etc/apt/preferences.d/rocm-pin-600 \
  && apt-get update -y \
  && apt-get install -y rocm-hip-sdk rocminfo \
  && apt-get clean -y