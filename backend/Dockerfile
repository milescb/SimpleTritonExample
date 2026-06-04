FROM docker.io/milescb/triton-server:25.02-py3_gcc13.3_alpaka2.1.1

# Build the example backend (CPU build, no GPU flags).
ENV SRC_DIR="/opt/SimpleTritonExample"
COPY backend ${SRC_DIR}/backend
COPY standalone ${SRC_DIR}/standalone

RUN cd ${SRC_DIR}/backend/example && mkdir build install \
  && cmake -B build -S . \
    -DCMAKE_INSTALL_PREFIX=${SRC_DIR}/backend/example/install \
    -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --target install -- -j20 \
  && rm -rf build

# Model repository for tritonserver --model-repository=$MODELS
ENV MODELS="${SRC_DIR}/backend/models"

RUN chmod -R 775 "${SRC_DIR}/backend/models"