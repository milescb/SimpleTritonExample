# Example `C++` Triton Backend Setup for NVIDIA and AMD GPUs

A simple environment to get started building a custom backend with the Triton Inference Server. 
For this example, I started from [Triton's official example backend](https://github.com/triton-inference-server/backend/tree/main), 
wrote a simple class to be called in the backend, and finally wrote a simple `python` client to 
interact with the backend. The backend is compatible with ROCM and Alpaka backends for AMD GPUs, 
and Cuda for Nvidia GPUs and it simply adds a client-specified
input tensor to itself and returns this. 

## Getting environment setup

In order to run, you need all the dependencies to build and run. I have built an image with required 
AMD and CUDA deps at `docker.io/milescb/triton-server:25.02-py3_gcc13.3_alpaka2.1.1`.
To run the below commands, pull and run the image with your favorite docker application. For instance:

```
apptainer pull apptainer pull --disable-cache <path_to_sif_files>/tritonserver_alpaka.sif 

apptainer run /eos/home-m/mcochran/images/traccc-aaS/tritonserver_alpaka.sif
```

Note: if running with NVIDIA, remember to enter the image with nvidia support via, 
for instance, `apptainer run --nv <sif_file>`. Conversely, **if compiling for AMD, do _not_ include
the CUDA libraries as this will cause compilation errors**. 

## Build backend

To build the example backend, navigate to `backend/example` and make a `build` and `install` directory. 
Then, `cd` to the `build` directory and run the configure and install commands below:

### CPU (default):

If none of the GPU backends are enabled, the backend builds against the CPU
standalone implementation:

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --target install -- -j20
```

### NVIDIA GPUs:

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_BUILD_TYPE=Release -DTRITON_ENABLE_CUDA=ON

cmake --build . --target install -- -j20
```

### AMD GPUs

To compile with AMD GPUs two options are available: ROCM and Alpaka.
For ROCM compatibility, add the argument `-DTRITON_ENABLE_ROCM=ON` during configuration: 

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ -DCMAKE_BUILD_TYPE=Release \
    -DTRITON_ENABLE_ROCM=ON

cmake --build . --target install -- -j20
```

For Alpaka compatibility, add the argument `-DTRITON_ENABLE_ALPAKA=ON` in addition to the ROCM flags:

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ -DCMAKE_BUILD_TYPE=Release \
    -DTRITON_ENABLE_ROCM=ON -DTRITON_ENABLE_ALPAKA=ON

cmake --build . --target install -- -j20
```

Note: currently, Alpaka is only configured to work with the AMD chip. 

### Building backend image for CPU

```
docker build -f backend/Dockerfile -t <repo_name>:<tag> .
```

## Run the backend

To start the server after building, run the command:

```
tritonserver --model-repository=../../models/
```

In the output, you should see somewhere:

```
+---------+---------+--------+
| Model   | Version | Status |
+---------+---------+--------+
| example | 1       | READY  |
+---------+---------+--------+
```

This backend takes in a 4xn array, and returns the array added to itself. To further configure this,
edit `backend/models/example/config.pbtxt`. In order to modify the backend, I recommend reading and 
attempting to understand the template code in `backend/example/src/recommended.cc`: figure out where 
initialization happens, and where the inference happens. 
The example is already well documented with comments throughout. 

## Running the client

To run the client, open another terminal window _on the same node_ as the backend, after starting 
the backend sever. Navigate to the `client` directory and run `python ExampleTritonClient.py`. 
If you would like to change the inputs or outputs, modify `backend/models/example/config.pbtxt`, 
then modify the `python` code to match and re-run. 

## Testbed internet access

If compiling on the [EF tracking testbed](https://ef-tb-docs.docs.cern.ch), you will need internet 
access. This is enabled by the following proxies:

```
export HTTP_PROXY=http://np04-web-proxy.cern.ch:3128
export HTTPS_PROXY=http://np04-web-proxy.cern.ch:3128
export NO_PROXY=".cern.ch"
export http_proxy=http://np04-web-proxy.cern.ch:3128
export https_proxy=http://np04-web-proxy.cern.ch:3128
export no_proxy=".cern.ch"
```

Note that if you export these, then when running the client, you must run with 
`no_proxy=localhost python ExampleTritonClient.py`.