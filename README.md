# Example `C++` Triton Backend Setup

A simple environment to get started on building a custom backend with the Triton Inference Server. For this example, I started from [Triton's official example backend](https://github.com/triton-inference-server/backend/tree/main), and wrote a simple `python` client to interact with it. 

## Getting environment setup

In order to run, you need all the dependencies to build and run. I have built an image with required packages at `docker.io/milescb/triton-server:25.02-py3_gcc13.3`. If you would like to change this image, modify the `Dockerfile` included in this repository, and build your own image. 

## Build backend

To build the example backend, navigate to `backend/example` and make a `build` and `install` directory. Then, `cd` to the `build` directory and run the configure and install commands below

```
cmake -B . -S ../ \
    -DCMAKE_INSTALL_PREFIX=../install/ \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --target install -- -j20
```
To start the sever after building, run the command:

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

This backend takes in a 4x4 array, and returns the same array; however, what data-types and the size of the array may be configured by modifying `backend/models/example/config.pbtxt`. In order to modify the backend, I recommend reading and attempting to understand the template code in `backend/example/src/recommended.cc`: figure out where initialization happens, and where the inference happens. The example is already well documented with comments throughout. 

## Running the client

To run the client, open another terminal window _on the same node_ as the backend, after starting the backend sever. Navigate to the `client` repository and run `python ExampleTritonClient.py`. If you would like to change the inputs or outputs, modify `backend/models/example/config.pbtxt`, then modify the `python` code to match and re-run. 