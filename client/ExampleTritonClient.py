import sys
import argparse
import numpy as np

import tritonclient.grpc as grpcclient

def main():
    
    try:
        concurrent_request_count = 1
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url, ssl=FLAGS.ssl
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    input_data = np.random.rand(4, 4).astype(np.float32)
    
    print("Input data:")
    print(input_data)
    
    inputs = [
        grpcclient.InferInput('INPUT', input_data.shape, "FP32")
    ]
    inputs[0].set_data_from_numpy(input_data)
    
    outputs = [
        grpcclient.InferRequestedOutput('OUTPUT')
    ]
    
    results = triton_client.infer(
        model_name='example',
        inputs=inputs,
        outputs=outputs
    )
    
    output_data = results.as_numpy('OUTPUT')

    print("Output data:")
    print(output_data)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-s",
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL encrypted channel to the server",
    )
    FLAGS = parser.parse_args()
    
    main()