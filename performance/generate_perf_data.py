import argparse
import json
import numpy as np

def main():
    input_data = np.random.rand(FLAGS.num_rows, 4).astype(np.float32)

    data = {
        "data": [
            {
                "INPUT": {
                    "content": input_data.flatten().tolist(),
                    "shape": list(input_data.shape),
                }
            }
        ]
    }

    with open(FLAGS.output, "w") as f:
        json.dump(data, f)

    print(f"Wrote {FLAGS.num_rows}x4 FP32 input to {FLAGS.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output",
        type=str, default="perf_data.json",
        help="Output JSON file path. Default is perf_data.json.",
    )
    parser.add_argument(
        "-n", "--num-rows",
        type=int, default=100,
        help="Number of rows in the input tensor. Default is 10000.",
    )
    FLAGS = parser.parse_args()

    main()
