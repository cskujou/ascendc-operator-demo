import numpy as np


def gen_golden_data_simple(shape, dtype):
    input_x = np.random.uniform(1, 100, shape).astype(dtype)
    input_y = np.random.uniform(1, 100, shape).astype(dtype)
    golden = (input_x + input_y).astype(dtype)

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    shape = (8, 2048)
    dtype = np.float16
    gen_golden_data_simple(shape, dtype)