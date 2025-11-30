import numpy as np


def gen_golden_data_simple(shape, tile_num, dtype):
    total_size = np.array(shape).prod()
    input_x = np.random.uniform(-1000, 1000, shape).astype(dtype)
    input_y = np.random.uniform(-1000, 1000, shape).astype(dtype)
    golden = (input_x + input_y).astype(dtype)
    tiling = np.array([total_size, tile_num], dtype=np.uint32)

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")
    tiling.tofile("./input/input_tiling.bin")


if __name__ == "__main__":
    gen_golden_data_simple(shape=(8, 4096), tile_num=8, dtype=np.float16)