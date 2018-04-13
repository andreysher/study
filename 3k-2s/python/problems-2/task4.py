import sys
import os


def get_byte_freq(filename):
    with open(filename, 'rb') as file:
        counter = 0
        byte_counter = {}
        while True:
            buffer = file.read(1024 * 1024)
            if not buffer:
                break
            for byte in buffer:
                if byte > 127:
                    counter += 1
                    byte_counter[byte] = byte_counter.get(byte, 0) + 1
                    # if byte not in byte_counter:
                    #     byte_counter[byte] = 0
                    #
                    # byte_counter[byte] += 1

    for byte, f in byte_counter.items():
        byte_counter[byte] = f / counter

    sorted_tuples = sorted(byte_counter.items(), key=lambda x: x[1], reverse=True)

    # print(sorted_tuples)

    return sorted_tuples

    # bytes_freqs = {byte: freq / counter for byte, freq in byte_counter.items()}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error! Need file name")

    if not os.access(sys.argv[1], os.R_OK):
        print("Error! Not enough access rights")

    if not os.path.isfile(sys.argv[1]):
        print("Error! Input path is not file")

    byte_freq = get_byte_freq(sys.argv[1])

    print(byte_freq)
