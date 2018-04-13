import os
import sys


def sorted_list_of_files() -> list:
    if (len(sys.argv) < 2):
        print("Enter directory path")
        return None

    if not os.path.isdir(sys.argv[1]):
        print("Enter DIRECTORY path")
        return None

    files = []
    inDir = os.listdir(sys.argv[1])
    for name in inDir:
        filePath = os.path.join(sys.argv[1], name)
        if (os.path.isfile(filePath)):
            files.append((name, os.stat(filePath).st_size))

    files = sorted(files, key=lambda f: (-f[1], f[0]))

    return files


if __name__ == '__main__':
    sortedFiles = sorted_list_of_files()
    print(sortedFiles)
