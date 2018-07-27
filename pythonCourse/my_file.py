import os
import tempfile


class File:
    def __init__(self, full_path):
        self.full_path = full_path
        self.name = os.path.split(full_path)[-1]
        self.file = open(self.full_path, 'a+')

    def write(self, s):
            self.file.write(s)

    def __add__(self, other):
        if type(other) == File:
            res_path = os.path.join(tempfile.gettempdir(), self.name + '_' + other.name)
            res = File(res_path)
            for row in self.file:
                res.write(row)
            for row in other.file:
                res.write(row)
            return res

    def __repr__(self):
        return self.full_path

    def __iter__(self):
        self.file.seek(0)
        return self

    def __del__(self):
        self.file.close()

    def __next__(self):
        row = self.file.readline().strip()
        if row:
            return row
        else:
            raise StopIteration
