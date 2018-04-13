class LazyFileReader:

    def __init__(self, file):
        self.file = file
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos == 0:
            self.chanck = self.file.read(512)

        if self.chanck is None:
            raise StopIteration

        res = self.chanck[self.pos]
        self.pos = (self.pos + 1) % 512
        return res


f = open("test.log", mode="rb")
for chanck in LazyFileReader(f):
    print(chanck)
