class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        try:
            with open(self.file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ''


reader = FileReader('drow_steps.py')
print(reader.read())
