import codecs
import task4
import sys

def predict_encoding(filename, frequencies, encodings):
    freqs = task4.get_byte_freq(filename)
    encoding_error = []
    for enc in encodings:
        error = 0
        for char in freqs:
            # print(char[0])
            # print(bytes([char[0]]))
            try:
                # print(char[0])
                ch = bytes(bytes([char[0]])).decode(enc).lower()
            except UnicodeDecodeError:
                ch = 0
            # print(ch)
            if not ch in frequencies:
                freq = 0
            else:
                freq = frequencies[ch]
            error += abs(char[1] - freq)
        encoding_error.append((enc, error))
        # print(encoding_error)

    encoding_error.sort(key=lambda x:x[1])
    # print(encoding_error)
    # print(freqs)
    return encoding_error[0][0]

if __name__ == '__main__':
    encodings = ['koi8-r', 'cp855', 'cp866', 'iso8859-5', 'cp1251', 'mac_cyrillic']

    frequencies = {'о': 0.10983, 'е': 0.08483, 'а': 0.07998, 'и': 0.07367, 'н': 0.067, 'т': 0.06318,
                   'с': 0.05473, 'р': 0.04746, 'в': 0.04533, 'л': 0.04343, 'к': 0.03486, 'м': 0.03203,
                   'д': 0.02977, 'п': 0.02804, 'у': 0.02615, 'я': 0.02001, 'ы': 0.01898, 'ь': 0.01735,
                   'г': 0.01687, 'з': 0.01641, 'б': 0.01592, 'ч': 0.0145, 'й': 0.01208, 'х': 0.00966,
                   'ж': 0.0094, 'ш': 0.00718, 'ю': 0.00639,'ц': 0.00486, 'щ': 0.00361, 'э': 0.00331,
                   'ф': 0.00267, 'ъ': 0.00037, 'ё': 0.00013}
    encoding = predict_encoding(sys.argv[1],frequencies,encodings)
    # print(encoding)
    # f = codecs.open(sys.argv[1], 'r', encoding=encoding,errors='ignore')
    if input('print file with encoding ' + encoding + ' y\\n\n') == 'y':
        with codecs.open(sys.argv[1],'r', encoding=encoding, errors='ignore') as f:
            s = f.read(1024*1024)
            while s:
                print(s)
                s = f.read(1024*1024)
