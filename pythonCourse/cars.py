import csv
import os


class BaseCar:
    def __init__(self, car_type, photo_file_name, brand, carrying):
        self.car_type = car_type
        self.photo_file_name = photo_file_name
        self.brand = brand
        self.carrying = carrying

    def get_photo_file_ext(self):
        return os.path.splitext(self.photo_file_name)[1]


class Car(BaseCar):
    def __init__(self, car_type, photo_file_name, brand, carrying, passenger_seats_count):
        BaseCar.__init__(self, car_type, photo_file_name, brand, carrying)
        self.passenger_seats_count = passenger_seats_count


class Truck(BaseCar):
    def __init__(self, car_type, photo_file_name, brand, carrying, body_width, body_height, body_length):
        BaseCar.__init__(self, car_type, photo_file_name, brand, carrying)
        self.body_width = body_width
        self.body_height = body_height
        self.body_length = body_length

    @staticmethod
    def get_whl(string):
        if string == '':
            return 0, 0, 0
        whl = string.split('x')
        try:
            whl = [float(i) for i in whl]
        except ValueError:
            return 0, 0, 0
        return whl[0], whl[1], whl[2]

    def get_body_volume(self):
        return self.body_height * self.body_length * self.body_width


class SpecMachine(BaseCar):
    def __init__(self, car_type, photo_file_name, brand, carrying, extra):
        BaseCar.__init__(self, car_type, photo_file_name, brand, carrying)
        self.extra = extra


def get_car_list(csv_file_name):
    car_list = []
    with open(csv_file_name) as csv_fd:
        reader = csv.reader(csv_fd, delimiter=';')
        next(reader)
        for row in reader:
            # print(row)
            if len(row) != 7:
                # print('err')
                continue
            car_type = row[0]
            brand = row[1]
            photo_file_name = row[3]
            carrying = row[5]
            if row[0] == 'car':
                try:
                    psc = int(row[2])
                except ValueError:
                    continue
                car_list.append(Car(car_type, photo_file_name, brand, carrying, psc))
            if row[0] == 'truck':
                w, h, l = Truck.get_whl(row[4])
                car_list.append(Truck(car_type, photo_file_name, brand, carrying, w, h, l))
            if row[0] == 'spec_machine':
                extra = row[6]
                car_list.append(SpecMachine(car_type, photo_file_name, brand, carrying, extra))
        return car_list


print(get_car_list('coursera_week3_cars.csv'))
