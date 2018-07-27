class Value:
    def __init__(self):
        self.value = None

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = (1 - instance.comission) * value


class Account:
    amount = Value()

    def __init__(self, comission):
        self.comission = comission


new_account = Account(0.1)
new_account.amount = 100

print(new_account.amount)
