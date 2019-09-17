import json
from enum import Enum
import sjson
import six

class VkKeyboardColor(Enum):
        """ Возможные цвета кнопок """

        #: Синяя
        PRIMARY = 'primary'

        #: Белая
        DEFAULT = 'default'

        #: Красная
        NEGATIVE = 'negative'

        #: Зелёная
        POSITIVE = 'positive'

class VkKeyboard(object):
    """ Класс для создания клавиатуры для бота (https://vk.com/dev/bots_docs_3)

    :param one_time: Если True, клавиатура исчезнет после нажатия на кнопку
    :type one_time: bool
    """

    def __init__(self, one_time=False):
        self.one_time = one_time
        self.lines = [[]]

        self.keyboard = {
            "one_time": self.one_time,
            "buttons": self.lines
        }

    def get_keyboard(self):
        """ Получить json клавиатуры """
        print(json.dumps(self.keyboard, ensure_ascii=False).encode("utf-8"))
        return json.dumps(self.keyboard, ensure_ascii=False).encode("utf-8")


    @classmethod
    def get_empty_keyboard(cls):
        """ Получить json пустой клавиатуры.
        Если отправить пустую клавиатуру, текущая у пользователя исчезнет.
        """
        keyboard = cls()
        keyboard.keyboard['buttons'] = []
        return keyboard.get_keyboard()


    def add_button(self, label, color=VkKeyboardColor.DEFAULT, payload=None):
        """ Добавить кнопку. Максимальное количество кнопок на строке - 4

        :param label: Надпись на кнопке и текст, отправляющийся при её нажатии.
        :type label: str

        :param color: цвет кнопки.
        :type color: VkKeyboardColor or str

        :param payload: Параметр для callback api
        :type payload: str or list or dict
        """

        if len(self.lines[-1]) == 4:
            self.lines.append([])

        current_line = self.lines[-1]

        color_value = color

        if isinstance(color, VkKeyboardColor):
            color_value = color_value.value

        if payload is not None and not isinstance(payload, six.string_types):
            payload = json.dumps(payload)

        current_line.append({
            'action': {
                'type': 'text',
                'payload': payload,
                'label': label,
            },
            'color': color_value
        })
