Чат-дерево

Разработать приложение для узла "надежной" сети для передачи сообщений. Узлы объединены в дерево и обмениваются UDP сообщениями. Каждый узел знает только своего непосредственного предка и непосредственных потомков. Приложение принимает в качестве параметров имя узла, процент потерь, собственный порт, а также опционально IP адрес и порт узла родителя. Приложение, которому не был передан IP адрес и порт узла родителя, становится корнем дерева.

Сообщение, введенное на любом из узлов сети, передается на все остальные узлы и выводится ровно один раз. Закрытие любого узла сети с помощью Ctrl-C не нарушает связность сети. Все сообщения идентифицируются с помощью GUID. Для обеспечения "надёжности" доставка сообщений должна быть подтверждена.
Для реализации требований, каждый узел может вести учёт отправленных и полученных сообщений, однако неограниченное расходование памяти не допускается.

Важно, что переотправка сообщений вследствие потерь не должна приводить к задержкам в доставке других сообщений, и не должна блокировать работу остальных функций программы.
При поступлении любого входящего сообщения, узел генерирует случайное число от 0 до 99 включительно. Если это число строго меньше, чем заданный в параметрах процент потерь, сообщение игнорируется полностью, имитируя сетевую потерю пакета. Это необходимо для тестирования надёжности доставки сообщений.

Баллов за задачу: 3.