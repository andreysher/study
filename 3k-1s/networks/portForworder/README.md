Port forwarder

Реализовать программу для перенаправления TCP-соединений (port forwarder), используя неблокирующийся ввод-вывод.
В параметрах программе передаются <lport> <rhost> <rport>
Форвардер ждёт TCP-соединений на <lport>.
При подключении клиента форвардер открывает новое TCP-соединение на <rhost>:<rport> и пересылает данные в обе стороны без изменений.
В случае невозможности установить соединение с <rhost>:<rport> (в том числе, если имя <rhost> не резолвится), форвардер закрывает соединение с клиентом, не передавая и не принимая никаких данных.
Для реализации форвардера использовать неблокирующийся ввод-вывод в рамках одного треда. Дополнительные треды использовать не допускается. Соответственно, никаких блокирующихся вызовов не допускается (кроме резолвинга DNS-имен, с которыми всё печально в стандартной библиотеке Java, но решается с помощью внешних библиотек. Для простоты можно резолвить имя один раз при старте программы).
Для тестирования работы форвардера направить его на какой-нибудь веб-сайт, например, lport=10080, rhost=ip_piwigo, rport=80, где ip_piwigo - это IP-адрес хоста piwigo.org, добавить в /etc/hosts запись "127.0.0.1 piwigo.org", затем открыть в браузере http://piwigo.org:10080/demo/index.php?/category/113. Не забудьте убрать запись из /etc/hosts после тестирования :)
Желательно проверить на нескольких нетривиальных сайтах, где много ресурсов хранится на одном и том же хосте (либо запустить несколько копий форвардера и перенаправить основные хосты, например, en.wikipedia.org и upload.wikimedia.org для Википедии).
В приложении не должно быть холостых циклов ни в каких ситуациях. Производительность не должна быть заметно хуже, чем без форвардера. Для отслеживания корректности и скорости работы можно глядеть в Developer tools браузера на вкладку Network.

Баллов за задачу: 2.