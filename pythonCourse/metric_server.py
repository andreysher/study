import asyncio
import re

metrics = dict()


def process_data(data):
    if re.match('put \w+ [0-9]+[.]*[0-9]+ [0-9]+\n', data):
        req = data.split()
        if req[1] not in metrics.keys():
            metrics[req[1]] = [(float(req[2]), int(req[3]))]
        else:
            metrics[req[1]].append((float(req[2]), int(req[3])))
        return 'ok\n\n'

    if data == 'get *\n':
        resp = 'ok\n'
        for metric in metrics.items():
            for val in metric[1]:
                resp += '{0} {1} {2}\n'.format(metric[0], val[0], val[1])
        resp += '\n'
        return resp

    if re.match('get \w+\n', data):
        req = data.split()
        if req[1] in metrics.keys():
            resp = 'ok\n'
            for val in metrics[req[1]]:
                resp += '{0} {1} {2}\n'.format(req[1], val[0], val[1])
            resp += '\n'
            return resp
        return 'ok\n\n'

    return 'error\nwrong command\n\n'


class ClientServerProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        resp = process_data(data.decode())
        self.transport.write(resp.encode())


def start_server(host, port):
    loop = asyncio.get_event_loop()
    coro = loop.create_server(ClientServerProtocol, host, port)
    server = loop.run_until_complete(coro)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()


start_server('127.0.0.1', 8181)
