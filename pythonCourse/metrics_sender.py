import socket
import time
import sys


class ClientError:
    pass


class Client:
    def __init__(self, host, port, timeout=None):
        self.sock = socket.create_connection((host, port))
        self.sock.settimeout(timeout)

    def put(self, key, value, timestamp=str(int(time.time()))):
        try:
            put_req = 'put {0} {1} {2}\n'.format(key, value, timestamp)
            self.sock.sendall(put_req.encode("utf-8"))

            response = self.sock.recv(1024).decode('utf-8')
            if response == 'ok\n\n':
                return
            else:
                raise ClientError
        except socket.timeout:
            print('send request timeout', file=sys.stderr)
        except socket.error:
            print('send data error', file=sys.stderr)

    def get(self, key):
        """Возвращает словарь метрика-список логов"""
        try:
            result = {}
            get_req = 'get {0}\n'.format(key)
            self.sock.sendall(get_req.encode('utf-8'))

            response = ''
            while True:
                data = self.sock.recv(1024)
                # if not data:
                #     break
                response += data.decode('utf-8')
                if response.endswith('\n\n'):
                    break
            if not response.startswith('ok'):
                raise ClientError

            metrics = response.split('\n')[1:-2]
            for metric in metrics:
                metric = metric.split(' ')
                metric_name = metric[0]
                metric_value = (float(metric[1]), int(metric[2]))
                if metric_name not in result.keys():
                    result[metric_name] = [metric_value]
                else:
                    result[metric_name].append(metric_value)
            return result
        except socket.timeout:
            print('send request timeout', file=sys.stderr)
        except socket.error:
            print('send data error', file=sys.stderr)


client = Client('127.0.0.1', 8181)
client.put('key2', 1.0)
print(client.get('key'))
