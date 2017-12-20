#include <cstdlib>//exit, atoi
#include <cstdio>//perror
#include <netdb.h>//gethostbyname
#include <cstring>//memcpy
#include <sys/socket.h>//sockaddr_in, AF_INET, socket, bind, listen, connect
#include <poll.h>//poll
#include <unistd.h>//read, write, close
#include <arpa/inet.h>//htonl, htons

#define INCOMINGNUMBER 510
#define OUTCOMINGNUMBER 510
#define SOCKETNUMBER (1 + INCOMINGNUMBER + OUTCOMINGNUMBER)//1 for serverSocket and 510 pairs for connections
#define STARTBUFFERSIZE (1024*10)
#define DEFAULTPORT 80

struct InAttr
{
  char *buffer;

  int start;
  int end;

  int size;
};

struct OutAttr
{
  char *buffer;

  int start;
  int end;

  int size;
};

void closeConnection(struct pollfd & connection)
{
  close(connection.fd);
  connection.fd = -1;
  connection.events = 0;
}

int main(int argc, char *argv[])
{
  if (4 != argc)
  {
    perror("Wrong number of arguments.\n");

    exit(EXIT_FAILURE);
  }

  int serverPort = atoi(argv[1]);

  if (0 == serverPort)
  {
    perror("Wrong port for listening.\n");

    exit(EXIT_FAILURE);
  }

//======================================================================================

  int serverSocket = socket(PF_INET, SOCK_STREAM, 0);

  if (-1 == serverSocket)
  {
    perror("Error while creating serverSocket.\n");

    exit(EXIT_FAILURE);
  }

  {
    struct sockaddr_in serverAddress;

    serverAddress.sin_family = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddress.sin_port = htons(serverPort);

    if (-1 == (bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress))))
    {
      perror("Error while binding.\n");

      exit(EXIT_FAILURE);
    }

    if (-1 == listen(serverSocket, 1024))
    {
      perror("Error while listen().\n");

      exit(EXIT_FAILURE);
    }
  }
//подготовка сервер сокета
//======================================================================================

  struct pollfd *sockets = new struct pollfd[SOCKETNUMBER];

  struct pollfd *incoming = sockets + 1;
  int incomingCount = 0;

  struct pollfd *outcoming = incoming + INCOMINGNUMBER;
  int outcomingCount = 0;

  struct InAttr *inAttrs = new struct InAttr[INCOMINGNUMBER];
  struct OutAttr *outAttrs = new struct OutAttr[OUTCOMINGNUMBER];

  for (int i = 0; i < SOCKETNUMBER; i++)
  {
    sockets[i].fd = -1;
    sockets[i].events = 0;
  }

  sockets[0].fd = serverSocket;
  sockets[0].events = POLLIN;
// создали пары incoming+outcoming, подготовились к поллу
  for ( ; ; )
  {
    printf("poll %d %d\n", sockets, SOCKETNUMBER);
    printf("\n\n\n\n%d () %d\n\n\n\n\n\n", incomingCount, outcomingCount);
    int readyCount = poll(sockets, SOCKETNUMBER, -1);

    printf("poll done\n");

    if (-1 == readyCount)
    {
      perror("Error while poll()");

      exit(EXIT_FAILURE);
    }
    else if (0 == readyCount)
    {
      perror("poll() returned 0");

      continue;
    }

    if (0 != (sockets[0].revents & POLLIN))
    {
      printf("server socket ready for accept\n");
      struct sockaddr_in incomingAddress;
      int incomingAddressLength = sizeof(incomingAddress);

      int incomingSocket = accept(serverSocket, (struct sockaddr *)&incomingAddress, (socklen_t *)&incomingAddressLength);

      if (-1 == incomingSocket)
      {
        perror("Error while accept()");

        exit(EXIT_FAILURE);
      }
      //пришло соединение на сокет - записали его в incoming
      incoming[incomingCount].fd = incomingSocket;
      incoming[incomingCount].events = POLLIN;

      inAttrs[incomingCount].start = 0;
      inAttrs[incomingCount].end = 0;
      inAttrs[incomingCount].size = STARTBUFFERSIZE;
      inAttrs[incomingCount].buffer = (char *)malloc(inAttrs[incomingCount].size * sizeof(*(inAttrs[incomingCount].buffer)));

      memset(inAttrs[incomingCount].buffer, 0, inAttrs[incomingCount].size);

      incomingCount++;

      readyCount--;
    }

    for (int i = 0; i < INCOMINGNUMBER && readyCount > 0; i++)
    {
        if (0 != (incoming[i].revents & POLLIN))
        {
            printf("on incoming %d get input\n" , i);

            int received = recv(incoming[i].fd, inAttrs[i].buffer + inAttrs[i].end, inAttrs[i].size - inAttrs[i].end, 0);
            //получаем столько сколько можем получить

            printf("%s\n", inAttrs[i].buffer);

            switch (received)
            {
                case -1:
                    perror("Error while recv()");
                    exit(EXIT_FAILURE);
                case 0:
                    // free(inAttrs[i].buffer);
                    closeConnection(incoming[i]);
                    incomingCount--;
                    break;
                default:
                    inAttrs[i].end += received;
                    if (inAttrs[i].end == inAttrs[i].size)
                    {
                        incoming[i].events &= ~POLLIN;
                        //pollin больше не ждем!
                    }
                    // printf("in %s\n", inAttrs[i].buffer);
                    //если на incoming что-то пришло, а outcoming для него не создан:
                    while (outcomingCount <= i)
                    {
                        struct hostent * hostInfo = gethostbyname(argv[2]);
                        if (NULL == hostInfo)
                        {
                            perror("Error while gethostbyname");
                            closeConnection(incoming[i]);
                        }
                        struct sockaddr_in destinationAddress;
                        destinationAddress.sin_family = AF_INET;
                        destinationAddress.sin_port = htons(atoi(argv[3]));
                        memcpy(&destinationAddress.sin_addr, hostInfo->h_addr, hostInfo->h_length);
                        int httpSocket = socket(AF_INET, SOCK_STREAM, 0);
                        if (-1 == httpSocket)
                        {
                            perror("Error while socket()");
                            exit(EXIT_FAILURE);
                        }
                        if (-1 == connect(httpSocket, (struct sockaddr *)&destinationAddress, sizeof(destinationAddress)))
                        {
                            perror("Error while connect().\n");
                            closeConnection(incoming[i]);
                        }
                        outcoming[outcomingCount].fd = httpSocket;
                        outcoming[outcomingCount].events = POLLIN | POLLOUT;
                        outAttrs[outcomingCount].start = 0;
                        outAttrs[outcomingCount].end = 0;
                        outAttrs[outcomingCount].size = STARTBUFFERSIZE;
                        outAttrs[outcomingCount].buffer = (char *)malloc(outAttrs[outcomingCount].size);
                        outcomingCount++;
                    }
                    outcoming[i].events |= POLLOUT;
                    break;
            }//switch ending

            readyCount--;
        }//ending if (0 != (incoming[i].revents & POLLIN))
        else if (0 != (incoming[i].revents & POLLOUT))
        {
            printf("on %d incoming geting output flag\n", i);
            //отправляем не 0 байт(тестил)
            printf("%s\n", outAttrs[i].buffer);
            int sent = send(incoming[i].fd, outAttrs[i].buffer + outAttrs[i].start, outAttrs[i].end - outAttrs[i].start, 0);
            switch (sent)
            {
              case -1:
                perror("Error while write()");
                exit(EXIT_FAILURE);
              case 0:
                perror("Can not send");
                free(outAttrs[i].buffer);
                closeConnection(incoming[i]);
                exit(EXIT_FAILURE);
              default://больше не хотим отсылать, потому что все отослали, чот могли, теперь надо положить новую порцию данных
                outAttrs[i].start += sent;
                if (outAttrs[i].start == outAttrs[i].end)
                {
                  incoming[i].events &= ~POLLOUT;
                  outAttrs[i].start = 0;
                  outAttrs[i].end = 0;
                }
            }
        }
    }
    // пошли по outcoming
    for (int i = 0; i < OUTCOMINGNUMBER; i++)
    {//есть что считaть
      if (0 != (outcoming[i].revents & POLLIN))
      {
        printf(" on outcoming %d pollin\n", i);
        int received = recv(outcoming[i].fd, outAttrs[i].buffer + outAttrs[i].end, outAttrs[i].size - outAttrs[i].end, 0);
        printf("received %d\n", received);
        printf("%s\n", outAttrs[i].buffer);
        switch (received)
        {
          case -1:
            perror("Error while recv(outcoming).\n");
            exit(EXIT_FAILURE);
          case 0:
            closeConnection(outcoming[i]);
            outcomingCount--;
            // free(outAttrs[i].buffer);
            break;
          default:
            outAttrs[i].end += received;
            if (outAttrs[i].end == outAttrs[i].size)
            {
                outcoming[i].events &= ~POLLIN;
            }
            incoming[i].events |= POLLOUT;
        }
      }
      else if (0 != (outcoming[i].revents & POLLOUT))
      {
        printf("outcoming %d pollout\n", i);

        int sent = send(outcoming[i].fd, inAttrs[i].buffer, inAttrs[i].end - inAttrs[i].start, 0);
        // cout << inAttrs[i].buffer << endl;
        write(0, inAttrs[i].buffer, inAttrs[i].end - inAttrs[i].start);
        switch (sent)
        {
          case -1:
            perror("Error while send(outcoming).\n");

            break;
          case 0:
            perror("Can not send");
            free(outAttrs[i].buffer);
            closeConnection(incoming[i]);
            exit(EXIT_FAILURE);
          default:
            inAttrs[i].start += sent;

            if (inAttrs[i].start == inAttrs[i].end)
            {
              outcoming[i].events &= ~POLLOUT;
              inAttrs[i].start = 0;
              inAttrs[i].end = 0;
            }
        }
      }
    }
  }

  close(serverSocket);

  delete[] inAttrs;
  delete[] outAttrs;
  delete[] sockets;

  return 0;
}

//piwigo ip = 87.98.147.22