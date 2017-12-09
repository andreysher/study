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
#define STARTBUFFERSIZE (4096)
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
	if (2 != argc)
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

	for ( ; ; )
	{
		printf("poll %d %d\n", sockets, SOCKETNUMBER);

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
			printf("serv\n");

			struct sockaddr_in incomingAddress;
			int incomingAddressLength = sizeof(incomingAddress);

			int incomingSocket = accept(serverSocket, (struct sockaddr *)&incomingAddress, (socklen_t *)&incomingAddressLength);

			if (-1 == incomingSocket)
			{
				perror("Error while accept()");

				exit(EXIT_FAILURE);
			}

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

		for (int i = 0; (i < INCOMINGNUMBER) && (readyCount > 0); i++)
		{
			if (0 != (incoming[i].revents & POLLIN))
			{
				printf("in in\n");

				int received = recv(incoming[i].fd, inAttrs[i].buffer + inAttrs[i].end, inAttrs[i].size - inAttrs[i].end, 0);
				printf("recv return %d\n", received);
				switch (received)
				{
					case -1:
						perror("Error while read()");

						exit(EXIT_FAILURE);

					case 0:
						closeConnection(incoming[i]);

						break;

					default:
						inAttrs[i].end += received;

						if (inAttrs[i].end == inAttrs[i].size)
						{
							inAttrs[i].size *= 2;

							char * result = (char *)realloc(inAttrs[i].buffer, inAttrs[i].size);

							if (NULL == result)
							{
								perror("Error while incoming realloc");

								exit(EXIT_FAILURE);
							}
							else
							{
								inAttrs[i].buffer = result;
							}
						}

						printf("in %s\n", inAttrs[i].buffer);

						if (outcomingCount <= i)
						{
							char * enter = strchr(inAttrs[i].buffer, '\n');

							if (enter != NULL)
							{
								int tempLength = enter - inAttrs[i].buffer;

								char * temp = new char[tempLength + 1];
								strncpy(temp, inAttrs[i].buffer, tempLength);
								temp[tempLength] = 0;

								char *method = strtok(temp, " ");
								//если null то работаем с предыдущим токеном
								char *uri = strtok(NULL, " ");
								char *version = strtok(NULL, "\n\0");

								printf("method: %s\n", method);
								printf("uri: %s\n", uri);
								printf("version: %s\n", version);

								char * protocolEnd = strstr(uri, "://");

								if (NULL == protocolEnd)
								{
									perror("Wrong protocol");

									closeConnection(incoming[i]);
									exit(EXIT_FAILURE);
								}

								char *hostEnd = strchr(protocolEnd + 3, '/');

								int hostLength = 0;

								printf("ok\n");

								if (NULL == hostEnd)
								{
									hostLength = strlen(protocolEnd + 3);
								}
								else
								{
									hostLength = hostEnd - (protocolEnd + 3);
								}

								char * hostName = new char[hostLength + 1];
								strncpy(hostName, protocolEnd + 3, hostLength);
								hostName[hostLength] = 0;

								printf("hostName: \'%s\'\n", hostName);

								struct hostent * hostInfo = gethostbyname(hostName);

								if (NULL == hostInfo)
								{
									perror("Error while gethostbyname");

									closeConnection(incoming[i]);

									exit(EXIT_FAILURE);
								}

								struct sockaddr_in destinationAddress;

								destinationAddress.sin_family = AF_INET;
								destinationAddress.sin_port = htons(DEFAULTPORT);
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

//								freehostent(hostInfo);
								delete[] temp;
							}
						}

						outcoming[i].events |= POLLOUT;

						break;
				}

				readyCount--;
			}
			else if (0 != (incoming[i].revents & POLLOUT))
			{
				printf("in out\n");

				int sent = send(incoming[i].fd, outAttrs[i].buffer + outAttrs[i].start, outAttrs[i].end - outAttrs[i].start, 0);

				switch (sent)
				{
					case -1:
						perror("Error while write()");

						exit(EXIT_FAILURE);

					case 0:
						closeConnection(incoming[i]);

						break;

					default:
						outAttrs[i].start += sent;

						if (outAttrs[i].start == outAttrs[i].end)
						{
							incoming[i].events &= ~POLLOUT;
						}
				}
			}
		}

		for (int i = 0; i < OUTCOMINGNUMBER; i++)
		{
			if (0 != (outcoming[i].revents & POLLIN))
			{
				printf("out in\n");

				int received = recv(outcoming[i].fd, outAttrs[i].buffer + outAttrs[i].end, outAttrs[i].size - outAttrs[i].end, 0);

				printf("received %d\n", received);

				switch (received)
				{
					case -1:
						perror("Error while recv(outcoming).\n");

						exit(EXIT_FAILURE);
					case 0:
						close(outcoming[i].fd);
						outcoming[i].fd = -1;

						break;

					default:
						outAttrs[i].end += received;

						if (outAttrs[i].end == outAttrs[i].size)
						{
							outAttrs[i].size *= 2;

							char * result = (char *)realloc(outAttrs[i].buffer, outAttrs[i].size);

							if (NULL == result)
							{
								perror("Error while outcoming realloc.\n");

								exit(EXIT_FAILURE);
							}
							else
							{
								outAttrs[i].buffer = result;
							}
						}

						incoming[i].events |= POLLOUT;
				}
			}
			else if (0 != (outcoming[i].revents & POLLOUT))
			{
				printf("out out %d %d\n", outcoming[i].events, POLLIN);

				int sent = send(outcoming[i].fd, inAttrs[i].buffer, inAttrs[i].end - inAttrs[i].start, 0);

				switch (sent)
				{
					case -1:
						perror("Error while send(outcoming).\n");

						break;
					case 0:
						close(outcoming[i].fd);
						outcoming[i].fd = -1;

						break;
					default:
						inAttrs[i].start += sent;

						if (inAttrs[i].start == inAttrs[i].end)
						{
							outcoming[i].events &= ~POLLOUT;
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