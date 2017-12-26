import com.sun.xml.internal.bind.v2.runtime.reflect.opt.Const;

import java.io.IOException;
import java.net.*;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.LinkedBlockingQueue;

class Background {
    private static final Object monitor = new Object();
    private BlockingQueue<Socket> acceptedSockets = new LinkedBlockingQueue<>();
    private ConcurrentMap<SocketAddress, Buffer> clients = new ConcurrentHashMap<>();
    private Thread backgroundThread = new Thread(this::execute);
    private DatagramSocket socket;
    private boolean isServerStopped = true;

    private final SocketProxy proxy;

    Background(int port) throws SocketException {
        socket = new DatagramSocket(port);

        proxy = new SocketProxy(socket);

        socket.setSoTimeout(Constants.REC_TIMEOUT);
        isServerStopped = false;
        backgroundThread.start();
    }

    Background(InetSocketAddress address, Buffer client) throws SocketException, TCPSocketException {
        socket = new DatagramSocket();

        proxy = new SocketProxy(socket);

        int count = Constants.COUNT_EFF;
        socket.setSoTimeout(Constants.REC_TIMEOUT);
        client.putToSendMessage((new MessageCreator(0)).genSYNMsg());
        clients.put(address, client);//кому хотим отпр
        while (count > 0 && !client.isConnected()) {
            sendAllMessages();
            recvMessage();
            count--;
        }

        if (!client.isConnected()) {
            throw new TCPSocketException("Can't reach the address!");
        }

        backgroundThread.start();
    }

    Socket getAccepted() throws TCPSocketException {
        try {
            return acceptedSockets.take();
            //заблокируется пока в очередь не положат хотя бы один сокет
        } catch (InterruptedException ex) {
            System.out.println(ex.getMessage());
            throw new TCPSocketException("Can't accept incoming connection...");
        }
    }

    void stop(SocketAddress addr) {//останавливаем соединение
        if (null == addr) {//
            isServerStopped = true;
        } else {
            if (clients.containsKey(addr)) {
                flushBuffer(addr);
                clients.remove(addr);
            }
        }

        if (!clients.isEmpty() || isServerStopped) {
            backgroundThread.interrupt();
        }
    }

    private void flushBuffer(SocketAddress address) {//НАДО КИДАТЬ ЭКСЕПШИОН ЕСЛИ ДАННЫЕ НЕ БЫЛИ ДОСТАВЛЕНЫ ВООБЩЕ
        Buffer client = clients.get(address);
        int count = Constants.COUNT_EFF;


        synchronized (monitor) {
            while (client.hasMessages() && count > 0) {
                count--;

                sendAllMessagesOfClient(address);
                while (-1 != recvMessage()) {
                }
            }

            if (client.hasMessages()) {//если все сообщ не удалось отпр.
                client.cleanSendBuffer();
            }

            closeConnection(address);
        }
    }

    private void execute() {
        while (!Thread.currentThread().isInterrupted()) {
            recvMessage();
            sendAllMessages();
        }
    }

    private void sendAllMessages() {
        Iterator itr = clients.entrySet().iterator();
        while(itr.hasNext()){
            Map.Entry clEntry = (Map.Entry) itr.next();
            SocketAddress clAddr = (SocketAddress) clEntry.getKey();
            if(clients.get(clAddr).isDead()){
                itr.remove();
                System.out.println("UBILI");
            }
            else {
                sendAllMessagesOfClient(clAddr);
            }
        }
    }

    private void sendAllMessagesOfClient(SocketAddress address) {
        Buffer clientBuf = clients.get(address);//достаем буфер клиента
        Map<Integer, Message> clientMsgs = clientBuf.getMsgsToSend();

        synchronized (monitor) {
            clientMsgs.entrySet().stream().filter(msg -> msg.getValue().isOld()).forEach(msg -> {
                byte[] data = msg.getValue().getMessage().array();
                try {
                    socket.send(new DatagramPacket(data, data.length, address));
                } catch (IOException ex) {
                    System.out.println(ex.getMessage());
                }
                msg.getValue().resetTime();
            });
        }
    }

    private int rrr = 0;
    private int recvMessage() {
        byte[] data = new byte[Constants.MAX_BUFFER_SIZE];
        DatagramPacket packet = new DatagramPacket(data, data.length);

        try {
            socket.receive(packet);
            proxy.put(packet);
            if (proxy.isReady())
                packet = proxy.get();
            else
                return -1;

            parseMessage(packet);
        } catch (SocketTimeoutException ex) {
            return -1;
        } catch (IOException ex) {
            System.out.println(ex.getMessage());
        }

        return 0;
    }

    private void closeConnection(SocketAddress address) {
        Buffer client = clients.get(address);
        int count = Constants.COUNT_EFF;

        client.putToSendMessage((new MessageCreator(client.getSendMsgId())).genDeadMsg());
        while (client.hasMessages() && count > 0) {
            count--;
            sendAllMessagesOfClient(address);
            while (-1 != recvMessage()){//когда ничего не пришло в течении таймаута
            }//дожидаемся пока все аки дойдут
        }
    }

    private void parseMessage(DatagramPacket packet) throws IOException {
        MessageCreator msg = new MessageCreator(packet.getData());
        switch (msg.messageType()) {
            case SYN:
                handleSYNMsg(msg, packet.getSocketAddress());
                break;
            case ACK:
                rrr++;
                if (rrr > 5) {
                    System.out.println(123321);
                    break;
                }
                handleAckMsg(msg, packet.getSocketAddress());
                break;
            case SYNACK:
                handleSYNAck(msg, packet.getSocketAddress());
                break;
            case PAYLOAD:
                handlePayLoad(msg, packet.getSocketAddress());
                break;
            case DEAD:
                handleDead(msg, packet.getSocketAddress());
                break;
            case DEADACK:
                handleDeadAck(msg, packet.getSocketAddress());
        }
    }

    private void handleSYNMsg(MessageCreator msg, SocketAddress address) {
        if (isServerStopped) {
            return;
        }
        if (!clients.containsKey(address)) {
            Buffer buf = new Buffer();

            clients.put(address, buf);
            try {
                acceptedSockets.put(new Socket(address, buf, this));//положили сокет
            } catch (InterruptedException ex) {
                System.out.println(ex.getMessage());
            }
        }
        Buffer clientBuf = clients.get(address);//достали клиентский буфер
        clientBuf.putToSendMessage(
        (new MessageCreator(clientBuf.getSendMsgId())).genSYNAckMsg(msg.getMessageId()));
        //положили синак msg,msgid-id сообщ кот хотим отпр
        clientBuf.connected();
    }

    private void handleAckMsg(MessageCreator msg, SocketAddress address) {
        if (!clients.containsKey(address)) {
            return;
        }
        Buffer clientBuf = clients.get(address);

        if (!clientBuf.isConnected()) {//если клиент не подсоединен
            return;
        }
        clientBuf.removeFromSendMessage(msg.getACKED());
    }

    private void handleSYNAck(MessageCreator msg, SocketAddress address) throws IOException {
        Buffer clientBuf = clients.get(address);//достали клиентский буфер
        clientBuf.removeFromSendMessage(msg.getACKED());//удаляем подтвержение сообщения из очереди на отправку
        byte[] data = (new MessageCreator(0)).genAckMsg(msg.getMessageId()).array();
        synchronized (monitor) {//к сокету конурентный доступ
            socket.send(new DatagramPacket(data, data.length, address));
        }

        clientBuf.connected();
    }

    private void handlePayLoad(MessageCreator msg, SocketAddress address) throws IOException {
        if (!msg.isValid()){
            return;
        }
        if (!clients.containsKey(address)) {
            return;
        }

        Buffer clientBuf = clients.get(address);//достаем буфер по адресу

        if (!clientBuf.isConnected()) {//если соедин разорвано
            return;
        }

        clientBuf.putToReceive(msg.getPayloadId(), msg.getPayload());
        //кладем в буфер принятых сообщ
        byte[] data = (new MessageCreator(0)).genAckMsg(msg.getMessageId()).array();
        synchronized (monitor){
            socket.send(new DatagramPacket(data, data.length, address));
        }
    }

    private void handleDead(MessageCreator msg, SocketAddress address) {
        if (!clients.containsKey(address)) {
            return;
        }

        Buffer client = clients.get(address);
        client.putToSendMessage((new MessageCreator(client.getSendMsgId()).genDeadAckMsg(msg.getMessageId())));
        client.dead();
    }

    private void handleDeadAck(MessageCreator msg, SocketAddress address) throws IOException {
        if (!clients.containsKey(address)) {
            return;
        }

        byte[] data = (new MessageCreator(0)).genAckMsg(msg.getMessageId()).array();
        synchronized (monitor){
            socket.send(new DatagramPacket(data, data.length, address));
        }
        clients.get(address).dead();//достали буфер умершего клиента по адресу и выставили его как умершего
    }
}