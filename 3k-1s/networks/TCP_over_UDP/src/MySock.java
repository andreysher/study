import java.net.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.PriorityBlockingQueue;

public class MySock {
/*  при send добавляет данные в буфер для send,
    занимается инициализацией udp socket controller
    в зависимости от текущего порта присваивает этот порт и ip сообщению в буфере
    формирует сообщения для того чтобы положить их в буфер(как)
    буфер - абстракция udp controller
    занимается сегменированием сообщений для помещения в очередь
    udp controller только отправляет!
* */

    public static final int QUEUE_CAPACITY = 20;
    public static final int CONNECT_TIMEOUT = 3000;
    public static final int TRIES = 5;
    public static final int DISCONNECT_TIMEOUT = 5000;

    public PriorityBlockingQueue<MyPack> input;
    public PriorityBlockingQueue<MyPack> output;
    public InetSocketAddress toAddr;

    public DatagramSocket udpSock;
    public int lastPackNumber;//последний отправленный в input
    public int lastReading;//номер последнего взятого из input пакета
    public int lastAckNum;//на какой пакет получен последний ак
    public int acknowledge;//номер последнего принятого нами пакета(подряд)
    public int timeWithoutAnsers;
    public long lastAnswerTime;

    public MySock(int port){

        /*клиентский сокет*/
        this.input = new PriorityBlockingQueue<>(QUEUE_CAPACITY, new MyComporator());
        this.output = new PriorityBlockingQueue<>(QUEUE_CAPACITY, new MyComporator());
        try {
            this.udpSock = new DatagramSocket(port);
        } catch (SocketException e) {
            e.printStackTrace();
        }
        this.lastPackNumber = -1;
        timeWithoutAnsers = 0;
        Thread sender = new Thread(new SenderThread(this));
        Thread receiver = new Thread(new ReceiverThread(this));
        sender.start();
        receiver.start();
    }

    public boolean connect(InetAddress addr, int port){
        this.toAddr = new InetSocketAddress(addr, port);
        MyPack syn = new MyPack(lastPackNumber++,lastAckNum, MyPack.SIN, null);
        output.put(syn);
        long putsTime = System.currentTimeMillis();
        //в ресивере смотрим, если син + ак, то выставить ак намбер и в очередь не добавлять
        while(lastAckNum != 0){//=0 когда пришел ак на синк
            if((System.currentTimeMillis() - putsTime) > DISCONNECT_TIMEOUT){
                return false;
            }
        }
        //sync - нулевой пакет
        //lastAckNum - номер последнего подтвержденного пакета
        //последнего пакета, на который пришел ак
        //значит пришел syn + ack
        MyPack ack = new MyPack(lastPackNumber++, lastAckNum, MyPack.ACK, null);
        output.put(ack);
        putsTime = System.currentTimeMillis();
        while(!output.isEmpty()){
            if((System.currentTimeMillis() - putsTime) > DISCONNECT_TIMEOUT){
                return false;
            }
        }
        return true;
    }

    public int receive(byte[] buffer){
        int howMany = 0;
        while(howMany != buffer.length) {
            MyPack tmp = null;
            try {
                tmp = input.take();
                if(tmp.packNumber != lastReading + 1){
                    output.put(tmp);// положить без расширения, поудмать над тем что будет,
                    // если доложить успеем
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            if (tmp.packNumber == lastReading + 1) {
                for (int i = 0; i < tmp.data.length; i++) {
                    buffer[i + howMany] = tmp.data[i];
                }
                howMany += tmp.data.length;
            }
        }
        return howMany;
    }

    public int send(byte[] buffer){
        /*пилим сообщения на сегменты*/
        int howMany = 0;

        return howMany;
    }

    public void close() throws MyDisconnectException{
        MyPack fin = new MyPack(lastPackNumber++, lastAckNum, MyPack.FIN, null);
        while(!output.isEmpty()){

        }
        output.put(fin);
    }
}
