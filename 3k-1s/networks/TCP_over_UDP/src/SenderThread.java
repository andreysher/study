import java.io.IOException;
import java.net.DatagramPacket;
import java.net.InetSocketAddress;
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.concurrent.PriorityBlockingQueue;

public class SenderThread implements Runnable {
    /*идет в output достает оттуда пакеты и отправляет в toaddr*/
    private MySock sock;
    private PriorityQueue<MyPack> sending;

    public SenderThread(MySock sock){
        this.sock = sock;
    }

    public void sendMyPack(MyPack pkg){
        byte[] buf = pkg.getData();
        DatagramPacket pack = new DatagramPacket(buf, 0, buf.length, sock.toAddr);
        try {
            sock.udpSock.send(pack);
        } catch (IOException e) {
            e.printStackTrace();
        }
        sending = new PriorityQueue<>(0, new MyComporator());
    }

    @Override
    public void run() {
        while (true) {
            try {
                MyPack current = sock.output.take();
                sending.add(current);
                Iterator itr = sending.iterator();
                int i = 0;
                while(i != MySock.TRIES){
                    if(itr.hasNext()){
                        MyPack tmp = (MyPack) itr.next();
                        if(tmp.ackNumber <= sock.lastAckNum) {
                            itr.remove();
                            continue;
                        }
                        sendMyPack(tmp);
                    }
                    else {
                        itr = sending.iterator();
                        i++;
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
