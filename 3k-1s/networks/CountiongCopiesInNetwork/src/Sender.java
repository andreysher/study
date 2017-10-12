import java.io.IOException;
import java.net.*;

public class Sender implements Runnable {
    public String groupIP;
    public Sender(String multicastIP){
        this.groupIP = multicastIP;
    }

    public void run(){
        int port = 2049;
        try {
            DatagramSocket senderSocket = new DatagramSocket();//не указываем порт
            senderSocket.setReuseAddress(true);
            byte[] sendbuf = new byte[100];
            InetAddress groupAddr = InetAddress.getByName(groupIP);
            DatagramPacket sendingPack = new DatagramPacket(sendbuf, sendbuf.length,
                    groupAddr, port);
            while(true) {
                senderSocket.send(sendingPack);
                Thread.sleep(300);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
