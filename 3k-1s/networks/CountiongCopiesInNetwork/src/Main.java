import java.io.IOException;
import java.net.*;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        int port = 2049;

        long dieTimeout = 3000;
        try {
            MulticastSocket recvSocket = new MulticastSocket(port); //datagram/multicast
            recvSocket.setReuseAddress(true);
            byte[] buf = new byte[100];
            DatagramPacket incoming = new DatagramPacket(buf, buf.length);
            Map<String, Long> adrTimeMap = new HashMap<>();
            Sender sendThread = new Sender(args[0]);
            new Thread(sendThread).start();
            recvSocket.joinGroup(InetAddress.getByName(args[0]));
            while(true) {
                Iterator itr = adrTimeMap.entrySet().iterator();
                while(itr.hasNext()){
                    Date moment = new Date();
                    Map.Entry tmp = (Map.Entry) itr.next();
                    if((long)tmp.getValue() < (moment.getTime() - dieTimeout)){
                        System.out.println("one copy was killed");
                        itr.remove();
                        for (String temp:adrTimeMap.keySet()
                                ) {
                            System.out.println(temp);
                        }
                    }
                }
                recvSocket.receive(incoming);
                SocketAddress addr = incoming.getSocketAddress();
                if(adrTimeMap.get(addr.toString()) == null){
                    Date moment1 = new Date();
                    adrTimeMap.put(addr.toString(), moment1.getTime());
                    for (String temp:adrTimeMap.keySet()
                            ) {
                        System.out.println(temp);
                    }
                }
                Date mom = new Date();
                adrTimeMap.put(addr.toString(),mom.getTime());
            }
        } catch (SocketException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}

