import java.io.IOException;
import java.net.*;
import java.util.*;

public class Receiver extends Thread {
    public static final int MAX_MASSAGE_LEN = 100;
    public DatagramPacket pack;
    public Node me;

    public Receiver(Node me){
        this.me = me;
    }
    // поля нумеруются с 0, везде в конце ; и не юзаем ; в именах и сообщениях
    public String getField(String massage, int fieldNumber){
        int curField = 0;
        String tmp = massage;
        int end = massage.indexOf(';');
        String rez = tmp.substring(0,end);
        while(curField != fieldNumber){
            curField++;
            tmp = tmp.substring(end+1);
            end = tmp.indexOf(';');
            rez = tmp.substring(0, end);
        }
        return rez;
    }

    public void run() {
        Random rnd = new Random();
        byte[] b;
        int type;
        while (true){
            try {
                if(me.missing < rnd.nextInt(99)){}
                else {
                    b = new byte[MAX_MASSAGE_LEN];
                    pack = new DatagramPacket(b, b.length);
                    me.mySock.receive(pack);
                    String massage = (new String(pack.getData())).substring(0, pack.getLength());
//                    System.out.println("geting" + massage);
                    type = Integer.parseInt(getField(massage, 0));
                    String uuid = getField(massage, 1);

                    InetSocketAddress sender = new InetSocketAddress(pack.getAddress(), pack.getPort());
                    String ac = Integer.toString(Node.ACK) + ';' + uuid + ";";
                    byte[] a = ac.getBytes();
                    DatagramPacket ack = new DatagramPacket(a, 0,  a.length, sender);
                    UUID uid = UUID.fromString(getField(massage, 1));
                    if(!me.massages.containsKey(uid)) {
                        switch (type) {
                            case Node.CONNECT:
                                //игнорить то что есть в massages
                                me.massages.put(uid, pack);
                                me.children.add(sender);
                                System.out.println("sending ack");
                                me.mySock.send(ack);
                                break;

                            case Node.TEXT:
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                System.out.println(getField(massage, 2) + " : " +
                                        getField(massage, 3));
                                if (!me.isRoot) {
                                    me.sendingMassages.put(UUID.fromString(getField(massage, 1)),
                                            me.parentSock);
                                }
                                for (InetSocketAddress tmp : me.children) {
                                    DatagramPacket childPack = new DatagramPacket(massage.getBytes(), 0,
                                            massage.length(), tmp);
                                    me.sendingMassages.put(UUID.fromString(getField(massage, 1)), tmp);
                                    me.mySock.send(childPack);
                                }
                                me.mySock.send(ack);
                                break;
                            case Node.ACK:
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                me.sendingMassages.remove(UUID.fromString(getField(massage, 1)));
                                break;
                            case Node.DISCONNECT:
                                System.out.println("disconnect");
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                Iterator iter = me.children.iterator();
                                while (iter.hasNext()) {
                                    InetSocketAddress tmp = (InetSocketAddress) iter.next();
                                    if ((tmp == pack.getSocketAddress())) {
                                        iter.remove();
                                    }
                                }
                                me.mySock.send(ack);
                                break;
                            case Node.NEW_PARENT:
                                System.out.println("new parent");
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                me.parentSock = new InetSocketAddress(InetAddress.getByName(getField(massage, 2))
                                        , Integer.parseInt(getField(massage, 3)));
                                UUID cid = UUID.randomUUID();
                                String connect = Integer.toString(Node.CONNECT) + ";" + cid + ";";
                                byte[] conb = connect.getBytes();
                                DatagramPacket connPack = new DatagramPacket(conb, conb.length);
                                me.sendingMassages.put(cid, me.parentSock);
                                me.massages.put(cid, connPack);
                                me.mySock.send(connPack);
                                me.mySock.send(ack);
                                break;
                            case Node.NEW_ROOT:
                                System.out.println("new root");
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                me.isRoot = true;
                                me.parentSock = null;
                                me.mySock.send(ack);
                                break;
                        }
                    }
//                    Iterator itr = me.massangeSendingTime.entrySet().iterator();
//                    while(itr.hasNext()){
//                        Map.Entry tmp = (Map.Entry) itr.next();
//                        if(((System.currentTimeMillis() - (long)tmp.getValue()) >= Node.MASSAGE_SENDING_TIME)
//                            &&(me.sendingMassages.containsKey(tmp.getKey()))){
//                            DatagramSocket sendSock = new DatagramSocket(me.sendingMassages.get(tmp.getKey()));
//                            sendSock.send(me.massages.get(tmp.getKey()));
//                        }
//                        if((System.currentTimeMillis() - (long)tmp.getValue()) >= Node.MASSAGE_LIVE_TIME){
//                            itr.remove();
//                            me.sendingMassages.remove(tmp.getKey());
//                            me.massages.remove(tmp.getKey());
//                        }
//                    }
                }
            } catch (SocketException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
