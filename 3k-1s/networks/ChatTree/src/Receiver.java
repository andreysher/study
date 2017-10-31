import java.io.IOException;
import java.net.*;
import java.util.*;

public class Receiver extends Thread {
<<<<<<< HEAD
    public static final int MAX_MASSAGE_LEN = 1000;
=======
    public static final int MAX_MASSAGE_LEN = 100;
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
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

<<<<<<< HEAD
    public static void send(Node me, DatagramPacket pack, InetSocketAddress target){
        DatagramPacket pkg = new DatagramPacket(pack.getData(), 0, pack.getData().length, target);
        UUID uid = UUID.randomUUID();
        me.massages.put(uid, pkg);
        me.sendingMassages.add(uid);
        me.massangeSendingTime.put(uid, System.currentTimeMillis());
        try {
            if(me.children.containsKey(target)){
                Node.counterIncrement(me.children, target);
            }
            me.mySock.send(pkg);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

=======
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
    public void run() {
        Random rnd = new Random();
        byte[] b;
        int type;
        while (true){
            try {
<<<<<<< HEAD

                if(me.missing > rnd.nextInt(99)){}
=======
                if(me.missing < rnd.nextInt(99)){}
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                else {
                    b = new byte[MAX_MASSAGE_LEN];
                    pack = new DatagramPacket(b, b.length);
                    me.mySock.receive(pack);
                    String massage = (new String(pack.getData())).substring(0, pack.getLength());
<<<<<<< HEAD
                    type = Integer.parseInt(getField(massage, 0));
                    String uuid = getField(massage, 1);
                    InetSocketAddress sender = new InetSocketAddress(pack.getAddress(), pack.getPort());
                    if(me.children.containsKey(sender)){
                        Node.counterZero(me.children, sender);
                    }
                    String ac = Integer.toString(Node.ACK) + ';' + uuid + ";";
                    byte[] a = ac.getBytes();
                    DatagramPacket ack = new DatagramPacket(a, 0,  a.length, sender);
                    UUID uid = UUID.fromString(uuid);

                    if(!me.massages.containsKey(uid)) {
                        switch (type) {
                            case Node.CONNECT:
                                System.out.println("connect");
                                //игнорить то что есть в massages
                                me.massages.put(uid, pack);//чтобы больше не принимать
                                me.children.put(sender, 0);
                                Node.counterIncrement(me.children, sender);
=======
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
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                                me.mySock.send(ack);
                                break;

                            case Node.TEXT:
<<<<<<< HEAD
//                                System.out.println("text");
                                me.massages.put(uid, pack);
                                System.out.println(getField(massage, 2) + " : " +
                                        getField(massage, 3));
                                System.out.flush();
                                if ((!me.isRoot) && (!sender.equals(me.parentSock))){
                                    send(me, pack, me.parentSock);
                                }
                                for (Map.Entry tmp : me.children.entrySet()) {
                                    InetSocketAddress tmpAddr = (InetSocketAddress) tmp.getKey();
                                    if(!sender.equals(tmpAddr)){
                                        send(me,pack,tmpAddr);
                                    }
                                }
                                me.mySock.send(ack);
                                break;

                            case Node.ACK:
                                System.out.println("ack");
                                if(me.massages.containsKey(uid)){
                                    me.massages.remove(uid);
                                }
                                if(me.sendingMassages.contains(uid)){
                                    me.sendingMassages.remove(uid);
                                }
                                if(me.massangeSendingTime.containsKey(uid)){
                                    me.massangeSendingTime.remove(uid);
                                }
                                break;

                            case Node.DISCONNECT:
                                System.out.println("disconnect");
                                me.massages.put(uid, pack);
                                Iterator iter = me.children.entrySet().iterator();
                                while (iter.hasNext()) {
                                    Map.Entry tmp = (Map.Entry) iter.next();
                                    InetSocketAddress tmpAddr = (InetSocketAddress) tmp.getKey();
                                    if (tmpAddr == pack.getSocketAddress()) {
=======
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
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                                        iter.remove();
                                    }
                                }
                                me.mySock.send(ack);
                                break;
<<<<<<< HEAD

                            case Node.NEW_PARENT:
                                try {
                                    System.out.println("new parent");
                                    me.massages.put(uid, pack);
                                    String addrName = getField(massage, 2);
                                    addrName = addrName.substring(1, addrName.length());
                                    InetAddress newAddr = InetAddress.getByName(addrName);
                                    int newPort = Integer.parseInt(getField(massage, 3));
                                    me.parentSock = new InetSocketAddress(newAddr, newPort);
                                    UUID cid = UUID.randomUUID();
                                    String connect = Integer.toString(Node.CONNECT) + ";" + cid + ";";
                                    byte[] conb = connect.getBytes();
                                    DatagramPacket connPack = new DatagramPacket(conb, 0, conb.length, me.parentSock);
                                    me.sendingMassages.add(cid);
                                    me.massages.put(cid, connPack);
                                    me.massangeSendingTime.put(cid, System.currentTimeMillis());
                                    me.mySock.send(connPack);
                                    me.mySock.send(ack);
                                    break;
                                }
                                catch (UnknownHostException e){
                                    e.printStackTrace();
                                }

=======
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
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                            case Node.NEW_ROOT:
                                System.out.println("new root");
                                me.massages.put(UUID.fromString(getField(massage, 1)), pack);
                                me.isRoot = true;
                                me.parentSock = null;
                                me.mySock.send(ack);
                                break;
                        }
                    }
<<<<<<< HEAD
                    else {
                        //send ack on "uid" если шлем ack постоянно, то постоянно его обрабатываем... и входим в цикл
                        if(!(type == Node.ACK)) {
//                            System.out.println(type);
                            me.mySock.send(ack);
                        }
                    }
                    Iterator itr = me.massangeSendingTime.entrySet().iterator();
                    while(itr.hasNext()){
                        Map.Entry tmp = (Map.Entry) itr.next();
                        if(((System.currentTimeMillis() - (long)tmp.getValue()) >= Node.MASSAGE_SENDING_TIME)
                            &&(me.sendingMassages.contains(tmp.getKey()))){
                            me.mySock.send(me.massages.get(tmp.getKey()));
                            tmp.setValue(Long.parseLong("0"));
                        }
                        if((System.currentTimeMillis() - (long)tmp.getValue()) >= Node.MASSAGE_LIVE_TIME){
                            itr.remove();
                            me.sendingMassages.remove(tmp.getKey());
                            me.massages.remove(tmp.getKey());
                        }
                    }
=======
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
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                }
            } catch (SocketException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
