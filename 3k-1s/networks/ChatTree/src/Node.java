import java.io.IOException;
import java.net.*;
import java.util.*;

public class Node {
    public boolean isRoot = false;
    public HashMap<InetSocketAddress, Integer> children;
    public DatagramSocket mySock;
    public InetSocketAddress parentSock;
    public String myName;
    public int missing;
    public HashMap<UUID, DatagramPacket> massages;
    public LinkedList<UUID> sendingMassages;
    public HashMap<UUID, Long> massangeSendingTime;

    public static final int MAX_WITHOUT_ACK = 10;
    public static final int CONNECT = 1;
    public static final int TEXT = 2;
    public static final int ACK = 3;
    public static final int DISCONNECT = 4;
    public static final int NEW_PARENT = 5;
    public static final int NEW_ROOT = 6;
    public static final long MASSAGE_LIVE_TIME = 1000;
    public static final long MASSAGE_SENDING_TIME = 500;
    //args:
    // 0 - name
    // 1 - missing %
    // 2 - port
    // 3? - parent ip
    // 4? - parent port
    public Node(String[] args) {
        children = new HashMap<>();
        myName = args[0];
        missing = Integer.parseInt(args[1]);
        massages = new HashMap<>();
        sendingMassages = new LinkedList<>();
        massangeSendingTime = new HashMap<>();
//в конструкторах datagram socket не нужны ip и port
        if(args.length == 3){
            isRoot = true;
            try {
                mySock = new DatagramSocket(Integer.parseInt(args[2]));
                parentSock = null;
            } catch (SocketException e) {
                mySock.close();
                e.printStackTrace();
            }
        }
        if(args.length == 5){
            isRoot = false;
            try {
                mySock = new DatagramSocket(Integer.parseInt(args[2]));
                parentSock = new InetSocketAddress(InetAddress.getByName(args[3]), Integer.parseInt(args[4]));
            } catch (SocketException e) {
                mySock.close();
                e.printStackTrace();
            } catch (UnknownHostException e) {
                e.printStackTrace();
            }
            catch (NumberFormatException e){
                System.out.println("incorrect port");
                e.printStackTrace();
            }
            String s = Integer.toString(CONNECT);
            UUID id = UUID.randomUUID();
            s += ";" + id + ";";
            byte[] b = s.getBytes();
            DatagramPacket initPack = new DatagramPacket(b, 0, b.length,
                    parentSock);
            sendingMassages.add(id);
            massages.put(id,initPack);
            massangeSendingTime.put(id, System.currentTimeMillis());
            try {
                mySock.send(initPack);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if((args.length != 5)&&(args.length != 3)){
            System.out.println(args.length);
            System.out.println("wrong quantity of arguments");
        }
    }

    public static void counterIncrement(HashMap<InetSocketAddress, Integer> children, InetSocketAddress to){
        children.put(to, children.get(to) + 1);
    }

    public static void counterZero(HashMap<InetSocketAddress, Integer> children, InetSocketAddress from){
        children.put(from,0);
    }

    public static void main(String[] args) {
        Node me = new Node(args);
        Receiver recv = new Receiver(me);
        recv.start();
        Runtime.getRuntime().addShutdownHook(new ShutdownHook(me, Thread.currentThread(), recv));
        while (true){
            Scanner scan = new Scanner(System.in);
            String str = scan.nextLine();
            System.out.println(me.myName + " : " + str);
            System.out.flush();
            UUID currentTextID = UUID.randomUUID();
            String massage = Integer.toString(TEXT) + ";" + currentTextID
                    + ";" + me.myName + ";" + str + ";";
            //находим позицию разделителя и говорим что дальше идет
            byte[] data = massage.getBytes();
            if(!me.isRoot) {//если мы не рут, то отсылаем родителю
                DatagramPacket pack = new DatagramPacket(data, 0, massage.length(),
                        me.parentSock);
                try {
                    me.massages.put(currentTextID, pack);
                    me.sendingMassages.add(currentTextID);
                    me.massangeSendingTime.put(currentTextID, System.currentTimeMillis());
                    me.mySock.send(pack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            try {
                for (Map.Entry tmp: me.children.entrySet()) {
                    InetSocketAddress tmpAddr = (InetSocketAddress) tmp.getKey();
                    DatagramPacket pack = new DatagramPacket(data, 0, massage.length(), tmpAddr);
                    me.massages.put(currentTextID, pack);
                    me.sendingMassages.add(currentTextID);
                    me.massangeSendingTime.put(currentTextID, System.currentTimeMillis());
                    counterIncrement(me.children, tmpAddr);
                    me.mySock.send(pack);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            if(Thread.currentThread().isInterrupted()){
                return;
            }
        }
    }
}
