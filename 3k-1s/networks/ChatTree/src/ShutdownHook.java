import java.io.IOException;
<<<<<<< HEAD
import java.net.*;
import java.util.*;
=======
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.Iterator;
import java.util.UUID;
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b

public class ShutdownHook extends Thread {
    Node me;
    public ShutdownHook(Node me){
        this.me = me;
    }

    @Override
    public void run() {
<<<<<<< HEAD
        if(!me.isRoot){//отправили disconnect родителю
=======
        UUID id = UUID.randomUUID();
        if(!me.isRoot){
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
            UUID did = UUID.randomUUID();
            String discon = Integer.toString(Node.DISCONNECT) + ";" + did + ";";
            byte[] d = discon.getBytes();
            DatagramPacket disPack = new DatagramPacket(d, 0, d.length, me.parentSock);
<<<<<<< HEAD
            try {
                me.massages.put(did, disPack);
                me.sendingMassages.add(did);
                me.massangeSendingTime.put(did, System.currentTimeMillis());
=======
            String newParent = Integer.toString(Node.NEW_PARENT) + ";" + id + ";" + me.parentSock.getAddress()
                + ";" + Integer.toString(me.parentSock.getPort()) + ";";
            byte[] np = newParent.getBytes();
            try {
                me.massages.put(did, disPack);
                me.sendingMassages.put(did, me.parentSock);
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                me.mySock.send(disPack);
            } catch (IOException e) {
                e.printStackTrace();
            }
<<<<<<< HEAD
            for (Map.Entry tmp: me.children.entrySet()) {
                InetSocketAddress tmpAddr = (InetSocketAddress) tmp.getKey();
                try {
                    UUID id = UUID.randomUUID();
                    String newParent = Integer.toString(Node.NEW_PARENT) + ";" + id + ";" + me.parentSock.getAddress()
                            + ";" + Integer.toString(me.parentSock.getPort()) + ";";
                    byte[] np = newParent.getBytes();
                    DatagramPacket newParentPack = new DatagramPacket(np, 0, np.length, tmpAddr);
                    me.massages.put(id, newParentPack);
                    me.sendingMassages.add(id);
                    me.massangeSendingTime.put(id, System.currentTimeMillis());
=======
            for (InetSocketAddress tmp: me.children) {
                try {
                    DatagramPacket newParentPack = new DatagramPacket(np, 0, np.length, tmp);
                    me.massages.put(id, newParentPack);
                    me.sendingMassages.put(id, tmp);
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                    me.mySock.send(newParentPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        else {
<<<<<<< HEAD
            Iterator itr = me.children.entrySet().iterator();
            if(itr.hasNext()){
                Map.Entry tmp = (Map.Entry) itr.next();
                InetSocketAddress tmpAddr = (InetSocketAddress) tmp.getKey();
                UUID id = UUID.randomUUID();
                String newRoot = Integer.toString(Node.NEW_ROOT) + ";" + id + ";";
                byte[] nr = newRoot.getBytes();
                DatagramPacket NewRootPack = new DatagramPacket(nr, 0, nr.length, tmpAddr);
                try {
                    me.massages.put(id, NewRootPack);
                    me.sendingMassages.add(id);
                    me.massangeSendingTime.put(id,System.currentTimeMillis());
=======
            Iterator itr = me.children.iterator();
            if(itr.hasNext()){
                InetSocketAddress tmp = (InetSocketAddress) itr.next();
                String newRoot = Integer.toString(Node.NEW_ROOT) + ";" + id + ";";
                byte[] nr = newRoot.getBytes();
                DatagramPacket NewRootPack = new DatagramPacket(nr, 0, nr.length, tmp);
                try {
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                    me.mySock.send(NewRootPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
<<<<<<< HEAD
                InetAddress newRootAddress = tmpAddr.getAddress();
                int newRootPort = tmpAddr.getPort();
                itr.remove();
                for (Map.Entry t:me.children.entrySet()) {
                    InetSocketAddress ta = (InetSocketAddress) t.getKey();
=======
                InetAddress newRootAddress = tmp.getAddress();
                int newRootPort = tmp.getPort();
                itr.remove();
                for (InetSocketAddress t:me.children) {
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                    UUID newParentID = UUID.randomUUID();
                    String newParent = Integer.toString(Node.NEW_PARENT) + ";" + newParentID + ";" +
                            newRootAddress + ";" + newRootPort + ";";
                    byte[] npp = newParent.getBytes();
<<<<<<< HEAD
                    DatagramPacket newParentPack = new DatagramPacket(npp, 0, npp.length, ta);
                    try {
                        me.massages.put(newParentID, newParentPack);
                        me.sendingMassages.add(newParentID);
                        me.massangeSendingTime.put(newParentID,System.currentTimeMillis());
=======
                    DatagramPacket newParentPack = new DatagramPacket(npp, 0, npp.length, t);
                    try {
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                        me.mySock.send(newParentPack);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            else {
                System.out.println("last node is shouting down");
            }
<<<<<<< HEAD
        }
        Iterator itr = me.massangeSendingTime.entrySet().iterator();
        while(!me.massangeSendingTime.isEmpty()) {
            if (itr.hasNext()) {
                Map.Entry tmp = (Map.Entry) itr.next();
                if (((System.currentTimeMillis() - (long) tmp.getValue()) >= Node.MASSAGE_SENDING_TIME)
                        && (me.sendingMassages.contains(tmp.getKey()))) {
                    try {
                        me.mySock.send(me.massages.get(tmp.getKey()));
                        tmp.setValue(Long.parseLong("0"));
=======
        }{
            Iterator itr = me.children.iterator();
            if(itr.hasNext()){
                InetSocketAddress tmp = (InetSocketAddress) itr.next();
                String newRoot = Integer.toString(Node.NEW_ROOT) + ";" + id + ";";
                byte[] nr = newRoot.getBytes();
                DatagramPacket NewRootPack = new DatagramPacket(nr, nr.length);
                InetAddress newRootAddress = tmp.getAddress();
                int newRootPort = tmp.getPort();
                itr.remove();
                for (InetSocketAddress t:me.children) {
                    UUID newParentID = UUID.randomUUID();
                    String newParent = Integer.toString(Node.NEW_PARENT) + ";" + newParentID + ";" +
                            newRootAddress + ";" + newRootPort + ";";
                    byte[] npp = newParent.getBytes();
                    DatagramPacket newParentPack = new DatagramPacket(npp, 0, npp.length, t);
                    try {
                        me.mySock.send(newParentPack);
>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
<<<<<<< HEAD
                if ((System.currentTimeMillis() - (long) tmp.getValue()) >= Node.MASSAGE_LIVE_TIME) {
                    if(itr.hasNext()){
                        itr.remove();
                    }
                    me.sendingMassages.remove(tmp.getKey());
                    me.massages.remove(tmp.getKey());
                }
            }
            else{
                itr = me.massangeSendingTime.entrySet().iterator();
            }
        }
=======
            }
            else {
                System.out.println("last node is shouting down");
            }
        }

>>>>>>> 9976413696d8404afd779153b3c777cf3501007b
    }
}
