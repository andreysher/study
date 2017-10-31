import java.io.IOException;
import java.net.*;
import java.util.*;

public class ShutdownHook extends Thread {
    Node me;
    public ShutdownHook(Node me){
        this.me = me;
    }

    @Override
    public void run() {
        if(!me.isRoot){//отправили disconnect родителю
            UUID did = UUID.randomUUID();
            String discon = Integer.toString(Node.DISCONNECT) + ";" + did + ";";
            byte[] d = discon.getBytes();
            DatagramPacket disPack = new DatagramPacket(d, 0, d.length, me.parentSock);
            try {
                me.massages.put(did, disPack);
                me.sendingMassages.add(did);
                me.massangeSendingTime.put(did, System.currentTimeMillis());
                me.mySock.send(disPack);
            } catch (IOException e) {
                e.printStackTrace();
            }
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
                    me.mySock.send(newParentPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        else {
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
                    me.mySock.send(NewRootPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                InetAddress newRootAddress = tmpAddr.getAddress();
                int newRootPort = tmpAddr.getPort();
                itr.remove();
                for (Map.Entry t:me.children.entrySet()) {
                    InetSocketAddress ta = (InetSocketAddress) t.getKey();
                    UUID newParentID = UUID.randomUUID();
                    String newParent = Integer.toString(Node.NEW_PARENT) + ";" + newParentID + ";" +
                            newRootAddress + ";" + newRootPort + ";";
                    byte[] npp = newParent.getBytes();
                    DatagramPacket newParentPack = new DatagramPacket(npp, 0, npp.length, ta);
                    try {
                        me.massages.put(newParentID, newParentPack);
                        me.sendingMassages.add(newParentID);
                        me.massangeSendingTime.put(newParentID,System.currentTimeMillis());
                        me.mySock.send(newParentPack);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            else {
                System.out.println("last node is shouting down");
            }
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
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
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
    }
}
