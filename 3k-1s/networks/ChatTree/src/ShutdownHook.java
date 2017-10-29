import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.util.Iterator;
import java.util.UUID;

public class ShutdownHook extends Thread {
    Node me;
    public ShutdownHook(Node me){
        this.me = me;
    }

    @Override
    public void run() {
        UUID id = UUID.randomUUID();
        if(!me.isRoot){
            UUID did = UUID.randomUUID();
            String discon = Integer.toString(Node.DISCONNECT) + ";" + did + ";";
            byte[] d = discon.getBytes();
            DatagramPacket disPack = new DatagramPacket(d, 0, d.length, me.parentSock);
            String newParent = Integer.toString(Node.NEW_PARENT) + ";" + id + ";" + me.parentSock.getAddress()
                + ";" + Integer.toString(me.parentSock.getPort()) + ";";
            byte[] np = newParent.getBytes();
            try {
                me.massages.put(did, disPack);
                me.sendingMassages.put(did, me.parentSock);
                me.mySock.send(disPack);
            } catch (IOException e) {
                e.printStackTrace();
            }
            for (InetSocketAddress tmp: me.children) {
                try {
                    DatagramPacket newParentPack = new DatagramPacket(np, 0, np.length, tmp);
                    me.massages.put(id, newParentPack);
                    me.sendingMassages.put(id, tmp);
                    me.mySock.send(newParentPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        else {
            Iterator itr = me.children.iterator();
            if(itr.hasNext()){
                InetSocketAddress tmp = (InetSocketAddress) itr.next();
                String newRoot = Integer.toString(Node.NEW_ROOT) + ";" + id + ";";
                byte[] nr = newRoot.getBytes();
                DatagramPacket NewRootPack = new DatagramPacket(nr, 0, nr.length, tmp);
                try {
                    me.mySock.send(NewRootPack);
                } catch (IOException e) {
                    e.printStackTrace();
                }
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
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            else {
                System.out.println("last node is shouting down");
            }
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
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            else {
                System.out.println("last node is shouting down");
            }
        }

    }
}
