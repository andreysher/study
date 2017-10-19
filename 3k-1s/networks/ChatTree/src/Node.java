import java.net.*;
import java.util.HashMap;
import java.util.Map;

public class Node {
    private boolean isRoot = false;
    private Map<String, DatagramSocket> children;
    private DatagramSocket mySock;
    private DatagramSocket parentSock;
    private String myName;
    private int missing;
    //args:
    // 0 - name
    // 1 - missing %
    // 2 - port
    // 3? - parent ip
    // 4? - parent port
    public Node(String[] args, boolean rootExist) throws MoreOneRootException {
        if((args.length == 3)&&(rootExist)) {
            throw new MoreOneRootException();
        }
        if((args.length == 3)&&(!rootExist)){
            isRoot = true;
            children = new HashMap<>();
            myName = args[0];
            missing = Integer.parseInt(args[1]);
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
            myName = args[0];
            children = new HashMap<>();
            missing = Integer.parseInt(args[1]);
            try {
                mySock = new DatagramSocket(Integer.parseInt(args[2]));
                parentSock = new DatagramSocket(Integer.parseInt(args[4]), InetAddress.getByName(args[3]));
            } catch (SocketException e) {
                mySock.close();
                e.printStackTrace();
            } catch (UnknownHostException e) {
                e.printStackTrace();
            }
        }
        else{
            System.out.println("wrong quantity of arguments");
        }
    }
}
