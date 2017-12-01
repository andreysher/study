import java.net.DatagramSocket;
import java.net.SocketException;
import java.util.concurrent.PriorityBlockingQueue;

public class UdpController {
/* Гарантирует что на сервер будет только 1 udp сокет*/
    public static final int DEFAULT_PORT = 8080;

    private DatagramSocket sock;
    private int port;
    public PriorityBlockingQueue<MyPack> input;
    public PriorityBlockingQueue<MyPack> output;

    private UdpController(){
        this.port = DEFAULT_PORT;
        try {
            sock = new DatagramSocket(port);
        } catch (SocketException e) {
            e.printStackTrace();
        }
    }

    private static class SingletonHelper{
        private static final UdpController uc = new UdpController();
    }

    public static UdpController getInstance(PriorityBlockingQueue<MyPack> in,
                                            PriorityBlockingQueue<MyPack> out){

        return SingletonHelper.uc;
    }


}
