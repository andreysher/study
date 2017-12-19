import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SocketProxy {

    private final DatagramSocket socket;

    private final Random random = new Random();
    private final List<DatagramPacket> packets = new ArrayList<>();

    public SocketProxy(DatagramSocket socket) {
        this.socket = socket;
    }

    public void put(DatagramPacket packet) {
        if (Math.random() > 0.4)
            return;
        packets.add(packet);
    }

    public DatagramPacket get() {
        if (!isReady())
            return null;
        return packets.get(random.nextInt(packets.size()));
    }

    public boolean isReady() {
        return packets.size() >= 3;
    }
}
