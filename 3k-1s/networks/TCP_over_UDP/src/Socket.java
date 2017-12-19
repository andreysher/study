import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.SocketException;

public class Socket {
    SocketAddress address;
    private Buffer buf;
    private Background background;

    Socket(SocketAddress address, Buffer buf, Background background) {
        this.address = address;
        this.buf = buf;
        this.background = background;
    }

    public Socket(String ip, int port) throws TCPSocketException {
        InetSocketAddress addr = new InetSocketAddress(ip, port);
        address = addr;
        buf = new Buffer();

        try {
            background = new Background(addr, buf);
        } catch (SocketException ex) {
            throw new TCPSocketException(ex.getMessage());
        }
    }

    public void send(byte[] buffer, int len) {
        buf.putUserPayload(buffer, len);
    }
    public int recv(byte[] buffer, int offset) {
        return buf.getUserPayload(buffer, offset);
    }
    public void close() {
        background.stop(address);
    }
}
