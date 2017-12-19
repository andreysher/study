import java.net.SocketException;

public class ServerSocket {
    private Background background;

    public ServerSocket(int port) throws TCPSocketException {
        try {
            background = new Background(port);
        } catch (SocketException ex) {
            throw new TCPSocketException(ex.getMessage());
        }
    }

    public Socket accept() throws TCPSocketException { //блок-ся на ассет
        return background.getAccepted();
    }

    public void close() {
        background.stop(null);
    }
}
