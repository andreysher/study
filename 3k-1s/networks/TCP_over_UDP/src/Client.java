import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;

public class Client {

    public static void main(String[] args) {
        try {
            Socket socket = new Socket(args[0], Integer.parseInt(args[1]));
            String path = new File(args[2]).getAbsolutePath();

            try (RandomAccessFile file = new RandomAccessFile(path, "r")) {
                byte[] fileData = new byte[Constants.BUF_SZ];
                int rc = file.read(fileData);
                if(rc == 0){
                    System.out.println("file is empty!");
                    return;
                }
                if(rc == -1){
                    System.out.println("file does not exist!");
                    return;
                }
                socket.send(ByteBuffer.allocate(8).putLong(file.length()).array(), 8);
                while(rc != -1){
                    socket.send(fileData, rc);
                    rc = file.read(fileData);
                }
            } catch (IOException ex) {
                System.out.println(ex.getMessage());
            } finally {
                socket.close();
            }
        } catch (TCPSocketException ex) {
            System.out.println(ex.getMessage());
        }
    }
}
