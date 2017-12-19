import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;

public class Server {
    public static void main(String[] args) {
        try {
            ServerSocket serverSocket = new ServerSocket(Integer.parseInt(args[0]));
            Socket socket = serverSocket.accept();

            byte[] data = new byte[8];
            int count = 0;

            while (count < 8){
                count += socket.recv(data, count);
            }

            long fileSize = ByteBuffer.wrap(data).getLong();//long c размером файла
            String path = new File("3.mp3").getPath();
            RandomAccessFile file = new RandomAccessFile(path, "rw");
            byte[] fileData = new byte[Constants.BUF_SZ];
            count = 0;
            int totalsize = 0;
            while(totalsize != fileSize) {
                count += socket.recv(fileData, count);//в ресиве count это offset
                //считаем хотябы что-нибудь(если читать нечего, то заблокируемся)
                file.write(fileData, 0, count);
                totalsize += count;
                count = 0;
            }

            System.out.println(totalsize);
            file.close();
            socket.close();
            serverSocket.close();
        } catch (TCPSocketException ex) {
            System.out.println(ex.getMessage());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
