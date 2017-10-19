import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;

import static java.lang.Integer.parseInt;

public class Server {
    public static void main(String[] args) {
        try (ServerSocket socket = new ServerSocket(parseInt(args[0]))) {
            System.out.println("Server started");
            File test = new File("./");
            if(test.canWrite()) {
                File dir = new File("./uploads/");
                dir.mkdir();
            }

            // TODO: addShutdownHook(done in class serverThread!)

            while (true){
                try {
                    new Thread(new ServerThread(socket.accept())).start();
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}