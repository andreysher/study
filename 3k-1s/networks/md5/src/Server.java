import java.io.*;
import java.net.ServerSocket;
import java.util.*;

public class Server extends Thread {

    private List<int[]> tasks;
    private int port;
    private String hashString;
    public int MAX_LENGTH = 16;

    public Server(int port, String hashString){
        this.hashString = hashString;
        this.port = port;

        tasks = Collections.synchronizedList(new LinkedList<int[]>());
        //generate tasks
        //таска - 2 лонга: 1-сам префикс, 2-его длина.

        for (int currentLength = 1; currentLength < MAX_LENGTH - Client.CLIENT_LENGTH; currentLength++){
            int max = (int) Math.pow(Client.GENES.length, currentLength);
            int number = 0;
            while(number < max){
                tasks.add(new int[]{number, currentLength});
                number++;
            }
        }

        /*задаем константу client_length. Когда выдаем префикс однобуквенный клиент генерит все возможные последовательности длины до
        * clientlenth включая, нольбуквенные последовательности не генерим на сервере, а на клиенте генерим начиная с пустой
        * клиент ленф на всех клиентах одинаковый. Когда последовательность от сервера не однобуквенная, генерим последовательности длины
        * только равной clientlength*/

    }

    @Override
    public void run(){
        List<String> results = Collections.synchronizedList(new LinkedList<String>());

        try(ServerSocket socket = new ServerSocket(port)){
            Accepter accepter = new Accepter(socket, hashString, tasks, results);
            accepter.start();
            
//            //test
//            for (int[] tmp:tasks
//                 ) {
//                System.out.println(Client.codeToString(tmp[0] ,tmp[1]));
//            }
            while(results.isEmpty()){
                try{
                    sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            accepter.interrupt();
            System.out.println("Is " + results.remove(0) + " your string?");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {

        Server server = new Server(Integer.parseInt(args[1]), args[0]);
        server.start();

        try {
            server.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
