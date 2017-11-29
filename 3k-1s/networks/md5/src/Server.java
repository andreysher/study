import java.io.*;
import java.net.ServerSocket;
import java.util.*;

public class Server extends Thread {

    private List<int[]> tasks;
    private int port;
    private String hashString;
    private static final int MAX_LENGTH = 4;
    private static final long COUNTING_TIMEOUT = 10000;
    private List<ControlledTask> executingTasks;

    private Server(int port, String hashString){
        this.hashString = hashString;
        this.port = port;
        executingTasks = Collections.synchronizedList(new LinkedList<ControlledTask>());
        tasks = Collections.synchronizedList(new LinkedList<int[]>());
        //generate tasks
        //таска - 2 лонга: 1-сам префикс, 2-его длина.

        for (int currentLength = 1; currentLength <= MAX_LENGTH - Client.CLIENT_LENGTH; currentLength++){
            int max = (int) Math.pow(Client.GENES.length, currentLength);
            int number = 0;
            while(number < max){
                tasks.add(new int[]{number, currentLength});
                number++;
            }
        }

//        for (int[] t:tasks
//             ) {
//            System.out.println(Client.codeToString(t[0], t[1]));
//        }
        /*задаем константу client_length. Когда выдаем префикс однобуквенный клиент генерит все возможные последовательности длины до
        * clientlenth включая, нольбуквенные последовательности не генерим на сервере, а на клиенте генерим начиная с пустой
        * клиент ленф на всех клиентах одинаковый. Когда последовательность от сервера не однобуквенная, генерим последовательности длины
        * только равной clientlength*/

    }

    @Override
    public void run(){
        List<String> results = new LinkedList<>();

        try(ServerSocket socket = new ServerSocket(port)){
            Accepter accepter = new Accepter(socket, hashString, tasks, results, executingTasks);
            accepter.start();
            while(results.isEmpty()){
                //TODO: когда получаем
                sleep(COUNTING_TIMEOUT);
                synchronized (executingTasks) {
                    Iterator iterator = executingTasks.iterator();
                    while (iterator.hasNext()) {
                        ControlledTask tmp = (ControlledTask) iterator.next();
                        if (System.currentTimeMillis() - tmp.time > COUNTING_TIMEOUT) {
                            tasks.add(tmp.task);
                            iterator.remove();
                        }
                    }
                }
            }
            String res = results.remove(0);
            accepter.interrupt();
            System.out.println("Is " + res + " your string?");
        } catch (IOException | InterruptedException e) {
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
