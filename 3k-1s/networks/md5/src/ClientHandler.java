import java.io.*;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.util.List;

public class ClientHandler extends Thread {

    private Socket socket;
    private String hashString;
    private List<int[]> tasks;
    private List<String> results;
    private int id;
    private InetSocketAddress clientAddress;

    public ClientHandler(Socket socket, String hashString, List<int[]> tasks,
                         List<String> results, int id){
        this.socket = socket;
        this.hashString = hashString;
        this.tasks = tasks;
        this.results = results;
        this.id = id;
        this.clientAddress = (InetSocketAddress) socket.getRemoteSocketAddress();
    }

    @Override
    public void run(){
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                PrintWriter writer = new PrintWriter(socket.getOutputStream())){

            writer.println(hashString);
            writer.flush();

            while(true){
                int[] task;

                synchronized (tasks){
                    if(tasks.isEmpty()){
                        task = null;
                    }
                    else {

                        task = tasks.remove(0);
                    }
                }

                if(task == null){
                    System.out.println("Client send stop " + id);

                    writer.println("STOP");
                    writer.flush();

                    socket.close();

                    return;
                }
                else {
                    writer.println("WORK");
                    writer.println(task[0]);
                    writer.println(task[1]);
                    writer.flush();

                    System.out.println("Task sent");

                    switch (reader.readLine()){
                        case "SUCCESS":
                            String result = reader.readLine();
                            writer.println("STOP");
                            tasks.clear();
                            results.add(result);
                            socket.close();
                            return;
                        case "UNSUCCESS" :
                            break;//выходим из switch обратно в начало while
                    }
                }


            }

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
