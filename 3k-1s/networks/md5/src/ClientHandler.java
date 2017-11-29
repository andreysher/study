import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketAddress;
import java.util.Iterator;
import java.util.List;

public class ClientHandler extends Thread {

    private Socket socket;
    private String hashString;
    private List<int[]> tasks;
    private List<String> results;
    private InetSocketAddress clientAddress;
    private List<ControlledTask> executingTasks;
    private SocketAddress prevAddr;

    ClientHandler(Socket socket, String hashString, List<int[]> tasks,
                  List<String> results, List<ControlledTask> executingTasks){
        this.socket = socket;
        this.hashString = hashString;
        this.tasks = tasks;
        this.results = results;
        this.executingTasks = executingTasks;
        this.clientAddress = (InetSocketAddress) socket.getRemoteSocketAddress();
    }

    @Override
    public void run(){
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter writer = new PrintWriter(socket.getOutputStream())){
            while(true){
                int counting = Integer.parseInt(reader.readLine());
                if(counting == 1){
                    int execution = 0;
                    synchronized (executingTasks) {
                        Iterator iterator = executingTasks.iterator();
                        prevAddr = new InetSocketAddress(reader.readLine(),
                                Integer.parseInt(reader.readLine()));
                        while (iterator.hasNext()) {
                            ControlledTask t = (ControlledTask) iterator.next();
                            if (t.clientAddr.toString().equals(prevAddr.toString())) {
                                execution = 1;
                            }
                        }
                    }
                    writer.println(execution);
                    writer.flush();
                    if(execution == 1){
                        String answer = reader.readLine();
                        System.out.println(answer);
                        if(answer.equals("SUCCESS")){
                            String ans = reader.readLine();
                            results.add(ans);
                            System.out.println("added" + ans);
                            tasks.clear();
                            socket.close();
                        }
                        else {
                            //client disconnected
                            socket.close();
                            return;
                            //handler kill
                        }
                    }
                    else {
                        socket.close();
                        return;
                    }
                }else {
                    int[] task;
                    synchronized (tasks){
                        if(tasks.isEmpty()){
                            task = null;
                        }
                        else {
                            task = tasks.remove(0);
                            synchronized (executingTasks) {
                                executingTasks.add(new ControlledTask(task, clientAddress,
                                        System.currentTimeMillis()));
                                System.out.println("add to exec");
                            }
                        }
                    }
                    if(task == null){
                        System.out.println("Stopped Client:" + clientAddress);
                        writer.println(0);
                        writer.flush();
                        if(reader.readLine().equals("SUCCESS")){
                            results.add(reader.readLine());
                        }
                        socket.close();
                        return;
                    }
                    else {
                        writer.println(1);
                        writer.flush();
                        writer.println(hashString);
                        writer.println(task[0]);
                        writer.println(task[1]);
                        writer.flush();
                    }
                }
                    System.out.println("Task sent");
                socket.close();
                return;
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
}
