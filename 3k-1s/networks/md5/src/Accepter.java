import java.io.IOException;
import java.net.ServerSocket;
import java.net.SocketException;
import java.util.List;

public class Accepter extends Thread
{
    private ServerSocket socket;
    private String hashString;
    private int length;
    List<String> results;
    List<int[]> tasks;
    private List<ControlledTask> executingTasks;

    Accepter(ServerSocket socket, String hashString, List<int[]> tasks,
             List<String> results, List<ControlledTask>  executingTasks)
    {
        this.socket = socket;
        this.hashString = hashString;
        this.results = results;
        this.tasks = tasks;
        this.executingTasks = executingTasks;
    }

    @Override
    public void run()
    {
        //дает handlerу готовый сокет для общения с клиентом
        while (!this.isInterrupted())
        {
            try
            {
                new ClientHandler(socket.accept(), hashString, tasks, results, executingTasks).start();
            }
            catch (SocketException e)
            {
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
    }
}
