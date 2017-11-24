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

    Accepter(ServerSocket socket, String hashString, List<int[]> tasks, List<String> results)
    {
        this.socket = socket;
        this.hashString = hashString;
        this.results = results;
        this.tasks = tasks;
    }

    @Override
    public void run()
    {
        int i = 0;

        while (!this.isInterrupted())
        {
            try
            {
                new ClientHandler(socket.accept(), hashString, tasks, results, i).start();
            }
            catch (SocketException e)
            {
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }

            i++;
        }
    }
}
