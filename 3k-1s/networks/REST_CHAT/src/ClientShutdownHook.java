import java.io.IOException;

public class ClientShutdownHook extends Thread {

    public Client me;

    public ClientShutdownHook(Client me){
        this.me = me;
    }

    @Override
    public void run() {
        try {
            me.logout();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
