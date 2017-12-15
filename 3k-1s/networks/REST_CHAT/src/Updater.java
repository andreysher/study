import com.google.gson.JsonArray;

import java.io.IOException;

public class Updater extends Thread {
    public Client me;

    public Updater(Client me){
        this.me = me;
    }

    public void run(){
        try {
    sleep(1000);
            while(true) {
                JsonArray prev = me.showUsers();
                sleep(100);
                if(prev != null) {
                    if (!prev.equals(me.showUsers())) {
                        System.out.println("Online clients is changed");
                        System.out.println(me.showUsers().toString());
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
