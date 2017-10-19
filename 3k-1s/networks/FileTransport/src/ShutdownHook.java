import java.io.*;

public class ShutdownHook extends Thread {
    private File f;
    public ShutdownHook(OutputStream out, File file){
        f = file;
    }

    public void run(){
        System.out.println("Server shouting down");
        f.delete();
    }
}
