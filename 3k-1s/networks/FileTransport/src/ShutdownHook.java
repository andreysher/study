import java.io.*;

public class ShutdownHook extends Thread {
    private File f;
    private int fileSize;
    public ShutdownHook(OutputStream out, File file, long size){
        f = file;
    }

    public void run(){
        System.out.println("Server shouting down");
        if(f.length() < fileSize) {
            f.delete();
        }
    }
}
