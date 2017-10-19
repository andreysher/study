import java.io.*;
import java.net.Socket;
import java.util.Date;

public class ServerThread implements Runnable {
    private Socket socket = null;
    public static final int BUFFERSIZE = 4096;
    public static final int EXIST = 2;
    public static final int SUCCESS = 0;
    public static final int QUANT = 4000;
    public static final int FAIL = 1;

    public ServerThread(Socket sock){
        socket = sock;
    }

    public double getCurrentTime(){
        Date moment = new Date();
        double result = (double)moment.getTime();
        return result;
    }

    public void run(){
        try {
            System.out.println("Server thread started");
            String name;
            long size;
            InputStream input = socket.getInputStream();
            OutputStream out = socket.getOutputStream();

            BufferedReader in = new BufferedReader(new InputStreamReader(input, "utf-8"));

            name = in.readLine();
            size = Long.parseLong(in.readLine());
            //удалить слеши
            int slash = name.indexOf('/');
            while (slash != -1) {
                char[] chars = name.toCharArray();
                chars[slash] = '?';
                name = String.valueOf(chars);
                slash = name.indexOf('/');
            }
            name = "uploads/" + name;
            File testik = new File("./uploads/");
            File recvFile;
            if ((testik.canWrite()) && (testik.getFreeSpace() >= size)) {
                recvFile = new File(name);
                if (recvFile.exists()) {
                    System.out.println("File already exist");
                    //write пишет первые 8 бит
                    out.write(EXIST);
                    out.flush();
                } else {
                    out.write(SUCCESS);
                    out.flush();
                    //NEW!!!
                    Runtime.getRuntime().addShutdownHook(new ShutdownHook(out, recvFile, size));
                    FileOutputStream fileOutput = new FileOutputStream(recvFile);
                    byte[] buffer = new byte[BUFFERSIZE];
                    double startSession = getCurrentTime();
                    double quantSize = 0;
                    double quantTime = 0;
                    double startOfQuant = getCurrentTime();
                    for (int i = 0; i < size; ) {
                        int read = input.read(buffer);
                        i += read;
                        quantSize += read;
                        fileOutput.write(buffer, 0, read);
                        quantTime = getCurrentTime() - startOfQuant;
                        if(quantTime >= QUANT){
                            System.out.println(socket.getRemoteSocketAddress() + " middle speed is " +
                                    (i/((getCurrentTime() - startSession)/1000)) + "B/s");
                            System.out.println(socket.getRemoteSocketAddress() + " current speed is " +
                                    (quantSize/(quantTime/1000)) + "B/s");
                            startOfQuant = getCurrentTime();
                            quantTime = 0;
                            quantSize = 0;
                        }
                        if (i == size){
                            System.out.println(socket.getRemoteSocketAddress() + " middle speed is " +
                                    (i/((getCurrentTime() - startSession)/1000)) + "B/s");
                            System.out.println(socket.getRemoteSocketAddress() + " current speed is " +
                                    (quantSize/(quantTime/1000)) + "B/s");
                            System.out.println(socket.getRemoteSocketAddress() + "file transport success");
                            out.write(SUCCESS);
                            out.flush();
                            socket.close();
                            break;
                        }
                    }
                }

            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
