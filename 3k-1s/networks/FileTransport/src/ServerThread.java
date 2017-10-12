import java.io.*;
import java.net.Socket;
import java.util.Date;

public class ServerThread implements Runnable {
    private Socket socket = null;
    public static final int BUFFERSIZE = 1024;
    public static final int EXIST = 2;
    public static final int SUCCESS = 0;
    public static final int QUANT = 1000;
    public static final int FAIL = 1;

    public ServerThread(Socket sock){
        socket = sock;
    }

    public void run(){
        try {
            System.out.println("Server thread started");
            String name;
            long size;
            boolean flag = false;
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

                    FileOutputStream fileOutput = new FileOutputStream(recvFile);

                    byte[] buffer = new byte[BUFFERSIZE];
                    Date dateStart = new Date();
                    long start = dateStart.getTime();
                    long tempstart = start;
                    long tempsize = 0;
                    long sz = 0;
                    for (int i = 0; i < size; ) {
                        int read = input.read(buffer);
                        if(read == -1){
                            System.out.println("error while read data");
                            out.write(FAIL);
                            out.flush();
                            socket.close();
                            break;
                        }
                        tempsize += read;
                        sz += tempsize;
                        Date moment = new Date();
                        long lasting = moment.getTime() - tempstart;
                        if (lasting >= QUANT) {
                            System.out.println("current speed is " + (tempsize / (lasting)) + "B/s");
                            Date moment1 = new Date();
                            System.out.println("middle speed is " + ((sz / (moment1.getTime() - start))/60) + "B/s");
                            tempstart = moment1.getTime();
                            tempsize = 0;
                        }


                        i += read;

                        fileOutput.write(buffer, 0, read);
                        flag = (boolean) (i == size);
                        if (flag){
                            break;
                        }
                    }
                }
                    out.write(SUCCESS);
                    out.flush();
                    socket.close();
            }
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }
}
