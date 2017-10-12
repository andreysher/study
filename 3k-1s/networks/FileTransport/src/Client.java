import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;


public class Client {
    public static void main(String[] args) {
        try(Socket socket = new Socket(args[1], Integer.parseInt(args[2]))) {
            String path = args[0];
            File file = new File(path);
            if(!file.exists()){
                System.out.println("Want to send non-existing file");
            }
            long fileSize = file.length();

            InputStream input = socket.getInputStream();
            OutputStream output = socket.getOutputStream();

            PrintStream out = new PrintStream(output, false, "utf-8");

            out.println(file.getName());
            out.println(fileSize);
            out.flush();

            int answer = input.read();
            System.out.println("answer");
            if(answer == ServerThread.SUCCESS){
                byte[] buffer = new byte[ServerThread.BUFFERSIZE];
                FileInputStream fileInput = new FileInputStream(file);
                byte[] err = new byte[1];
                while(fileInput.available() > 0){
                    int read = fileInput.read(buffer);
                    output.write(buffer, 0, read);
                    output.flush();
                    int readErr = input.read(err);
                    //обработка ошибок
                    if(readErr == 1){
                        if((int)err[1] == ServerThread.FAIL){
                            System.out.println("server is dead");
                            socket.close();
                            break;
                        }
                    }
                }
                output.flush();
                System.out.println(input.read());
                if(input.read() == ServerThread.SUCCESS){
                    System.out.println("SUCCESS");
                }
                socket.close();
            }
        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("передача не удалась");
            e.printStackTrace();
        }
    }
}
