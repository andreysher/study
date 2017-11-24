import java.io.*;
import java.math.BigInteger;
import java.net.Socket;
import java.security.MessageDigest;

public class Client extends Thread {

    public static final char[] GENES = {'A', 'C', 'G', 'T'};
    private String host;
    public int port;
    private String hashString;
    public static final int CLIENT_LENGTH = 10;

    public static String codeToString(int code, int length) {
    /*
    code - десятичное число, в котором закодирована четвериная последовательность длинны length
    причем закодирована в обратном порядке, то есть по сути это преобразование в другую СС
    */
        StringBuilder str = new StringBuilder();

        for (int i = 0; i < length; i++) {
            str.append(GENES[code % 4]);
            code /= 4;
        }
        return String.valueOf(str.reverse());
    }

    @Override
    public void run(){
        try(Socket socket = new Socket(host,port)){
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            PrintWriter writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()));

            MessageDigest MD5Counter = MessageDigest.getInstance("MD5");

            hashString = reader.readLine();
            System.out.println(hashString);
            while (true){
                switch (reader.readLine()){
                    case ("STOP"):
                        return;
                    case ("WORK"):
                        System.out.println("work");
                        int prefix = Integer.parseInt(reader.readLine());
                        int length = Integer.parseInt(reader.readLine());
                        System.out.println(codeToString(prefix,length));
                        if(length == 1){
                        //генерим все последовательности до длины client_length
                            for (int currentLength = 0; currentLength <= CLIENT_LENGTH; currentLength++){
                                System.out.println("current length is" + currentLength);
                                if(check( codeToString(prefix, length), currentLength,MD5Counter,hashString,writer)){
                                    return;
                                }
                            }
                            writer.println("UNSECCESS");
                            writer.flush();
                        }
                        else{
                            if(check(codeToString(prefix,length),CLIENT_LENGTH,MD5Counter,hashString,writer)){
                                return;
                            }
                            else {
                                writer.println("UNSECCESS");
                                writer.flush();
                            }
                        }
                }
            }
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    public static boolean check(String prefix,int currentLength, MessageDigest MD5Counter, String hashString, PrintWriter writer){
        int max = (int) Math.pow(Client.GENES.length, currentLength);
        int number = 0;
        while(number < max){
            String temp = prefix + codeToString(number,currentLength);
            String temphash = new BigInteger(1,MD5Counter.digest(temp.getBytes())).toString(16);
            System.out.println(temp);
            if(hashString.equals(temphash)){
                System.out.println("success");
                writer.println("SUCCESS");
                writer.println(temp);
                writer.flush();

                System.out.println("flush");
                return true;
            }
            number++;
        }
        return false;
    }

    public Client(String host, int port){
        this.host = host;
        this.port = port;
    }

    public static void main(String[] args) {
            new Client(args[0], Integer.parseInt(args[1])).start();
    }
}
