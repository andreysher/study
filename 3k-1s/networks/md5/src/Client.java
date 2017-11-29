import java.io.*;
import java.math.BigInteger;
import java.net.*;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public class Client extends Thread {

    static final char[] GENES = {'A', 'C', 'G', 'T'};
    private String host;
    private int port;
    private String hashString;
    static final int CLIENT_LENGTH = 2;//максимальная длина клиентской подстроки
    private int counting = 0;
    private String answer;
    private String answerString;
    private InetSocketAddress prevAddr;
    private int tryes;

    private static String codeToString(int code, int length) {
    /*переводит код в строку в зависимости от длины*/
        StringBuilder str = new StringBuilder();

        for (int i = 0; i < length; i++) {
            str.append(GENES[code % 4]);
            code /= 4;
        }
        return String.valueOf(str.reverse());
    }

    @Override
    public void run(){
        while(true){
            /*слип можно убрать, это просто для того чтобы смотреть в режиме реального времени*/
            try {
                sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }//крутой трай - все закроется при неудаче
            try(Socket socket = new Socket(host,port);
                BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
                PrintWriter writer = new PrintWriter(new OutputStreamWriter(socket.getOutputStream()))){
                System.out.println(answer);
                MessageDigest MD5Counter = MessageDigest.getInstance("MD5");
                /*counting - флаг считал ли что-то клиент на предыдущем подключении
                адрес предидущего подключения храним, идентифицируем клиентов по ip + port
                * executing - ждет ли сервер ответа на эту задачу*/
                writer.println(counting);
                writer.flush();
//                System.out.println(counting);
                if(counting == 1){
                    writer.println(prevAddr.getAddress());
                    writer.println(prevAddr.getPort());
                    writer.flush();
                    int execution = Integer.parseInt(reader.readLine());
                    System.out.println(execution);
                    if(execution == 1) {
                        writer.println(answer);
                        writer.flush();
                        if(answer.equals("SUCCESS")){
                            System.out.println(answerString);
                            writer.println(answerString);
                            writer.flush();
                            prevAddr = new InetSocketAddress(socket.getLocalAddress(), socket.getLocalPort());
                            socket.close();
                            System.out.println("send " + answerString);
                        }
                        else {
                            counting = 0;
                            prevAddr = new InetSocketAddress(socket.getLocalAddress(), socket.getLocalPort());
                            socket.close();
                        }
                    }
                    else {
                        counting = 0;
                        prevAddr = new InetSocketAddress(socket.getLocalAddress(), socket.getLocalPort());
                        socket.close();
                    }
                }
                else {
                    counting = 1;
                    int works = 0;
                    works = Integer.parseInt(reader.readLine());
                    if(works != 1){
                        prevAddr = new InetSocketAddress(socket.getLocalAddress(), socket.getLocalPort());
                        writer.println(answer);
                        writer.flush();
                        if(answer.equals("SUCCESS")){
                            writer.println(answerString);
                            writer.flush();
                        }
                        socket.close();
                        return;//kill client
                    }
                    System.out.println("alive");
                    hashString = reader.readLine();
                    int prefix = Integer.parseInt(reader.readLine());
                    int length = Integer.parseInt(reader.readLine());
                    System.out.println("disconnect");
                    prevAddr = new InetSocketAddress(socket.getLocalAddress(), socket.getLocalPort());
                    socket.close();
                    System.out.println(codeToString(prefix,length));
                    if(length == 1){//генерим все последовательности до длины client_length
                        for (int currentLength = 0; currentLength <= CLIENT_LENGTH; currentLength++){
                            if((answerString = check( codeToString(prefix, length), currentLength,MD5Counter,hashString)) != null){
                                answer = "SUCCESS";
                            }
                        }
                    }
                    else{
                        if((answerString = check(codeToString(prefix,length),CLIENT_LENGTH,MD5Counter,hashString)) != null){
                            answer = "SUCCESS";
                        }
                        else {
                            answer = "UNSUCCESS";
                        }
                    }
                }

            } catch (UnknownHostException e) {
                e.printStackTrace();
            } catch (ConnectException e){
                try {
                    //on this catch
                    if(tryes == 5){
                        return;
                    }
                    tryes++;
                    System.out.println(tryes);
                    System.out.println("try to connect");
                    sleep(3000);
                } catch (InterruptedException e1) {
                    e1.printStackTrace();
                }
            } catch (SocketException e){
            }
            catch (IOException e) {
                e.printStackTrace();
            } catch (NoSuchAlgorithmException e) {
                e.printStackTrace();
            }
        }
    }

    private static String check(String prefix, int currentLength, MessageDigest MD5Counter,
                                String hashString){
        String answerString;
        int max = (int) Math.pow(Client.GENES.length, currentLength);
        int number = 0;
        while(number < max){
            String temp = prefix + codeToString(number,currentLength);
            String temphash = new BigInteger(1,MD5Counter.digest(temp.getBytes())).toString(16);
            System.out.println(temp);
            if(hashString.equals(temphash)){
                answerString = temp;
                return answerString;
            }
            number++;
        }
        return null;
    }

    private Client(String host, int port){
        this.host = host;
        this.port = port;
        this.answer = "UNSUCCESS";
        tryes = 0;
    }

    public static void main(String[] args) {
        new Client(args[0], Integer.parseInt(args[1])).start();
    }
}
