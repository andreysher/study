import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.LinkedList;

public class Server implements Runnable {
    public static final int MAX_CLIENTS = 100;
    public LinkedList<MyClient> clients;
    public HttpServer me;
    public int lastGivenID;
    public int lastMessageID;
    public LinkedList<MyMessage> messanges;
    public static final long USER_ACTIVE_TIME = 600000;//10 минут(для теста)

    public Server(String addr, int port){
        try {
            clients = new LinkedList<>();
            messanges = new LinkedList<>();
            InetSocketAddress adr = new InetSocketAddress(InetAddress.getByName(addr), port);
            me = HttpServer.create(adr,MAX_CLIENTS);
            HttpContext loginCont = me.createContext("/login", new MyLoginHandler(this));
            HttpContext logoutCont = me.createContext("/logout" ,new MyLogoutHandler(this));
            HttpContext usersCont = me.createContext("/users", new MyUsersHandler(this));
            HttpContext massagesCont = me.createContext("/messages", new MyMessagesHandler(this));
            logoutCont.setAuthenticator(new Auth(this));
            usersCont.setAuthenticator(new Auth(this));
            massagesCont.setAuthenticator(new Auth(this));
            lastGivenID = 0;
            lastMessageID = -1;
        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {
        me.start();
    }

    public int nextClientID() {
        return lastGivenID++;
    }

    public String getNewTocken() {
        return Integer.toString(lastGivenID);
    }

    public int getNextMessageID(){
        lastMessageID++;
        return lastMessageID;
    }
}

