import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.sql.SQLException;
import java.util.*;

public class Server implements Runnable {
    public Map<String, String> tockens;
    public HttpServer me;
    public DBController dbController;

    public Server(int port){
        try {
            this.tockens = new HashMap<>();
            this.dbController = DBController.getInstance();
            InetSocketAddress adr = new InetSocketAddress(port);
            me = HttpServer.create(adr,0);
            HttpContext authCont = me.createContext("/auth", new MyAuthHandler(this));
            HttpContext scheduleCont = me.createContext("/schedule" ,new MyScheduleHandler(this));
        } catch (IOException | SQLException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {
        me.start();
    }
}

