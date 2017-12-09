import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Iterator;

public class MyUsersHandler implements HttpHandler {

    public Server serv;

    public MyUsersHandler(Server server){
        serv = server;
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        System.out.println(httpExchange.getRequestMethod());
        if(!(httpExchange.getRequestMethod().equals("GET"))){
            httpExchange.sendResponseHeaders(405, 0);
            httpExchange.close();
        }
        if(httpExchange.getRequestURI().toString().equals("/users")){
            synchronized (serv.clients){
                Iterator itr = serv.clients.iterator();
                JsonArray arr = new JsonArray();
                while (itr.hasNext()){
                    MyClient cl = (MyClient) itr.next();
                    boolean online = false;
                    if((System.currentTimeMillis() - cl.online) < Server.USER_ACTIVE_TIME){
                        online = true;
                        JsonObject obj = new JsonObject();
                        obj.addProperty("id", cl.id);
                        obj.addProperty("username", cl.username);
                        obj.addProperty("online", online);
                        arr.add(obj);
                    }
                }
                OutputStreamWriter out = new OutputStreamWriter(httpExchange.getResponseBody());
                out.write(arr.toString());
                int len = arr.toString().length();
                Headers head = httpExchange.getResponseHeaders();
                head.set("Content-Type", "application/json");
                httpExchange.sendResponseHeaders(200, len);
                out.close();
            }
        }
        String uri = httpExchange.getRequestURI().toString();
        System.out.println(uri);
        if((uri.contains("/users/"))&&(uri.lastIndexOf('/') == 6)){
            int id = Integer.parseInt(uri.substring(7));
            System.out.println(id);
            synchronized (serv.clients){
                Iterator itr = serv.clients.iterator();
                while(itr.hasNext()){
                    MyClient t = (MyClient) itr.next();
                    if(t.id == id){
                        JsonObject obj = new JsonObject();
                        String online = "true";
                        if(System.currentTimeMillis() - t.online > Server.USER_ACTIVE_TIME){
                            online = null;
                        }
                        obj.addProperty("id", t.id);
                        obj.addProperty("username", t.username);
                        obj.addProperty("online", online);
                        OutputStreamWriter out = new OutputStreamWriter(httpExchange.getResponseBody());
                        out.write(obj.toString());
                        Headers h = httpExchange.getResponseHeaders();
                        h.set("Content-Type", "application/json");
                        int len = obj.toString().length();
                        httpExchange.sendResponseHeaders(200, len);
                        out.close();
                    }
                }
                httpExchange.sendResponseHeaders(404, 0);
                httpExchange.close();
                return;
            }
        }
        else {
            httpExchange.sendResponseHeaders(400,0);
            httpExchange.close();
        }
    }
}
