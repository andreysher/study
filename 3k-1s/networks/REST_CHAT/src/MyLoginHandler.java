import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.*;
import java.util.Iterator;

public class MyLoginHandler implements HttpHandler {
    public Server serv;
    public MyLoginHandler(Server server){
        serv = server;
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        System.out.println("get request");
        System.out.println(httpExchange.getRequestURI());
        if(!httpExchange.getRequestURI().toString().equals("/login")){
            httpExchange.sendResponseHeaders(400, 0);
        }
        if(!httpExchange.getRequestMethod().equals("POST")){
            httpExchange.sendResponseHeaders(405,0);
        }
        InputStream body = httpExchange.getRequestBody();
        JsonParser parser = new JsonParser();
        JsonObject jsonObject = (JsonObject) parser.parse(new InputStreamReader(body, "UTF-8"));
        String username = jsonObject.get("username").getAsString();
        body.close();
        synchronized (serv.clients){
            Iterator itr = serv.clients.iterator();
            while(itr.hasNext()){
                MyClient t = (MyClient) itr.next();
                if(t.username.equals(username)){
                    Headers h = httpExchange.getResponseHeaders();
                    h.set("WWW-Authenticate", "Token realm=’Username is already in use");
                    httpExchange.sendResponseHeaders(401, 0);
                    return;
                }
            }
        }
        Headers headers = httpExchange.getResponseHeaders();
        headers.set("Content-Type", "application/json");
        JsonObject resp = new JsonObject();
        int id = serv.nextClientID();
        resp.addProperty("id", id);
        resp.addProperty("username", username);
        long online = System.currentTimeMillis();
        resp.addProperty("online", online);
        String tock = serv.getNewTocken();
        resp.addProperty("tocken", tock);
        OutputStream answer = httpExchange.getResponseBody();
        OutputStreamWriter out = new OutputStreamWriter(answer);
        out.write(resp.toString());
        int len = resp.toString().length();
        httpExchange.sendResponseHeaders(200, len);
        serv.clients.add(new MyClient(username, id, online, tock));
        out.close();
        //ретернится? или блокируется?
    }
}
