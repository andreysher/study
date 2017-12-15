import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import sun.net.www.http.HttpClient;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Iterator;

public class MyMessagesHandler implements HttpHandler {
    public Server serv;

    public MyMessagesHandler(Server server) {
        serv = server;
    }

    public static final int DEFAULT_OFFSET = 0;
    public static final int DEFAULT_COUNT = 10;

    public void showMessages(HttpExchange exchange, int offset, int count) throws IOException {
        synchronized (serv.messanges) {
            int quantity = 0;
            Iterator itr = serv.messanges.iterator();
            JsonArray arr = new JsonArray();
            while ((itr.hasNext()) && (quantity < count)) {
                MyMessage m = (MyMessage) itr.next();
                if (m.id >= offset) {
                    Gson g = new Gson();
                    arr.add(g.toJson(m));
                }
            }
            OutputStreamWriter out = new OutputStreamWriter(exchange.getResponseBody());
            out.write(arr.toString());
            Headers h = exchange.getResponseHeaders();
            h.set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, arr.toString().length());
            out.close();
        }
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        String auth = httpExchange.getRequestHeaders().getFirst("Authorization");
        synchronized (serv.clients) {
            Iterator itr = serv.clients.iterator();
            while (itr.hasNext()) {
                MyClient t = (MyClient) itr.next();
                if (auth.equals(t.tocken)) {
                    t.online = System.currentTimeMillis();
                    String method = httpExchange.getRequestMethod();
                    if (method.equals("GET")) {
                        String uri = httpExchange.getRequestURI().toString();
                        System.out.println(uri.contains("/messages?offset="));
                        System.out.println(uri.contains("&count="));
                        System.out.println(uri.indexOf('=') == 16);
                        if ((uri.contains("/messages?offset=")) && (uri.contains("&count=")) && (uri.indexOf('=') == 16)) {
                            int offset = Integer.parseInt(uri.substring((uri.indexOf('=') + 1), uri.indexOf('&')));
                            System.out.println(offset);
                            System.out.println(uri.lastIndexOf('='));
                            System.out.println(uri.substring(uri.lastIndexOf("=")));
                            int count = Integer.parseInt(uri.substring(uri.lastIndexOf('=') + 1));
                            System.out.println(count);
                            System.out.println("offset = " + offset);
                            System.out.println("count = " + count);
                            showMessages(httpExchange, offset, count);
                        }
                    }
                    Headers head = httpExchange.getRequestHeaders();
                    if ((method.equals("POST")) && ("application/json".equals(head.getFirst("Content-Type"))) &&
                            ("/messages".equals(httpExchange.getRequestURI().toString()))) {
                        System.out.println("tut");
                        InputStreamReader input = new InputStreamReader(httpExchange.getRequestBody(), "UTF-8");
                        JsonParser p = new JsonParser();
                        JsonObject obj = (JsonObject) p.parse(input);
                        String message = obj.get("message").getAsString();
                        int id = serv.getNextMessageID();
                        serv.messanges.add(new MyMessage(message, id));
                        OutputStreamWriter out = new OutputStreamWriter(httpExchange.getResponseBody());
                        JsonObject answer = new JsonObject();
                        answer.addProperty("id", id);
                        answer.addProperty("message", message);
                        out.write(answer.toString());
                        System.out.println(answer.toString());
                        Headers h = httpExchange.getResponseHeaders();
                        h.set("Content-Type", "application/json");
                        httpExchange.sendResponseHeaders(200, answer.toString().length());
                        out.close();
                }
            }

            }
        }
    }
}
