import com.google.gson.JsonObject;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Iterator;

public class MyLogoutHandler implements HttpHandler {
    public Server serv;

    public MyLogoutHandler(Server server){
        serv = server;
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        if(!httpExchange.getRequestURI().toString().equals("/logout")){
            httpExchange.sendResponseHeaders(400, 0);
        }
        if(!httpExchange.getRequestMethod().equals("GET")){
            httpExchange.sendResponseHeaders(405,0);
        }
        Headers h = httpExchange.getRequestHeaders();
        String tock = h.getFirst("Authorization");

        synchronized (serv.clients) {
            Iterator itr = serv.clients.iterator();
            while (itr.hasNext()) {
                MyClient t = (MyClient) itr.next();
                if (t.tocken.equals(tock)) {
                    itr.remove();
                    Headers headers = httpExchange.getResponseHeaders();
                    headers.set("Content-Type", "application/json");
                    JsonObject resp = new JsonObject();
                    resp.addProperty("message", "bye!");
                    httpExchange.sendResponseHeaders(200, resp.toString().length());
                    OutputStream answer = httpExchange.getResponseBody();
                    OutputStreamWriter out = new OutputStreamWriter(answer);
                    out.write(resp.toString());
                    out.close();
                    return;
                }
            }
        }
        httpExchange.sendResponseHeaders(401, 0);
        return;
    }
}
