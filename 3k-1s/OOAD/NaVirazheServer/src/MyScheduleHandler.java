import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;
import java.util.Iterator;
import java.util.Map;

public class MyScheduleHandler implements HttpHandler {
    Server server;

    public MyScheduleHandler(Server server) {
        this.server = server;
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        OutputStream answ = httpExchange.getResponseBody();
        OutputStreamWriter out = new OutputStreamWriter(answ);
        System.out.println("schedule");
        String uri = httpExchange.getRequestURI().toString();
        if (!uri.equals("/schedule")) {
            httpExchange.sendResponseHeaders(404, 0);
        }
        if (!httpExchange.getRequestMethod().equals("GET")) {
            httpExchange.sendResponseHeaders(405, 0);
        }
        Headers headers = httpExchange.getRequestHeaders();
        String tock = headers.getFirst("Authorization");
        if (tock == null) {
            httpExchange.sendResponseHeaders(401, 0);
        }
        synchronized (server.tockens) {
            Iterator itr = server.tockens.entrySet().iterator();
            while (itr.hasNext()) {
                Map.Entry ent = (Map.Entry) itr.next();
                String t = (String) ent.getKey();
                if (t.equals(tock)) {
                    JSONArray ans = server.dbController.getScheduleForUser((String) ent.getValue());
                    Headers resph = httpExchange.getResponseHeaders();
                    resph.set("Content-Type", "application/json");
                    String ansStr = ans.toJSONString();
                    out.write(ansStr);
                    int len = ansStr.length();
                    httpExchange.sendResponseHeaders(200, len);
                }
            }
        }
    out.close();
    }

}
