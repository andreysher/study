import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.*;

public class MyAuthHandler implements HttpHandler {
    private Server server;
    public MyAuthHandler(Server server) {
        this.server = server;
    }

    @Override
    public void handle(HttpExchange httpExchange) throws IOException {
        System.out.println("auth");
        String uri = httpExchange.getRequestURI().toString();
        if(!uri.equals("/auth")){
            httpExchange.sendResponseHeaders(404, 0);
        }
        if(!httpExchange.getRequestMethod().equals("POST")){
            httpExchange.sendResponseHeaders(405,0);
        }
        InputStream body = httpExchange.getRequestBody();
        JSONParser parser = new JSONParser();
        try {
            JSONObject obj = (JSONObject) parser.parse(new InputStreamReader(body));
            String login = (String) obj.get("login");
            String passwd = (String) obj.get("password");
            if(login == null || passwd == null){
                httpExchange.sendResponseHeaders(401, 0);
            }
            String tocken = server.dbController.getTockenForUser(login,passwd);
            if(tocken == null){
                //если такого пользователя не загреано
                httpExchange.sendResponseHeaders(401,0);
            }
            else {
                server.tockens.put(tocken, login);
                Headers resph = httpExchange.getResponseHeaders();
                resph.set("Content-Type","application/json");
                JSONObject resp = new JSONObject();
                resp.put("tocken", tocken);
                OutputStream answer = httpExchange.getResponseBody();
                OutputStreamWriter out = new OutputStreamWriter(answer);
                String ansStr = resp.toJSONString();
                int len = ansStr.length();
                out.write(ansStr);
                httpExchange.sendResponseHeaders(200, len);
                out.close();
            }
        } catch (ParseException e) {
            e.printStackTrace();
        }
    }
}
