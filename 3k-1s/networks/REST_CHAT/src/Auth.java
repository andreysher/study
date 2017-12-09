import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.sun.net.httpserver.Authenticator;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import sun.net.www.http.HttpClient;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.Iterator;

public class Auth extends Authenticator {
    public Server server;

    public Auth(Server server) {
        this.server = server;
    }

    @Override
    public Result authenticate(HttpExchange httpExchange) {
        System.out.println("tut");
        Headers headers = httpExchange.getRequestHeaders();
        String tock = headers.getFirst("Authorization");
        System.out.println(tock);
        if(tock.equals("")){
            //нету такого на сервере
            return new Failure(401);
        }
        synchronized (server.clients) {
            Iterator itr = server.clients.iterator();
            while (itr.hasNext()) {
                MyClient t = (MyClient) itr.next();
                if (tock.equals(t.tocken)) {
                    System.out.println("success ");
                    return new Success(new HttpPrincipal("username", "tocken"));
                }
            }
        }
        //умер по таймауту
        return new Failure(403);
    }
}
