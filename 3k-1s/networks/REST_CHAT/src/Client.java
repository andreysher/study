import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.*;

import static java.net.HttpURLConnection.HTTP_BAD_METHOD;
import static java.net.HttpURLConnection.HTTP_BAD_REQUEST;
import static java.net.HttpURLConnection.HTTP_OK;

public class Client {
    public InetAddress adr;
    public int port;
    public int tocken;
    public int lastMyMessageID;

    public static final int WAIT_ANSWER = 15 * 1000;
    public static final int DEFAULT_OFFSET = 0;
    public static final int DEFAULT_COUNT = 10;

    public Client(InetAddress address, int port) {
    this.adr = address;
    this.port = port;
    }

    public int login(){
        System.out.println("Please enter your name for login");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        try {
            String name = in.readLine();
            String loginURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/login";
//            System.out.println(loginURL);
            HttpURLConnection con = (HttpURLConnection) new URL(loginURL).openConnection();
//            con.setReadTimeout(WAIT_ANSWER);
//            con.setReadTimeout(WAIT_ANSWER);
            con.setDoOutput(true);
            con.setDoInput(true);
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type","application/json");
//            System.out.println(con.getPermission());
            OutputStreamWriter out = new OutputStreamWriter(con.getOutputStream());
            JsonObject obj = new JsonObject();
            obj.addProperty("username", name);
            out.write(obj.toString());
            out.close();
            con.connect();
            int responseCode = con.getResponseCode();
            if(responseCode == HTTP_OK){
                InputStreamReader response = new InputStreamReader(con.getInputStream());
                JsonParser parser = new JsonParser();
                JsonObject ans = (JsonObject) parser.parse(response);
                response.close();
                System.out.println("Your id is " + ans.get("id").getAsInt());
                return ans.get("tocken").getAsInt();
            }
            if(responseCode == HTTP_BAD_REQUEST){
                System.out.println("Invalid URL");
            }
            if(responseCode == HTTP_BAD_METHOD){
                System.out.println("Invalid method");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -1;
    }
    /*возвращает id сообщения, отрицательное число в случае ошибки*/
    public int postMessage() throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String message = in.readLine();
        String messageURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/messages";
        HttpURLConnection con = (HttpURLConnection) new URL(messageURL).openConnection();
            con.setReadTimeout(WAIT_ANSWER);
            con.setReadTimeout(WAIT_ANSWER);
        con.setDoOutput(true);
        con.setDoInput(true);
        con.setRequestMethod("POST");
        con.setRequestProperty("Authorization", String.valueOf(this.tocken));
        con.setRequestProperty("Content-Type","application/json");
        OutputStreamWriter out = new OutputStreamWriter(con.getOutputStream());
        JsonObject obj = new JsonObject();
        obj.addProperty("message", message);
        out.write(obj.toString());
        out.close();
        con.connect();
        System.out.println("sended");
        int responseCode = con.getResponseCode();
        if(responseCode == HTTP_OK){
            InputStreamReader response = new InputStreamReader(con.getInputStream());
            JsonParser parser = new JsonParser();
            JsonObject ans = (JsonObject) parser.parse(response);
            response.close();
            return ans.get("id").getAsInt();
        }
        if(responseCode == HTTP_BAD_METHOD){
            System.out.println("Invalid method");
        }
        if(responseCode == HTTP_BAD_REQUEST){
            System.out.println("bad request");
        }
        return -1;
    }

    private void getMessages() throws IOException {
        System.out.println("Please enter offset(press Enter for default)");
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        int offset = 0;
        String ofs;
        if(!(ofs = in.readLine()).equals("")){
            offset = Integer.parseInt(ofs);
        }
        System.out.println("Please enter count(press Enter for default)");
        String cnt;
        int count = 10;
        if(!(cnt = in.readLine()).equals("")){
            count = Integer.parseInt(cnt);
        }
        String messageURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/messages?offset=" + offset
                + "&count=" + count;
        HttpURLConnection con = (HttpURLConnection) new URL(messageURL).openConnection();
        con.setReadTimeout(WAIT_ANSWER);
        con.setReadTimeout(WAIT_ANSWER);
        con.setDoOutput(true);
        con.setDoInput(true);
        con.setRequestMethod("GET");
        con.setRequestProperty("Authorization", String.valueOf(this.tocken));
        con.connect();
        System.out.println("sended");
        int responseCode = con.getResponseCode();
        if(responseCode == HTTP_OK){
            InputStreamReader response = new InputStreamReader(con.getInputStream());
            JsonParser parser = new JsonParser();
            JsonArray ans = (JsonArray) parser.parse(response);
            response.close();
            System.out.println(ans.toString());
        }
        if(responseCode == HTTP_BAD_METHOD){
            System.out.println("Invalid method");
        }
        if(responseCode == HTTP_BAD_REQUEST){
            System.out.println("bad request");
        }
    }

    private void logout() throws IOException {
        String logoutURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/logout";
        HttpURLConnection con = (HttpURLConnection) new URL(logoutURL).openConnection();
        con.setReadTimeout(WAIT_ANSWER);
        con.setReadTimeout(WAIT_ANSWER);
        con.setDoOutput(true);
        con.setDoInput(true);
        con.setRequestMethod("GET");
        con.setRequestProperty("Authorization", String.valueOf(this.tocken));
        con.connect();
        System.out.println("sended");
        int responseCode = con.getResponseCode();
        if(responseCode == HTTP_OK){
            InputStreamReader response = new InputStreamReader(con.getInputStream());
            JsonParser parser = new JsonParser();
            JsonObject ans = (JsonObject) parser.parse(response);
            response.close();
            System.out.println(ans.get("message").getAsString());
        }
        if(responseCode == HTTP_BAD_METHOD){
            System.out.println("Invalid method");
        }
        if(responseCode == HTTP_BAD_REQUEST){
            System.out.println("bad request");
        }
    }

    public void showUsers() throws IOException {
        String usersURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/users";
        HttpURLConnection con = (HttpURLConnection) new URL(usersURL).openConnection();
        con.setReadTimeout(WAIT_ANSWER);
        con.setReadTimeout(WAIT_ANSWER);
        con.setDoOutput(true);
        con.setDoInput(true);
        con.setRequestMethod("GET");
        con.setRequestProperty("Authorization", String.valueOf(this.tocken));
        con.connect();
        System.out.println("sended");
        int responseCode = con.getResponseCode();
        if(responseCode == HTTP_OK){
            InputStreamReader response = new InputStreamReader(con.getInputStream());
            JsonParser parser = new JsonParser();
            JsonArray ans = (JsonArray) parser.parse(response);
            response.close();
            System.out.println(ans.toString());
        }
        if(responseCode == HTTP_BAD_METHOD){
            System.out.println("Invalid method");
        }
        if(responseCode == HTTP_BAD_REQUEST){
            System.out.println("bad request");
        }
    }

    public void showUser() throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("Please enter user id");
        String userID = in.readLine();
        String userURL = "http://" + this.adr.getHostAddress() + ":" + this.port + "/users/" + userID;
        HttpURLConnection con = (HttpURLConnection) new URL(userURL).openConnection();
        con.setReadTimeout(WAIT_ANSWER);
        con.setReadTimeout(WAIT_ANSWER);
        con.setDoOutput(true);
        con.setDoInput(true);
        con.setRequestMethod("GET");
        con.setRequestProperty("Authorization", String.valueOf(this.tocken));
        con.connect();
        System.out.println("sended");
        int responseCode = con.getResponseCode();
        if(responseCode == HTTP_OK){
            InputStreamReader response = new InputStreamReader(con.getInputStream());
            JsonParser parser = new JsonParser();
            JsonObject ans = (JsonObject) parser.parse(response);
            response.close();
            System.out.println(ans.toString());
        }
        if(responseCode == HTTP_BAD_METHOD){
            System.out.println("Invalid method");
        }
        if(responseCode == HTTP_BAD_REQUEST){
            System.out.println("bad request");
        }
    }

    public static void main(String[] args) {
        System.out.println("hello! How are you? Where's Sam?");
        Client me = null;
        try {
            me = new Client(InetAddress.getByName(args[0]), Integer.parseInt(args[1]));
            me.tocken = me.login();
            if(me.tocken == -1){
                System.out.println("login fail");
            }
            else {
                BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
                while(true){
                    switch (in.readLine()){
                        case "p m"://post message
                            me.lastMyMessageID = me.postMessage();
                            System.out.println(me.lastMyMessageID);
                            continue;
                        case "g m"://get message
                            me.getMessages();
                            continue;
                        case "logout":
                            me.logout();
                            continue;
                        case "users":
                            me.showUsers();
                            continue;
                        case "user":
                            me.showUser();
                    }
                }
            }
        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
