public class Main {
    public static void main(String[] args) {
        Server server = new Server(args[0], Integer.parseInt(args[1]));
        Thread t = new Thread(server);
        t.start();
    }
}
