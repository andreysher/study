import java.sql.*;

public class Main {
    public static void main(String[] args) {
        try {
            Connection connection = DriverManager.getConnection(
                    "jdbc:sqlite:/home/andrey/study/OOAD/database/database.db");
//            System.out.println("connected");
            Server server = new Server(Integer.parseInt(args[0]));
            Thread st =  new Thread(server);
            st.start();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
