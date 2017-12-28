import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.sqlite.JDBC;

import java.sql.*;

public class DBController {

    public static final String CON_STR = "jdbc:sqlite:/home/andrey/study/OOAD/database/database.db";

    public static DBController instance = null;

    public static synchronized DBController getInstance() throws SQLException {
        if (instance == null) {
            instance = new DBController();
        }
        return instance;
    }

    private Connection connection;

    private DBController() throws SQLException {
        DriverManager.registerDriver(new JDBC());

        this.connection = DriverManager.getConnection(CON_STR);
        System.out.println("connected");
    }

    String getTockenForUser(String login, String password) {
        System.out.println("tut");
        try (Statement statement = this.connection.createStatement()) {
            String tocken = null;
            ResultSet resultSet = statement.executeQuery(
                    "SELECT login FROM Users WHERE login == " + "'" + login + "'" + ";");
            if (resultSet.next()) {
                if (resultSet.getString("login").equals(login)) {
                    //генерация токена
                    tocken = login;
                }
            }
            System.out.println(tocken);
            return tocken;
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

    JSONArray getScheduleForUser(String login) {
        JSONArray res = new JSONArray();
        try (Statement statement = this.connection.createStatement()) {
            ResultSet resultSet = statement.executeQuery(
                    "select * from Classes where Classes.type IN (select activityType " +
                            "from (select id from Users where login == " + "'" + login + "') as usr " +
                            "join Users_Activitis on usr.id = Users_Activitis.userID)");
            while (resultSet.next()) {
                JSONObject obj = new JSONObject();
                obj.put("name", resultSet.getString("name"));
                obj.put("type", resultSet.getString("type"));
                obj.put("startTime", resultSet.getString("startTime"));
                obj.put("endTime", resultSet.getString("endTime"));
                obj.put("day", resultSet.getString("day"));
                obj.put("room", resultSet.getString("room"));
                obj.put("location", resultSet.getString("location"));
                try(Statement st = this.connection.createStatement()) {
                    ResultSet instructorsSet = st.executeQuery(
                            "select Users.name, Users.surname\n" +
                                    "from Users\n" +
                                    "where Users.id IN\n" +
                                    "(select Classes_Instructors.instructorID\n" +
                                    "from Classes_Instructors\n" +
                                    "where Classes_Instructors.classID =" +
                                    "'" + resultSet.getString("id") + "'" + ")");
                    String instructors = "";
                    while (instructorsSet.next()) {
                        instructors += instructorsSet.getString("name");
                        instructors += " ";
                        instructors += instructorsSet.getString("surname");
                        instructors += ", ";
                    }
                    instructors = instructors.substring(0, instructors.length()-2);
                    obj.put("instructors", instructors);
                } catch (Exception e){
                    e.printStackTrace();
                }

                res.add(obj);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
//        System.out.println(res.toJSONString());
        return res;
    }

    JSONArray getAllSchedule(){
        JSONArray res = new JSONArray();
        try (Statement statement = this.connection.createStatement()) {
            ResultSet resultSet = statement.executeQuery(
                    "select * from Classes");
            while (resultSet.next()) {
                JSONObject obj = new JSONObject();
                obj.put("name", resultSet.getString("name"));
                obj.put("type", resultSet.getString("type"));
                obj.put("startTime", resultSet.getString("startTime"));
                obj.put("endTime", resultSet.getString("endTime"));
                obj.put("day", resultSet.getString("day"));
                obj.put("room", resultSet.getString("room"));
                obj.put("location", resultSet.getString("location"));
                try(Statement st = this.connection.createStatement()) {
                    ResultSet instructorsSet = st.executeQuery(
                            "select Users.name, Users.surname\n" +
                                    "from Users\n" +
                                    "where Users.id IN\n" +
                                    "(select Classes_Instructors.instructorID\n" +
                                    "from Classes_Instructors\n" +
                                    "where Classes_Instructors.classID =" +
                                    "'" + resultSet.getString("id") + "'" + ")");
                    String instructors = "";
                    while (instructorsSet.next()) {
                        instructors += instructorsSet.getString("name");
                        instructors += " ";
                        instructors += instructorsSet.getString("surname");
                        instructors += ", ";
                    }
                    instructors = instructors.substring(0, instructors.length()-2);
                    obj.put("instructors", instructors);
                } catch (Exception e){
                    e.printStackTrace();
                }

                res.add(obj);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        System.out.println(res.toJSONString());
        return res;
    }
}
