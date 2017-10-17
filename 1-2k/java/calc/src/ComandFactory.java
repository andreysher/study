//package main.java;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Properties;

/**
 * Created by Андрей on 11.03.2017.
 */
public class ComandFactory {
     public static command getCommand(String comandName) throws ClassNotFoundException{
          Class c = null;
          InputStream ins = ComandFactory.class.getResourceAsStream("/name.properties");
          Properties properties = new Properties();
          try{
               properties.load(ins);
               ins.close();
          }
          catch(IOException e){
               System.out.println("no such property");
          }

          String comandClassName = properties.getProperty(comandName);
          c = Class.forName(comandClassName);
          Object obj = null;
          try {
               obj = c.newInstance();
          } catch (InstantiationException e) {
               e.printStackTrace();
          } catch (IllegalAccessException e) {
               e.printStackTrace();
          }
          command res = (command) obj;
          return res;
     }
}
