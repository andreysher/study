//package main.java;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import org.apache.log4j.Logger;
import org.apache.log4j.LogManager;

/**
 * Created by Андрей on 03.03.2017.
 */
public class Calculator {
    private static final Logger logger = LogManager.getLogger(Calculator.class);
    public void run(InputStream inStream){
        logger.info("\n===============================" +
                "\nNew start of program at " + java.util.Calendar.getInstance().getTime().toString());
        String val;
        Context context = new Context();
        Scanner scan = new Scanner(inStream);
            while (scan.hasNextLine()) {
                val = scan.nextLine();
                if(val.length() == 0){
                    continue;
                }
                List<String> commandLine = parse(val);
                String comandName = commandLine.get(0);
                commandLine.remove(0);
                command cmd = null;
                try {
                    cmd = ComandFactory.getCommand(comandName);
                }catch (ClassNotFoundException e){
                    System.out.println(e.getMessage());
                }
                try {
                    if(cmd != null) {
                        cmd.execute(commandLine, context);
                    }
                }catch(CalcException e){
                    System.out.println(e.getMessage());
                }
                }
    }
    private List<String> parse(String val){
        Scanner scan = new Scanner(val);
        List<String> res = new LinkedList<String>();
        while(scan.hasNext())
            res.add(scan.next());
        return res;
    }
}
