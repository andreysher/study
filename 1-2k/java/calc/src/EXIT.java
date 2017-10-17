//package main.java;

import java.util.List;

/**
 * Created by Андрей on 16.03.2017.
 */
public class EXIT extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(commandArgs.size() == 0){
            System.exit(0);
        }
        else {
            throw new InvalidCommandArgs();
        }
    }
}
