//package main.java;

import java.util.List;

/**
 * Created by Андрей on 11.03.2017.
 */
public class DEFINE extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(commandArgs.size() == 2){
            context.setConst(commandArgs.get(0), Double.parseDouble(commandArgs.get(1)));
        }
        else{
            throw new InvalidCommandArgs();
        }
    }
}