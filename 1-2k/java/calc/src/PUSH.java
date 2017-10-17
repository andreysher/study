//package main.java;

import java.util.List;

/**
 * Created by Андрей on 11.03.2017.
 */
public class PUSH extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(commandArgs.size() == 1){
            try {
                context.pushOnSteck(Double.parseDouble(commandArgs.get(0)));
            }catch (NumberFormatException e){
             try {
                 context.pushOnSteck(context.getConstVal(commandArgs.get(0)));
             }catch(Exception ex){
                 throw new InvalidCommandArgs();
             }
            }
        }
        else {
            throw new InvalidCommandArgs();
        }
    }
}
