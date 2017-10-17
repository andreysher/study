//package main.java;

import java.util.List;

/**
 * Created by Андрей on 11.03.2017.
 */
public class MINUS extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(commandArgs.size() == 0){
            if(context.sizeOfSteck() < 2){
                throw new YourStackIsEmpty();
            }
            else {
                context.pushOnSteck((context.popFromSteck() - context.popFromSteck()));
            }
        }
    else {
            throw new InvalidCommandArgs();
        }
    }
}
