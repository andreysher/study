//package main.java;

import java.util.List;

import static java.lang.Math.sqrt;

/**
 * Created by Андрей on 11.03.2017.
 */

public class SQRT extends command {

    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(commandArgs.size() == 0){
            if(context.sizeOfSteck() == 0){
                throw new YourStackIsEmpty();
            }
            else {
                context.pushOnSteck(sqrt(context.popFromSteck()));
            }
        }
        else{
            throw new InvalidCommandArgs();
        }
    }
}
