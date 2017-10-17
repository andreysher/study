//package main.java;

import java.util.List;

/**
 * Created by Андрей on 11.03.2017.
 */
public class PRINT extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        if(context.sizeOfSteck() != 0) {
            System.out.println(context.peeek());
        }
        else {
            throw new YourStackIsEmpty();
        }
    }
}
