//package main.java;

import java.util.List;

/**
 * Created by Андрей on 16.03.2017.
 */
public class USAGE extends command {
    void execute(List<String> commandArgs, Context context) throws CalcException {
        System.out.println("This calculator support functions:" + System.lineSeparator()
        + "DEFINE - add new variable" + System.lineSeparator()
        + "");
    }
}