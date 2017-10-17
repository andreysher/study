//package main.java;

import java.util.List;

/**
 * Created by Андрей on 03.03.2017.
 */
abstract public class command {
        abstract void execute(List<String> commandArgs, Context context) throws CalcException;
}
