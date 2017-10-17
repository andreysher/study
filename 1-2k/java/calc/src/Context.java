//package main.java;

import java.util.HashMap;
import java.util.Stack;

/**
 * Created by Андрей on 11.03.2017.
 */
public class Context {
    private Stack<Double> stack = new Stack<Double>();
    private HashMap<String,Double> consts = new HashMap<String,Double>();

    public double getConstVal(String constName) throws Exception {
        if(consts.get(constName) == null){
            throw new Exception("This constant not exist");
        }
        double res = consts.get(constName);
        return res;
}

    public void setConst(String name, double value){
        consts.put(name, value);
    }

    public void pushOnSteck(double value){
        stack.push(value);
    }

    public double popFromSteck(){
            return stack.pop();
    }

    public double sizeOfSteck() {
        return stack.size();
    }

    public double peeek(){return stack.peek();}
}
