package ru.nsu.fit.g15206.Shcherbin;

public class ParseException extends Exception{
    String reason;
    public ParseException(String reason){
        this.reason = reason;
    }
    public void printMessage(){
        System.out.println(reason);
    }
}
