//package main.java;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by Андрей on 10.03.2017.
 */
public class main {
    public static void main(String[] args){
        Calculator calc = new Calculator();
        if(args.length == 0){
            calc.run(System.in);
        }
        if(args.length == 1){
            try  {
                calc.run(new BufferedInputStream(new FileInputStream(args[0])));
            }
            catch(IOException fileNotOpen) {
                System.out.println(fileNotOpen.getMessage());
            }
        }
        else{
            System.out.println("too many arguments!");
        }

    }
}
