package ru.nsu.fit.g15206.Shcherbin;

import java.io.*;
import java.util.LinkedList;
import java.util.Scanner;

public class FileUtils {

    private static String cutComments(String s){//([/][/])(.*)
        String res = s.replaceFirst("( *)//(.*)","");
        return res;
    }

    public static Model readFromFile(File file) throws ParseException {
        try(Scanner scanner = new Scanner(file)){
            String curStr;
            int horisontal;
            int vertical;
            int lineSize;
            int cellSize;
            int cellsQuantity;

            Scanner strSc;
            //---------------------------------------------------------------------
            //читаем размеры поля
            curStr = scanner.nextLine();
            String noComents = cutComments(curStr);
            if(noComents.matches("^([0-9]+)([ ])([0-9]+)([ ]*)$")){
                strSc = new Scanner(noComents);
                vertical = strSc.nextInt();
                horisontal = strSc.nextInt();
            }
            else {
                throw new ParseException("field size incorrect");
            }
            Model model = new Model(vertical,horisontal);
            //--------------------------------------------------------------------
            //читаем ширину линии
            curStr = scanner.nextLine();
            noComents = cutComments(curStr);
            if(noComents.matches("([0-9]+)([ ]*)")){
                lineSize = Integer.parseInt(noComents);
            }
            else {
                throw new ParseException("line size incorrect");
            }
            //--------------------------------------------------------------------
            //читаем размер клетки
            curStr = scanner.nextLine();
            noComents = cutComments(curStr);
            if(noComents.matches("([0-9]+)([ ]*)")){
                cellSize = Integer.parseInt(noComents);
            }
            else {
                throw new ParseException("cell size incorrect");
            }
            //--------------------------------------------------------------------
            //читаем количество живых
            curStr = scanner.nextLine();
            noComents = cutComments(curStr);
            if(noComents.matches("([0-9]+)([ ]*)")){
                cellsQuantity = Integer.parseInt(noComents);
            }
            else {
                throw new ParseException("live quantity incorrect");
            }
            //--------------------------------------------------------------------
            for(int i = 0; i < cellsQuantity; i++) {//пока следующая строка не пустая
                if (!scanner.hasNextLine()){
                    //то есть сюда зайдем если конец файла
                    throw new ParseException("quantity of strings incorrect");
                }
                curStr = scanner.nextLine();
                String withoutComments = cutComments(curStr);
                if (withoutComments.matches("([0-9]+)([ ])([0-9]+)([ ]*)")) {
                    strSc = new Scanner(withoutComments);
                    int x = strSc.nextInt();
                    int y = strSc.nextInt();
                    if(x%2 == 1 && y == horisontal-1){
                        throw new ParseException("invalid cell number");
                    }
                    System.out.println(x+" fileUtils "+y);
                    model.field[x][y] = 1;
                } else {
                    throw new ParseException("cords string incorrect");
                }
            }
            Params.crossLineSize = lineSize;
            Params.cellSize = cellSize;
            Params.modelHeight = vertical;
            Params.modelWidth = horisontal;
            return model;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void writeToFile(File file, Model model) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter(file);
        writer.println(Params.modelWidth + " " + Params.modelHeight);
        writer.println(Params.crossLineSize);
        writer.println(Params.cellSize);
        LinkedList<int []> alive = new LinkedList<>();
        for (int i = 0; i < Params.modelWidth; i++) {
            for (int j = 0; j < Params.modelHeight; j++) {
                if(model.field[i][j] == 1){
                    alive.add(new int[]{i,j});
                }
            }
        }
        writer.println(alive.size());
        for (int[] el:alive) {
            writer.println(Integer.toString(el[0])+ " " + Integer.toString(el[1]));
        }
        writer.flush();
    }
}
