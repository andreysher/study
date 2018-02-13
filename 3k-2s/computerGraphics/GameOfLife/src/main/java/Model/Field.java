package Model;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Scanner;

public class Field {
    int height;
    int width;
    int lineSize = 1;
    int cellSize = 15;
    LinkedList<Cell> cells;

    Field(int height, int width, int lineSize, int cellSize, LinkedList<Cell> cells){
        this.width = width;
        this.height = height;
        this.lineSize = lineSize;
        this.cellSize = cellSize;
        this.cells = cells;
    }

    private static String cutComments(String s){//([/][/])(.*)
        String res = s.replaceFirst("( *)//(.*)","");
        System.out.println(s + " | " + res );
        return res;
    }

    public static Field readFromFile(String filePath) throws ParseException {
        try(Scanner scanner = new Scanner(new File(filePath))){
            String curStr;
            int horisontal;
            int vertical;
            int lineSize;
            int cellSize;
            int cellsQuantity;
            LinkedList<Cell> cells = new LinkedList<>();
            Scanner strSc;
            //---------------------------------------------------------------------
            //читаем размеры поля
            curStr = scanner.nextLine();
            String noComents = cutComments(curStr);
            if(noComents.matches("^([0-9]+)([ ])([0-9]+)([ ]*)$")){
                System.out.println("size of field is correct");
                strSc = new Scanner(noComents);
                horisontal = strSc.nextInt();
                vertical = strSc.nextInt();
            }
            else {
                throw new ParseException("field size incorrect");
            }
            //--------------------------------------------------------------------
            //читаем ширину линии
            curStr = scanner.nextLine();
            noComents = cutComments(curStr);
            if(noComents.matches("([0-9]+)([ ]*)")){
                System.out.println("size of line is correct");
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
                System.out.println("size of cell is correct");
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
                System.out.println("quantity of cell is correct");
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

                    //TODO: указать impact для живых клеток
                    Cell tmp = new Cell(strSc.nextInt(), strSc.nextInt(), 123);

                    cells.add(tmp);
                } else {
                    throw new ParseException("cords string incorrect");
                }
            }
            return new Field(vertical, horisontal, lineSize, cellSize, cells);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) {
        try {
            Field f = Field.readFromFile("/home/andrey/IdeaProjects/GameOfLife/src/main/java/Model/test.txt");
        } catch (ParseException e) {
            e.printMessage();
        }
    }

    void saveToFile(File file){

    }


}
