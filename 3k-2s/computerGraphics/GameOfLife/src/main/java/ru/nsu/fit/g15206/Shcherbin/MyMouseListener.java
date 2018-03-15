package ru.nsu.fit.g15206.Shcherbin;

import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Arrays;

public class MyMouseListener implements MouseListener {

    MyPanel panel;

    MyMouseListener(MyPanel p){
        this.panel = p;
    }

    @Override
    public void mouseClicked(MouseEvent mouseEvent) {

    }

    @Override
    public void mousePressed(MouseEvent mouseEvent) {
        Params.saved = false;
        int newX = mouseEvent.getX();
        int newY = mouseEvent.getY();
        if(panel.fieldView.getRGB(newX,newY) == Params.borderColor){
            return;
        }
        int[] cell = panel.getCell(newX, newY);
        //
        //
        if (cell != null) {
            if(Arrays.equals(cell,Params.curPoint)){
                return;
            }
            else {
                Params.curPoint = cell;
            }
            if (Params.clickMode == 0) {
                panel.model.field[cell[0]][cell[1]] = 1;
                panel.spanFilling(newX, newY, panel.fieldView, Params.fillingColor);
                panel.paint(panel.getGraphics());
                return;
            }
            if (Params.clickMode == 1) {
                if (panel.model.field[cell[0]][cell[1]] == 1) {
                    panel.model.field[cell[0]][cell[1]] = 0;
                    panel.spanFilling(newX, newY, panel.fieldView, panel.getBackground().getRGB());
                    panel.paint(panel.getGraphics());
                    return;
                }
                if (panel.model.field[cell[0]][cell[1]] == 0) {
                    panel.model.field[cell[0]][cell[1]] = 1;
                    panel.spanFilling(newX, newY, panel.fieldView, Params.fillingColor);
                    panel.paint(panel.getGraphics());
                    return;
                }
            }
        }
    }

    @Override
    public void mouseReleased(MouseEvent mouseEvent) {

    }

    @Override
    public void mouseEntered(MouseEvent mouseEvent) {

    }

    @Override
    public void mouseExited(MouseEvent mouseEvent) {

    }
}
//        }
//        panel.paint(panel.getGraphics());
//    }
//}
