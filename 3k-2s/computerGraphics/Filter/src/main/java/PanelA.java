import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;

public class PanelA extends JPanel {

    Image scaledImg;
    JPanel select;

    int originalW;
    int originalH;

    PanelA(){
        select = new JPanel();
        setPreferredSize(new Dimension(350, 350));
        setBorder(BorderFactory.createDashedBorder(Color.BLACK));
        setLayout(null);
        addMouseMotionListener(new MouseMotionListener() {
            @Override
            public void mouseDragged(MouseEvent mouseEvent) {

            }

            @Override
            public void mouseMoved(MouseEvent mouseEvent) {
                selected(mouseEvent.getX(),mouseEvent.getY());
            }
        });
    }

    public void setOriginalImageSize(int w, int h){
        originalW = w;
        originalH = h;
    }

    public void setScaledImg(Image img){
        scaledImg = img;
    }

    public void drawScaledImage(){
        this.getGraphics().drawImage(scaledImg, 0,0,null);
    }

    public void paintComponent(Graphics graphics){
        drawScaledImage();
    }

    public void selected(int x, int y){
        select.setOpaque(false);
        select.setBackground(null);
        select.setBorder(BorderFactory.createDashedBorder(Color.BLACK,1,4,2,false));

        int selectSize;
        if(originalW > originalH){
            selectSize = 350*350/originalW;
        }
        else {
            selectSize = 350*350/originalH;
        }

        select.setSize(selectSize,selectSize);
        int selectX = x - selectSize/2;
        int selectY = y - selectSize/2;
        if(x - selectSize/2 < 0){
            selectX = 0;
        }
        if(x + selectSize/2 > 350){
            selectX = 350 - selectSize;
        }
        if(y - selectSize/2 < 0){
            selectY = 0;
        }
        if(y + selectSize/2 > 350){
            selectY = 350 - selectSize;
        }

        add(select);
        select.setBounds(selectX,selectY, selectSize,selectSize);

        System.out.println(selectSize);
        System.out.println(select.getWidth() + " " + select.getHeight() +  " " + select.getLocation());
    }
}