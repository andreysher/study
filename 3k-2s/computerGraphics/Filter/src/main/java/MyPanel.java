import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MyPanel extends JPanel {
    PanelA zoneA;
    JPanel zoneB;
    JPanel zoneC;

    BufferedImage imageB;
    BufferedImage imageC;

    int miniW = -1;
    int miniH = -1;

    MyPanel(){
        setPreferredSize(new Dimension(1090,730));
        setLayout(new FlowLayout(FlowLayout.CENTER,10,10));

        zoneA = new PanelA();
        add(zoneA);

        imageB = new BufferedImage(350,350,BufferedImage.TYPE_3BYTE_BGR);
        zoneB = new JPanel();
        zoneB.setPreferredSize(new Dimension(350,350));
        zoneB.setBorder(BorderFactory.createDashedBorder(Color.BLACK));
        add(zoneB);

        imageC = new BufferedImage(350,350,BufferedImage.TYPE_3BYTE_BGR);
        zoneC = new JPanel();
        zoneC.setPreferredSize(new Dimension(350,350));
        zoneC.setBorder(BorderFactory.createDashedBorder(Color.BLACK));
        add(zoneC);


    }

    public void setImageA(File f){
        try {
            BufferedImage img = ImageIO.read(f);
            zoneA.setOriginalImageSize(img.getWidth(),img.getHeight());
            if(img.getWidth() > 350 || img.getHeight() > 350){
                if(img.getWidth() > img.getHeight()){
                    miniW = 350;
                    miniH = 350*img.getHeight()/img.getWidth();
                }
                else {
                    miniH = 350;
                    miniW = 350*img.getWidth()/img.getHeight();
                }
            }

            Image scaledImageA = img.getScaledInstance(miniW,miniH,Image.SCALE_SMOOTH);
            zoneA.setScaledImg(scaledImageA);
            zoneA.drawScaledImage();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}