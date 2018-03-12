import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;

public class Params {
    public static double FST_IMPACT = 1;
    public static double SND_IMPACT = 0.3;
    public static double BIRTH_BEGIN = 2.3;
    public static double LIVE_BEGIN = 2.0;
    public static double LIVE_END = 3.3;
    public static double BIRTH_END = 2.9;
    public static int crossLineSize = 1;
    public static int cellSize = 15;
    public static int modelWidth = 50;
    public static int modelHeight = 50;
    public static int borderColor = new Color(0,0,0).getRGB();
    public static int fillingColor = new Color(0, 255, 0).getRGB();
    //0-replace, 1-xor
    public static int clickMode = 0;
    public static boolean withImpact = false;
    public static long timerPeriod = 1000;
    public static BufferedImage impacts = new BufferedImage(modelWidth * 2 * cellSize + (crossLineSize * modelWidth),
            modelHeight * 2 * cellSize + (crossLineSize * modelHeight) , BufferedImage.TYPE_INT_ARGB);
    public static JScrollPane scr;
    public static HashMap<Point, Point> centres = new HashMap<>();
    public static int[] curPoint;
    public static boolean saved = true;

    public static boolean setBirthBegin(double birthBegin) {
        if(birthBegin >= LIVE_BEGIN && birthBegin <= BIRTH_END) {
            BIRTH_BEGIN = birthBegin;
            return true;
        }
        return false;
    }

    public static void setFstImpact(double fstImpact) {
        FST_IMPACT = fstImpact;
    }

    public static boolean setBirthEnd(double birthEnd) {
        if(birthEnd >= BIRTH_BEGIN && birthEnd <= LIVE_END){
            BIRTH_END = birthEnd;
            return true;
        }
        return false;
    }

    public static boolean setLiveBegin(double liveBegin) {
        if (liveBegin <= BIRTH_BEGIN){
            LIVE_BEGIN = liveBegin;
            return true;
        }
        return false;
    }

    public static boolean setLiveEnd(double liveEnd) {
        if (liveEnd >= BIRTH_END) {
            LIVE_END = liveEnd;
            return true;
        }
        return false;
    }

    public static void setSndImpact(double sndImpact) {
        SND_IMPACT = sndImpact;
    }

    public static void setCellSize(int cellSize) {
        Params.cellSize = cellSize;
    }

    public static void setCrossLineSize(int crossLineSize) {
        Params.crossLineSize = crossLineSize;
    }
}

