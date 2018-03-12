import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Stack;

public class MyPanel extends JPanel {

    public BufferedImage fieldView;
    public Model model;

    @Override
    public void update(Graphics g){
        paint(g);
    }

    public MyPanel(final Model model){
        setDoubleBuffered(true);
        fieldView = new BufferedImage((Params.modelWidth*3*(Params.cellSize+Params.crossLineSize)),
                (Params.modelHeight) * 2 * Params.cellSize + (3*Params.crossLineSize * Params.modelHeight) , BufferedImage.TYPE_INT_ARGB);
        setPreferredSize(new Dimension(fieldView.getWidth(), fieldView.getHeight()));
        this.model = model;
        if(Params.crossLineSize == 1) {
            drawHexGread();
        }
        else {
            drawFatHexGreed();
        }
        for (int i = 0; i < Params.modelHeight; i++) {
            int curj = Params.modelWidth;
            if(i%2 != 0){
                curj -= 1;
            }
            for (int j = 0; j < curj; j++) {
                if(model.field[i][j] == 1){
                    fillingCell(i,j, Params.fillingColor);
                }
            }
        }

        addMouseListener(new MyMouseListener(this));
        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent mouseEvent) {
                Params.saved = false;
                int newX = mouseEvent.getX();
                int newY = mouseEvent.getY();
                if(fieldView.getRGB(newX,newY) == Params.borderColor){
                    return;
                }
                int[] cell = getCell(newX,newY);
                if(cell != null){
                    if(Arrays.equals(cell,Params.curPoint)){
                        return;
                    }
                    else {
                        Params.curPoint = cell;
                    }
                    if(Params.clickMode == 0) {
                        model.field[cell[0]][cell[1]] = 1;
                        spanFilling(newX, newY, fieldView, Params.fillingColor);
//                        paintModel();
                        paint(MyPanel.super.getGraphics());
                    }
                    if(Params.clickMode == 1){
                        if(model.field[cell[0]][cell[1]] == 0){
                            model.field[cell[0]][cell[1]] = 1;
                            spanFilling(newX, newY, fieldView, Params.fillingColor);
//                            paintModel();
                            paint(MyPanel.super.getGraphics());
                            return;
                        }
                        if(model.field[cell[0]][cell[1]] == 1){
                            model.field[cell[0]][cell[1]] = 0;
                            spanFilling(newX, newY, fieldView, getBackground().getRGB());
//                            paintModel();
                            paint(MyPanel.super.getGraphics());
                            return;
                        }
                    }
                }
            }
        });
    }

    public void drawFatHexGreed(){
        for (int i = 0; i < Params.modelHeight; i++) {
            int curj = Params.modelWidth;
            if (i % 2 != 0) {
                curj -= 1;
            }
            for (int j = 0; j < curj; j++) {
                drawFatHex(Params.crossLineSize,i, j, fieldView);
            }
        }
    }

    public void drawFatHex(int lineSize, int x, int y, BufferedImage img){
        double xdef = 0;
        int[] A = new int[2];
        int[] B = new int[2];
        int[] C = new int[2];
        int[] D = new int[2];
        int[] E = new int[2];
        int[] F = new int[2];
        double sqrt3mulSize = Math.sqrt(3) * (Params.cellSize+lineSize);
        if(x%2 != 0){
            xdef = sqrt3mulSize/2;
        }
        A[0] = (int)Math.floor(y*sqrt3mulSize + xdef + lineSize/2);//+
        A[1] = (int)Math.floor((Params.cellSize+lineSize)/2 + 1.5*(Params.cellSize+lineSize)*x);//+
        B[0] = (int)Math.floor(y*sqrt3mulSize + xdef + sqrt3mulSize/2 + lineSize/2);
        B[1] = (int) Math.floor(1.5*(Params.cellSize+lineSize)*x);
        C[0] = (int) Math.floor(y*sqrt3mulSize + xdef + sqrt3mulSize+ lineSize/2);
        C[1] = A[1];
        D[0] = C[0];
        D[1] = C[1] + (Params.cellSize+lineSize);
        E[0] = B[0];
        E[1] = B[1] + 2*(Params.cellSize+lineSize);
        F[0] = A[0];
        F[1] = D[1];

        Graphics2D graph = (Graphics2D)img.getGraphics();
        graph.setStroke(new BasicStroke(lineSize));
        graph.setColor(new Color(0,0,0));

        graph.drawLine(A[0], A[1], B[0], B[1]);
        graph.drawLine(B[0], B[1], C[0], C[1]);
        graph.drawLine(A[0], A[1], F[0], F[1]);



        if(y == 0 && x%2 == 0){
            graph.drawLine(F[0],F[1],E[0],E[1]);
        }

        if(x == Params.modelHeight-1){
            graph.drawLine(E[0],E[1],D[0],D[1]);
            graph.drawLine(F[0],F[1],E[0],E[1]);
        }

        if(y == Params.modelWidth-1 && x%2 == 0){
            graph.drawLine(C[0],C[1],D[0],D[1]);
            graph.drawLine(E[0],E[1],D[0],D[1]);
        }

        if(y == Params.modelWidth -2 && x%2 == 1){
            graph.drawLine(C[0],C[1],D[0],D[1]);
        }

//        bresenhamLine(C[0],C[1],D[0],D[1],img);
//        bresenhamLine(E[0],E[1],D[0],D[1],img);
//        bresenhamLine(F[0],F[1],E[0],E[1],img);

//        bresenhamLine(B[0],B[1],A[0],A[1],img);

//        img.setRGB(A[0],A[1],Color.BLUE.getRGB());
//        img.setRGB(B[0],B[1],Color.BLUE.getRGB());
//        img.setRGB(C[0],C[1],Color.BLUE.getRGB());

//        System.out.println((A[0] + ((C[0] - A[0])/2)) + " list " + (B[1] + ((E[1] - B[1])/2)));


        Params.centres.put(new Point((A[0] + ((C[0] - A[0])/2)),B[1] + ((E[1] - B[1])/2)), new Point(x,y));

    }

    public void drawHexGread() {
//        System.out.println("modelWidth " + Params.modelWidth);
        for (int i = 0; i < Params.modelHeight; i++) {
            int curj = Params.modelWidth;
            if (i % 2 != 0) {
                curj -= 1;
            }
            for (int j = 0; j < curj; j++) {
//                System.out.println("drawHex" + i + " " + j);
                drawHexagon(i, j, fieldView);
            }
        }
    }

    public int[] getPixelCords(int x, int y){
        int[] ret = new int[2];
        double xdef = 0;
        double sqrt3mulSize = Math.sqrt(3) * (Params.cellSize+Params.crossLineSize);
        if(x%2 != 0){
            xdef = sqrt3mulSize/2;
        }
        ret[0] = (int)Math.floor(y*sqrt3mulSize + xdef + Params.crossLineSize/2 + Params.cellSize/2);//+
        ret[1] = (int)Math.floor((Params.cellSize+Params.crossLineSize)/2 + 1.5*(Params.cellSize+Params.crossLineSize)*x + Params.cellSize/2);//+
        return ret;
    }

    public void drawFatImpacts(){
        Graphics2D gr2d =(Graphics2D) Params.impacts.getGraphics();
        //чтобы при клике мышкой старые impact не были видны
        //кажется это очень не оптимально!!!

        gr2d.setBackground(new Color(255,255,255,255));
        gr2d.clearRect(0,0,Params.impacts.getWidth(),Params.impacts.getHeight());
        gr2d.setBackground(new Color(255,255,255,0));
        gr2d.clearRect(0,0,Params.impacts.getWidth(),Params.impacts.getHeight());

        gr2d.setColor(Color.red);
        gr2d.setFont(new Font("TimesRoman", Font.PLAIN, Params.cellSize - (Params.cellSize/5)));
        for (int i = 0; i < Params.modelHeight; i++) {
            int currY = Params.modelWidth;
            if (i % 2 != 0) {
                currY -= 1;
            }
            for (int j = 0; j < currY; j++) {
                int[] cords = getPixelCords(i, j);
                double impact = model.getImpact(i,j);
                String imp = String.format("%(.1f",impact);
                if(Math.floor(impact) == impact) {
                    imp = Integer.toString((int) impact);
                }
                gr2d.drawString(imp,cords[0], cords[1]);
            }
        }
        this.getGraphics().drawImage(Params.impacts, 0,0,null);
    }

    public void drawImpacts(){
        Graphics2D gr2d =(Graphics2D) Params.impacts.getGraphics();
        //чтобы при клике мышкой старые impact не были видны
        //кажется это очень не оптимально!!!

        gr2d.setBackground(new Color(255,255,255,255));
        gr2d.clearRect(0,0,Params.impacts.getWidth(),Params.impacts.getHeight());
        gr2d.setBackground(new Color(255,255,255,0));
        gr2d.clearRect(0,0,Params.impacts.getWidth(),Params.impacts.getHeight());

        gr2d.setColor(Color.red);
        gr2d.setFont(new Font("TimesRoman", Font.PLAIN, Params.cellSize - (Params.cellSize/5)));
        for (int i = 0; i < Params.modelHeight; i++) {
            int currY = Params.modelWidth;
            double xdef = 0;
            if (i % 2 != 0) {
                xdef = Math.sqrt(3)*Params.cellSize/2;
                currY -= 1;
            }
            for (int j = 0; j < currY; j++) {
                int[] cords = new int[2];
                cords[0] = (int)Math.ceil(Math.sqrt(3)*Params.cellSize*j + xdef);
                cords[1] = (int)Math.round(Params.cellSize + 1.5*Params.cellSize*i);
                double impact = model.getImpact(i,j);
                String imp = String.format("%(.1f",impact);
                if(Math.floor(impact) == impact) {
                    imp = Integer.toString((int) impact);
                }
                gr2d.drawString(imp,cords[0], cords[1]);
            }
        }
        this.getGraphics().drawImage(Params.impacts, 0,0,null);
    }

    public void paintFatModel(){
        int curY = Params.modelWidth;
        for (int i = 0; i < Params.modelHeight; i++) {
            if(i%2 != 0){
                curY = Params.modelWidth - 1;
            }
            for (int j = 0; j < curY; j++) {
                if(model.field[i][j] == 1){
                    fillingFatCell(Params.crossLineSize,i,j,Params.fillingColor);
                }
                if(model.field[i][j] == 0){

                    fillingFatCell(Params.crossLineSize,i,j, this.getBackground().getRGB());
                }
            }
        }
        this.paint(this.getGraphics());
    }

    public void paintModel(){
        int curY = Params.modelWidth;
        for (int i = 0; i < Params.modelHeight; i++) {
            if(i%2 != 0){
                curY = Params.modelWidth - 1;
            }
            for (int j = 0; j < curY; j++) {
                if(model.field[i][j] == 1){
                    fillingCell(i,j,Params.fillingColor);
                }
                if(model.field[i][j] == 0){

                    fillingCell(i,j, this.getBackground().getRGB());
                }
            }
        }
        this.paint(this.getGraphics());
    }

    public void fillingFatCell(int lineSize, int x, int y, int color){
        double xdef = 0;

        double sqrt3mulSize = Math.sqrt(3) * (Params.cellSize+lineSize);
        if(x%2 != 0){
            xdef = sqrt3mulSize/2;
        }
        int A0 = (int)Math.floor(y*sqrt3mulSize + xdef);//+
        int B1 = (int) Math.floor(1.5*(Params.cellSize+lineSize)*x);
        int C0 = (int) Math.floor(y*sqrt3mulSize + xdef + sqrt3mulSize);
        int E1 = B1 + 2*(Params.cellSize+lineSize);

        int X = A0 + (C0 - A0)/2;
        int Y = B1 + (E1 - B1)/2;
        spanFilling(X,Y,fieldView,color);
    }

    public void fillingCell(int x, int y, int color){
        double dif = 0;
        int[] ret = new int[2];
        if(x % 2 != 0){
            dif = Math.sqrt(3)*Params.cellSize/2;
        }
        ret[0] = (int) Math.ceil((Math.sqrt(3)*Params.cellSize*y) + dif + (Math.sqrt(3)*Params.cellSize/2));
        ret[1] = (int) Math.ceil(Params.cellSize + 1.5*Params.cellSize*x);
        spanFilling(ret[0],ret[1],fieldView,color);
    }

    public int[] getCell(int x, int y){
        int[] ret = new int[2];
        int horisontal = 0;
        int vertical = 0;
        int curX = x;
        int curY = y;
        while(fieldView.getRGB(curX,curY) != Params.borderColor){
            horisontal++;
            curX++;
        }
        curX = x;
        while(fieldView.getRGB(curX,curY) != Params.borderColor){
            horisontal++;
            curX--;
        }
        curX += horisontal/2;
        while (fieldView.getRGB(curX,curY) != Params.borderColor){
            vertical++;
            curY++;
        }
        curY = y;
        try {
            while (fieldView.getRGB(curX, curY) != Params.borderColor) {
                vertical++;
                curY--;
            }
//            TODO: наделать таких варнингов, чтобы нельзя было ткнуть за пределы поля!!!
        } catch (IndexOutOfBoundsException e){
            JOptionPane.showMessageDialog(this,"Out of field", "Warning!", JOptionPane.INFORMATION_MESSAGE);
            return null;
        }
        curY += vertical/2;

//        System.out.println(curX + " cur " + curY);;

        Point p = Params.centres.get(new Point(curX ,curY));
        if(p == null){
            p = Params.centres.get(new Point(curX,curY+1));
        }
        if(p == null){
            p = Params.centres.get(new Point(curX+1, curY));
        }
        if(p == null){
            p = Params.centres.get(new Point(curX+1,curY+1));
        }
        if(p == null){
            p = Params.centres.get(new Point(curX-1, curY+1));
        }
        ret[0] = (int) p.getX();
        ret[1] = (int) p.getY();
//        System.out.println(ret[0] + " " + ret[1]);

        return ret;
    }

    public static void drawHexagon(int x, int y, BufferedImage img){
        double xdef = 0;
        int[] A = new int[2];
        int[] B = new int[2];
        int[] C = new int[2];
        int[] D = new int[2];
        int[] E = new int[2];
        int[] F = new int[2];
        double sqrt3mulSize = Math.sqrt(3) * Params.cellSize;
        if(x%2 != 0){
            xdef = sqrt3mulSize/2;
        }
        A[0] = (int)Math.round(y*sqrt3mulSize + xdef);//+
        A[1] = (int)Math.round(Params.cellSize/2 + 1.5*Params.cellSize*x);//+
        B[0] = (int)Math.round(y*sqrt3mulSize + xdef + sqrt3mulSize/2);
        B[1] = (int)Math.round(1.5*Params.cellSize*x);
        C[0] = (int)Math.round(y*sqrt3mulSize + xdef + sqrt3mulSize);
        C[1] = A[1];
        D[0] = C[0];
        D[1] = C[1] + Params.cellSize;
        E[0] = B[0];
        E[1] = B[1] + 2*Params.cellSize;
        F[0] = A[0];
        F[1] = D[1];

        bresenhamLine(A[0],A[1],F[0],F[1],img);
        bresenhamLine(A[0],A[1],B[0],B[1],img);
        bresenhamLine(B[0],B[1],C[0],C[1],img);

        if(y == 0 && x%2 == 0){
            bresenhamLine(F[0],F[1],E[0],E[1],img);
        }

        if(x == Params.modelHeight-1){
            bresenhamLine(E[0],E[1],D[0],D[1],img);
            bresenhamLine(F[0],F[1],E[0],E[1],img);
        }
//        System.out.println(y);
//        System.out.println(Params.modelWidth);
        if(y == Params.modelWidth-1 && x%2 == 0){
            bresenhamLine(C[0],C[1],D[0],D[1],img);
            bresenhamLine(E[0],E[1],D[0],D[1],img);
        }

        if(y == Params.modelWidth -2 && x%2 == 1){
            bresenhamLine(C[0],C[1],D[0],D[1],img);
        }

//        System.out.println((A[0] + ((C[0] - A[0])/2)) + " list " + (B[1] + ((E[1] - B[1])/2)));

        Params.centres.put(new Point((A[0] + ((C[0] - A[0])/2)),B[1] + ((E[1] - B[1])/2)), new Point(x,y));

//        bresenhamLine(C[0],C[1],D[0],D[1],img);
//        bresenhamLine(E[0],E[1],D[0],D[1],img);
//        bresenhamLine(F[0],F[1],E[0],E[1],img);

//        bresenhamLine(B[0],B[1],A[0],A[1],img);
    }

//    @Override
//    public void paintComponent(Graphics graphics){
//        if(Params.withImpact){
//            drawImpacts();
//        }
//        graphics.drawImage(this.fieldView, 0, 0, null);
//    }


    @Override
    public void paint(Graphics graphics){

        graphics.drawImage(this.fieldView, 0, 0, null);
        if(Params.withImpact){
            if(Params.crossLineSize > 1){
                drawFatImpacts();
            }
            if(Params.crossLineSize == 1){
                drawImpacts();
            }
        }
    }

    public static void bresenhamLine(int x1, int y1, int x2, int y2, BufferedImage img){
        int x = x1;
        int y = y1;
        int dx = Math.abs(x2 - x1);
        int dy = Math.abs(y2 - y1);
        int error = 0;
        int xStep = Integer.signum(x2 - x1);
        int yStep = Integer.signum(y2 - y1);
        img.setRGB(x1,y1,Params.borderColor);
        if(dx > dy){
            error = dx / 2;
            while(x != x2){
                error -= dy;
                if(error < 0){
                    error += dx;
                    x += xStep;
                    y += yStep;
                }
                else {
                    x += xStep;
                }
                img.setRGB(x,y,Params.borderColor);
            }
        }
        else {
            error = dy / 2;
            while (y != y2){
                error -= dx;
                if(error < 0){
                    error += dy;
                    x += xStep;
                    y += yStep;
                }
                else {
                    y += yStep;
                }
                img.setRGB(x,y,Params.borderColor);
            }
        }
    }

    static class Span {
        int rigthX;
        int Y;
        int leftX;
    }

    public static void findNewSpansUp(Stack<Span> spanStack, int x, Span currentSpan,BufferedImage img, int color){
        Span newSpanH = new Span();
        int currentX = x;
        int currentY = currentSpan.Y - 1;
        if(!spanStack.empty() && currentY == spanStack.peek().Y && currentX <= spanStack.peek().rigthX
                && currentX >= spanStack.peek().leftX){
            return;
        }
        while (img.getRGB(currentX, currentY) != Params.borderColor && img.getRGB(currentX,currentY) != color) {
            currentX++;
        }
        newSpanH.rigthX = currentX;
        newSpanH.Y = currentY;
        currentX = x;
        while (img.getRGB(currentX, currentY) != Params.borderColor && img.getRGB(currentX,currentY) != color) {
            currentX--;
        }
        newSpanH.leftX = currentX;
        spanStack.push(newSpanH);
    }

    public static void findNewSpansDown(Stack<Span> spanStack, int x, Span currentSpan,BufferedImage img, int color){
        Span newSpan = new Span();
        int currentX = x;
        int currentY = currentSpan.Y + 1;
        if(!spanStack.empty() && currentY == spanStack.peek().Y && currentX <= spanStack.peek().rigthX
                && currentX >= spanStack.peek().leftX){
            return;
        }
        while (img.getRGB(currentX, currentY) != Params.borderColor && img.getRGB(currentX,currentY) != color) {
            currentX++;
        }
        newSpan.rigthX = currentX;
        newSpan.Y = currentY;
        currentX = x;
        while (img.getRGB(currentX, currentY) != Params.borderColor && img.getRGB(currentX,currentY) != color) {
            currentX--;
        }
        newSpan.leftX = currentX;
        spanStack.push(newSpan);
    }

    public void spanFilling(int x, int y, BufferedImage img, int color){
//        if(color != this.getBackground().getRGB()){
//            Random r = new Random();
//            ArrayList<Integer> colors = new ArrayList<>();
//            colors.add(Color.BLUE.getRGB());
//            colors.add(Color.PINK.getRGB());
//            colors.add(Color.RED.getRGB());
//            colors.add(Color.GREEN.getRGB());
//            colors.add(Color.YELLOW.getRGB());
//            colors.add(Color.ORANGE.getRGB());
//            colors.add(Color.MAGENTA.getRGB());
//            colors.add(Color.DARK_GRAY.getRGB());
//            colors.add(Color.GRAY.getRGB());
//            colors.add(Color.LIGHT_GRAY.getRGB());
//            colors.add(new Color(255, 26, 103).getRGB());
//            colors.add(new Color(18, 178, 255).getRGB());
//            colors.add(new Color(142, 255, 79).getRGB());
//            colors.add(new Color(171, 74, 255).getRGB());
//            colors.add(new Color(102, 255, 196).getRGB());
//
//            Params.fillingColor = colors.get(Math.abs(r.nextInt())%colors.size());
//        }
        int currentX = x;
        int currentY = y;
        Span currentSpan = new Span();
        while(img.getRGB(currentX,currentY) != Params.borderColor){
            currentX++;
        }
        currentSpan.rigthX = currentX;
        currentSpan.Y = currentY;
        currentX = x;
        currentY = y;
        while(img.getRGB(currentX, currentY) != Params.borderColor){
            currentX--;
        }
        currentSpan.leftX = currentX;
        Stack<Span> spanStack = new Stack<>();
        spanStack.push(currentSpan);

        while(!spanStack.empty()) {
            currentSpan = spanStack.pop();
            for (int i = currentSpan.leftX + 1; i < currentSpan.rigthX; i++) {
                img.setRGB(i, currentSpan.Y, color);
                findNewSpansUp(spanStack, i, currentSpan, img,color);
            }
            for (int i = currentSpan.leftX + 1; i < currentSpan.rigthX; i++) {
                img.setRGB(i, currentSpan.Y, color);
                findNewSpansDown(spanStack, i, currentSpan, img,color);
            }
        }
    }
}
