import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.InvalidParameterException;
import java.util.Timer;
import java.util.TimerTask;

public class Interface extends JFrame {
    public JMenuBar menuBar;
    public JToolBar toolBar;
    public Model model;
    public MyPanel panel;
    public JScrollPane scroll;

    JToggleButton impact;
    JToggleButton replace;
    JToggleButton xor;
    JToggleButton run;
    boolean timerCreated = false;

    public Interface(){
        super("FIT_15206_Shcherbin_Life");
        setPreferredSize(new Dimension(800,600));
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        menuBar = new JMenuBar();
        this.setJMenuBar(menuBar);
        toolBar = new JToolBar("FIT_15206_Shcherbin_Life");
        toolBar.setRollover(true);
        add(toolBar, BorderLayout.PAGE_START);

        model = new Model(Params.modelWidth, Params.modelHeight);
        panel = new MyPanel(model);
        scroll = new JScrollPane(panel);
        scroll.setDoubleBuffered(true);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener() {
            @Override
            public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                scroll.paint(scroll.getGraphics());
                panel.paint(panel.getGraphics());
            }
        });
        scroll.getHorizontalScrollBar().addAdjustmentListener(new AdjustmentListener() {
            @Override
            public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                scroll.paint(scroll.getGraphics());
                panel.paint(panel.getGraphics());
            }
        });
        add(scroll, BorderLayout.CENTER);
        fillInterface();
        pack();
    }

    public void refrash(){
        remove(scroll);
        remove(panel);
        panel = new MyPanel(model);
        scroll = new JScrollPane(panel);
        scroll.setDoubleBuffered(true);
        scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener() {
            @Override
            public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                scroll.paint(scroll.getGraphics());
                panel.paint(panel.getGraphics());
            }
        });
        scroll.getHorizontalScrollBar().addAdjustmentListener(new AdjustmentListener() {
            @Override
            public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                scroll.paint(scroll.getGraphics());
                panel.paint(panel.getGraphics());
            }
        });
        add(scroll, BorderLayout.CENTER);
        pack();
        setVisible(true);
        panel.paint(panel.getGraphics());
        scroll.paint(scroll.getGraphics());
    }



    /** Создает объект MenuItem для использования в JToolBar
     * @param title - Имя объекта
     * @param tooltip - Описание при наведении мышкой
     * @param icon - Путь до иконки
     * @param actionMethod - Строка название метода, который вызывается при нажатии на кнопку
     * @return JMenuItem
     * @throws NoSuchMethodException
     */
    public JMenuItem createMenuItem(String title, String tooltip, String icon, String actionMethod) throws NoSuchMethodException {
        JMenuItem item = new JMenuItem(title);
        item.setToolTipText(tooltip);
        if(icon != null){
            item.setIcon(new ImageIcon(getClass().getResource(""+icon), title));
        }
        final Method method = getClass().getMethod(actionMethod);
        item.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                try{
                    method.invoke(Interface.this);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
        });
        return item;
    }

    /** Создает JMenu
     * @param title
     * @param actionMethod
     * @return JMenuItem
     * @throws SecurityException
     * @throws NoSuchMethodException
     */
    private JMenuItem createMenuItem(String title, String actionMethod) throws SecurityException, NoSuchMethodException {
        return createMenuItem(title,null, null, actionMethod);
    }

    private JMenu createSubMenu(String title) {
        JMenu menu = new JMenu(title);
        return menu;
    }

    private JMenu addSubMenu(MenuElement parent, String title) {
//        MenuElement element = getParentMenuElement(title);
        if(parent == null)
            throw new InvalidParameterException("Menu path not found: "+title);
        JMenu subMenu = createSubMenu(title);
        if(parent instanceof JMenuBar) {
            ((JMenuBar) parent).add(subMenu);
        }
        else if(parent instanceof JMenu)
            ((JMenu)parent).add(subMenu);
        else if(parent instanceof JPopupMenu)
            ((JPopupMenu)parent).add(subMenu);
        else
            throw new InvalidParameterException("Invalid menu path: "+title);
        return subMenu;
    }

    public void addMenuItem(MenuElement parent, String title, String tooltip, String icon, String actionMethod) throws SecurityException, NoSuchMethodException {
//        MenuElement element = getParentMenuElement(title);
        if(parent == null)
            throw new InvalidParameterException("Menu path not found: "+title);
        JMenuItem item = createMenuItem(title, tooltip, icon, actionMethod);
        if(parent instanceof JMenu)
            ((JMenu)parent).add(item);
        else if(parent instanceof JPopupMenu)
            ((JPopupMenu)parent).add(item);
        else
            throw new InvalidParameterException("Invalid menu path: "+title);
    }

    public void addMenuItem(MenuElement parent, String title, String actionMethod) throws SecurityException, NoSuchMethodException {
//        MenuElement element = getParentMenuElement(title);
        if(parent == null)
            throw new InvalidParameterException("Menu path not found: "+title);
        JMenuItem item = createMenuItem(title, actionMethod);
        if(parent instanceof JMenu)
            ((JMenu)parent).add(item);
        else if(parent instanceof JPopupMenu)
            ((JPopupMenu)parent).add(item);
        else
            throw new InvalidParameterException("Invalid menu path: "+title);
    }

    private JButton createToolBarButton(JMenuItem item) {
        JButton button = new JButton(item.getIcon());
        for(ActionListener listener: item.getActionListeners())
            button.addActionListener(listener);
        button.setToolTipText(item.getToolTipText());
        return button;
    }

    private JToggleButton createToolBarToggleButton(JMenuItem item){
        JToggleButton button = new JToggleButton(item.getIcon());
        for(ActionListener listener: item.getActionListeners())
            button.addActionListener(listener);
        button.setToolTipText(item.getToolTipText());
        return button;
    }

    public void addToolBarButton(JMenuItem menuItem)
    {
        toolBar.add(createToolBarButton(menuItem));
    }

    public void addToolBarSeparator()
    {
        toolBar.addSeparator();
    }



    public void onOpen(){
        JFileChooser fileOpen = new JFileChooser();
        int ret = fileOpen.showDialog(null, "Открыть файл");
        if(ret == JFileChooser.APPROVE_OPTION){
            File f = fileOpen.getSelectedFile();
            try {
                model = FileUtils.readFromFile(f);
                refrash();
            } catch (ParseException e) {
                e.printStackTrace();
            }
        }
    }

    public void onSave(){
        JFileChooser s = new JFileChooser();
        if(s.showOpenDialog(this) == JFileChooser.APPROVE_OPTION){
            File f = s.getSelectedFile();
            try {
                FileUtils.writeToFile(f,model);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    public void onExit(){
        if(!Params.saved){
            onSave();
        }
        System.exit(EXIT_ON_CLOSE);
    }

    public void onOptions(){
//        System.out.println("132456");
        MyOptionsDialog dialog = new MyOptionsDialog(panel, this);
    }

    public void onReplace(){
        if(replace.isSelected()){
            Params.clickMode = 0;
        }
        else {
            Params.clickMode = 1;
        }
    }

    public void onXOR(){
        if(replace.isSelected()){
            Params.clickMode = 0;
        }
        else {
            Params.clickMode = 1;
        }
    }


    public void onImpact(){

        if(!Params.withImpact){
            Params.withImpact = true;
            panel.paint(panel.getGraphics());
            return;
        }

        if (Params.withImpact){
            Graphics2D gr2d =(Graphics2D) Params.impacts.getGraphics();
            gr2d.setColor(new Color(255,255,255,0));
            gr2d.fillRect(0,0,Params.impacts.getWidth(),Params.impacts.getHeight());
            Params.withImpact = false;
            scroll.paint(scroll.getGraphics());
            return;
        }
    }

    public void onInit(){
        model.field = new int[Params.modelHeight][Params.modelWidth];
//        model.width = Params.modelWidth;зл
//        model.height = Params.modelHeight;
        Params.impacts = new BufferedImage(Params.modelWidth * 2 * Params.cellSize + (Params.crossLineSize * Params.modelWidth),
                Params.modelHeight * 2 * Params.cellSize + (Params.crossLineSize * Params.modelHeight) , BufferedImage.TYPE_INT_ARGB);
        refrash();
//        panel.drawImpacts();
        onNext();
    }

    public void onNext(){
        System.out.println("123456");
        panel.model.refrash();
        if(Params.crossLineSize ==1) {
            panel.paintModel();
        }
        else {
            panel.paintFatModel();
        }
    }

    class MyTask extends TimerTask{

        @Override
        public void run() {
            if(!run.isSelected()){
                return;
            }
            onNext();
        }
    }

//придумать как убивать поток таймера при отжатии кнопки
    public void onRun(){
        MyTask task = null;
        Timer t = null;
        if(run.isSelected() && timerCreated){
            return;
        }
        if(!run.isSelected()){
            return;
        }
        if(run.isSelected() && !timerCreated) {
            task = new MyTask();
            t = new Timer();
            t.schedule(task, 0, Params.timerPeriod);
            timerCreated = true;
            return;
        }
    }


    public void onAbout(){
        JOptionPane.showMessageDialog(this, "Life, version 1.0\nCopyright 2018 Andrey Shcherbin, FIT, group 15206", "About Life", JOptionPane.INFORMATION_MESSAGE);
    }



    public void fillInterface(){
        try {

            JMenu file = addSubMenu(this.menuBar, "File");
            addMenuItem(file, "Open", "onOpen");
            addMenuItem(file, "Save", "onSave");
            file.addSeparator();
            addMenuItem(file, "Exit", "onExit");
            JMenu modify = addSubMenu(this.menuBar, "Modify");
            addMenuItem(modify, "Options", "onOptions");
            modify.addSeparator();
            addMenuItem(modify, "Replace", "onReplace");
            addMenuItem(modify, "XOR", "onXOR");
            modify.addSeparator();
            addMenuItem(modify, "Impact", "onImpact");
            JMenu action = addSubMenu(this.menuBar, "Action");
            addMenuItem(action, "Init", "onInit");
            addMenuItem(action, "Next", "onNext");
            addMenuItem(action, "Run", "onRun");
            JMenu help = addSubMenu(this.menuBar,"Help");
            addMenuItem(help, "about program", "onAbout");

            addToolBarButton(createMenuItem("Open", "Open file with game field", "Open.png", "onOpen"));
            addToolBarButton(createMenuItem("Save", "Save current game state to file", "Save.png", "onSave"));
            addToolBarSeparator();
            addToolBarButton(createMenuItem("Options", "Change game options", "Options.png", "onOptions"));
            impact = createToolBarToggleButton(createMenuItem("Impact", "Show the impact values for cells", "Impact.png", "onImpact"));
            toolBar.add(impact);
            replace = createToolBarToggleButton(createMenuItem("Replace", "After mouse click the Cell become alive", "Replace.png", "onReplace"));
            replace.setSelected(true);
            toolBar.add(replace);
            xor = createToolBarToggleButton(createMenuItem("XOR", "After mouse click the Cell change state", "XOR.png", "onXOR"));
            toolBar.add(xor);
            ButtonGroup g = new ButtonGroup();
            g.add(replace);
            g.add(xor);
            addToolBarSeparator();
            addToolBarButton(createMenuItem("Init", "Clear game field", "Init.png", "onInit"));
            addToolBarButton(createMenuItem("Next", "Show next moment of life", "Next.png", "onNext"));
            run = createToolBarToggleButton(createMenuItem("Run", "Run the game", "Run.png", "onRun"));
            toolBar.add(run);
            addToolBarSeparator();
            addToolBarButton(createMenuItem("Exit", "Leave the game", "Exit.png", "onExit"));
            addToolBarButton(createMenuItem("About", "Information about student", "About.gif", "onAbout"));


        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        pack();
        setVisible(true);
    }

    public static void main(String[] args) {
        Interface inter = new Interface();
        inter.onNext();
    }

}
