import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.AdjustmentEvent;
import java.awt.event.AdjustmentListener;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.InvalidParameterException;

public class Interface extends JFrame{
    JToolBar toolBar;
    JMenuBar menuBar;
    MyPanel mainPanel;
    JScrollPane scroll;

    Interface(){
        super("FIT_15206_Shcherbin_Filter");
        setPreferredSize(new Dimension(800,600));
        setResizable(false);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        toolBar = new JToolBar("FIT_15206_Shcherbin_Filter");
        toolBar.setRollover(true);
        add(toolBar, BorderLayout.PAGE_START);
        fillInterface();
        pack();
        setVisible(true);
    }


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

    public void onOpen() {

        JFileChooser fileOpen = new JFileChooser();
        int ret = fileOpen.showDialog(null, "Открыть файл");
        if(ret == JFileChooser.APPROVE_OPTION){
            File f = fileOpen.getSelectedFile();
            mainPanel.setImageA(f);
        }
    }

    public void onSave(){

    }

    public void blackAndWhite(){

    }

    public void negative(){

    }

    public void f_s_dither(){

    }

    public void orderDither(){

    }

    public void x2(){

    }

    public void onExit(){
        System.exit(EXIT_ON_CLOSE);
    }

    public void onAbout(){
        JOptionPane.showMessageDialog(this, "Filter, version 1.0\nCopyright 2018 Andrey Shcherbin, FIT, group 15206", "About Life", JOptionPane.INFORMATION_MESSAGE);
    }

    public void roberts(){

    }

    public void sobel(){

    }

    public void border(){

    }

    public void smoothing(){

    }

    public void harshness(){

    }

    public void embossing(){

    }

    public void watercolor(){

    }

    public void rotate(){

    }

    public void gamma(){

    }

    public void bulkDensity(){

    }

    public void onNew(){

    }

    public void onSelect(){

    }

    public void fillInterface(){
        try{

            JMenu file = addSubMenu(menuBar, "File");
            addMenuItem(file, "Open", "onOpen");
            addMenuItem(file, "Save", "onSave");
            file.addSeparator();
            addMenuItem(file, "Exit", "onExit");
            JMenu filters = addSubMenu(this.menuBar, "Filters");
            addMenuItem(filters, "Black & White", "blackAndWhite");
            addMenuItem(filters, "Negative", "negative");
            filters.addSeparator();
            addMenuItem(filters, "F-S dithering", "f_s_dither");
            addMenuItem(filters, "Order dithering", "orderDither");
            filters.addSeparator();
            addMenuItem(filters, "X2", "x2");
            addMenuItem(filters, "Roberts", "roberts");
            addMenuItem(filters, "Sobel", "sobel");
            addMenuItem(filters, "Border", "border");
            filters.addSeparator();
            addMenuItem(filters, "Smoothing", "smoothing");
            addMenuItem(filters, "Harshness", "harshness");
            addMenuItem(filters, "Embossing", "embossing");
            addMenuItem(filters, "Watercolor", "watercolor");
            addMenuItem(filters, "Rotate", "rotate");
            addMenuItem(filters, "Gamma", "gamma");
            addMenuItem(filters, "Bulk density", "bulkDensity");

            JMenu help = addSubMenu(this.menuBar,"Help");
            addMenuItem(help, "about program", "onAbout");

            addToolBarButton(createMenuItem("New document", "Go to initial state", "new.jpeg", "onNew"));
            addToolBarButton(createMenuItem("Open", "Open file with image", "Open.png", "onOpen"));
            addToolBarButton(createMenuItem("Save", "Save current image to file", "Save.png", "onSave"));
            addToolBarButton(createMenuItem("Select", "Select 350x350 image zone", "select.png", "onSelect"));
            addToolBarSeparator();
            addToolBarButton(createMenuItem("Black&White", "Convert image to black and white colors", "blackWhite.jpeg", "onSave"));
            addToolBarButton(createMenuItem("Negative", "Convert image to negative colors", "negative.jpeg", "negative"));
            addToolBarButton(createMenuItem("Floyd", "Floyd-Staming dithering filter", "floyd.png", "f_s_dither"));
            addToolBarButton(createMenuItem("Ordered dithering", "Ordered dithering filter", "ordered.jpeg", "orderDither"));
            addToolBarButton(createMenuItem("Doubling", "Doubling center of image", "x2.jpeg", "x2"));
            addToolBarButton(createMenuItem("Roberts", "Rpberts dither filter", "R.jpeg", "roberts"));
            addToolBarButton(createMenuItem("Sobel", "Sobel dither filter", "S.jpeg", "sobel"));
            addToolBarButton(createMenuItem("Border", "Border different filter", "border.jpeg", "border"));
            addToolBarSeparator();
            addToolBarButton(createMenuItem("Smoothing", "Smoothing filter", "ordered.jpeg", "smoothing"));
            addToolBarButton(createMenuItem("Harshness", "Harshness filter", "ordered.jpeg", "harshness"));
            addToolBarButton(createMenuItem("Embossing", "Embossing filter", "ordered.jpeg", "embossing"));
            addToolBarButton(createMenuItem("Watercolor", "Watercolor filter", "ordered.jpeg", "watercolor"));
            addToolBarButton(createMenuItem("Rotate", "Rotate image", "ordered.jpeg", "rotate"));
            addToolBarButton(createMenuItem("Gamma", "Gamma filter", "ordered.jpeg", "gamma"));
            addToolBarButton(createMenuItem("Bulk density", "Bulk density filter", "ordered.jpeg", "bulkDensity"));
            addToolBarSeparator();
            addToolBarButton(createMenuItem("About", "Information about program", "About.gif", "onAbout"));

            mainPanel = new MyPanel();
            scroll = new JScrollPane(mainPanel);
            scroll.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
            scroll.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED);
            scroll.getVerticalScrollBar().addAdjustmentListener(new AdjustmentListener() {
                @Override
                public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                    mainPanel.zoneA.drawScaledImage();
                }
            });
            scroll.getHorizontalScrollBar().addAdjustmentListener(new AdjustmentListener() {
                @Override
                public void adjustmentValueChanged(AdjustmentEvent adjustmentEvent) {
                    mainPanel.zoneA.drawScaledImage();
                }
            });
            add(scroll);

        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        pack();
        setVisible(true);
    }

    public static void main(String[] args) {
        Interface frame = new Interface();

    }
}
