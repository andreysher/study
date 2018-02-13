package View;

import javax.swing.*;
import java.awt.*;

public class MainScreen extends JFrame {
    public MainScreen(){
        super("Заголовок окна");
        this.setPreferredSize(new Dimension(800,600));
        JMenuBar menu = new JMenuBar();
        JMenu file = new JMenu("File");
        JMenuItem file_new = new JMenuItem("New");
        JMenuItem file_open = new JMenuItem("Open");
        JMenuItem file_save = new JMenuItem("Save");
        JMenuItem file_exit = new JMenuItem("Exit");
        JMenu modify = new JMenu("Modify");
        JMenuItem modify_options = new JMenuItem("Options");
        JMenuItem modify_replace = new JMenuItem("Replace???");
        JMenuItem modify_XOR = new JMenuItem("XOR");
        JMenuItem modify_impact = new JMenuItem("Impact");
        JMenu action = new JMenu("Action");
        JMenuItem action_init = new JMenuItem("Init");
        JMenuItem action_next = new JMenuItem("Next");
        JMenuItem action_run = new JMenuItem("Run");
        JMenu help = new JMenu("Help");
        JMenuItem help_about = new JMenuItem("About program");
        menu.add(file);
        file.add(file_new);
        file.add(file_open);
        file.add(file_save);
        file.add(file_exit);
        menu.add(modify);
        modify.add(modify_options);
        modify.add(modify_replace);
        modify.add(modify_XOR);
        modify.add(modify_impact);
        action.add(action_init);
        action.add(action_next);
        action.add(action_run);
        menu.add(help);
        help.add(help_about);
        setJMenuBar(menu);
        pack();
        setVisible(true);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    public static void main(String[] args) {
        MainScreen ms = new MainScreen();
    }
}
