import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

public class MyOptionsDialog extends JDialog {

    MyPanel panel;
    Interface anInterface;

    MyOptionsDialog(MyPanel panel, Interface inter){
        super((Frame) null,"Options",true);
        this.panel = panel;
        this.anInterface = inter;
        setPreferredSize(new Dimension(400,700));
        setLayout(new GridLayout(0,1));
//        setResizable(false);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
//        System.out.println(123.123);
        fillingDialog();
    }

    public void fillingDialog(){
        add(new JLabel("Field properties"));

        add(new JLabel("Width"));
        final JSlider widthSlider = new JSlider(1, 101, Params.modelWidth);
        final JTextField width = new JTextField();
        width.setText(Integer.toString(Params.modelWidth));
        width.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent keyEvent) {
                super.keyReleased(keyEvent);
                if(width.getText().matches("\\d+") && width.getText().length() < 4) {
                    widthSlider.setValue(Integer.parseInt(width.getText()));
                }
                else {
                    width.setText("");
                }
            }
        });
        widthSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent changeEvent) {
               width.setText(Integer.toString(widthSlider.getValue()));
            }
        });
        add(width);
        add(widthSlider);

        add(new JLabel("Height"));
        final JSlider heightSlider = new JSlider(1, 101, Params.modelWidth);
        final JTextField height = new JTextField();
        height.setText(Integer.toString(Params.modelHeight));
        height.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent keyEvent) {
                super.keyReleased(keyEvent);
                if(height.getText().matches("\\d+") && height.getText().length() < 4) {
                    heightSlider.setValue(Integer.parseInt(height.getText()));
                }
                else {
                    height.setText("");
                }
            }
        });
        heightSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent changeEvent) {
                height.setText(Integer.toString(heightSlider.getValue()));
            }
        });
        add(height);
        add(heightSlider);

        add(new JLabel("Cell size"));
        final JSlider cellSizeSlider = new JSlider(3, 103, Params.cellSize);
        final JTextField cellSize = new JTextField();
        cellSize.setText(Integer.toString(Params.cellSize));
        cellSize.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent keyEvent) {
                super.keyReleased(keyEvent);
                if(cellSize.getText().matches("\\d+") && cellSize.getText().length() < 4) {
                    cellSizeSlider.setValue(Integer.parseInt(cellSize.getText()));
                }
                else {
                    cellSize.setText("");
                }
            }
        });
        cellSizeSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent changeEvent) {
                cellSize.setText(Integer.toString(cellSizeSlider.getValue()));
            }
        });
        add(cellSize);
        add(cellSizeSlider);

        add(new JLabel("Cross line size"));
        final JSlider crossLineSlider = new JSlider(1, 101, Params.crossLineSize);
        final JTextField crossLineSize = new JTextField();
        crossLineSize.setText(Integer.toString(Params.crossLineSize));
        crossLineSize.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent keyEvent) {
                super.keyReleased(keyEvent);
                if(crossLineSize.getText().matches("\\d+") && crossLineSize.getText().length() < 4) {
                    crossLineSlider.setValue(Integer.parseInt(crossLineSize.getText()));
                }
                else {
                    crossLineSize.setText("");
                }
            }
        });
        crossLineSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent changeEvent) {
                crossLineSize.setText(Integer.toString(crossLineSlider.getValue()));
            }
        });
        add(crossLineSize);
        add(crossLineSlider);

        add(new JLabel(" "));
        add(new JLabel("Environment"));
        add(new JLabel("LIVE_BEGIN"));
        final JTextField liveBegin = new JTextField();
        liveBegin.setText(Double.toString(Params.LIVE_BEGIN));
        add(liveBegin);
        add(new JLabel("LIVE_END"));
        final JTextField liveEnd = new JTextField();
        liveEnd.setText(Double.toString(Params.LIVE_END));
        add(liveEnd);
        add(new JLabel("BIRTH_BEGIN"));
        final JTextField birthBegin = new JTextField();
        birthBegin.setText(Double.toString(Params.BIRTH_BEGIN));
        add(birthBegin);
        add(new JLabel("BIRTH_END"));
        final JTextField birthEnd = new JTextField();
        birthEnd.setText(Double.toString(Params.BIRTH_END));
        add(birthEnd);

        add(new JLabel(" "));
        add(new JLabel("Impact"));
        add(new JLabel("FST_IMPACT"));
        final JTextField fstImpact = new JTextField();
        fstImpact.setText(Double.toString(Params.FST_IMPACT));
        add(fstImpact);
        add(new JLabel("SND_IMPACT"));
        final JTextField sndImpact = new JTextField();
        sndImpact.setText(Double.toString(Params.SND_IMPACT));
        add(sndImpact);

        add(new JLabel(" "));
        add(new JLabel("Mod"));
        JRadioButton replace = new JRadioButton("Replace");
        final JRadioButton xor = new JRadioButton("XOR");
        ButtonGroup modGr = new ButtonGroup();
        if(Params.clickMode == 1){
            xor.setSelected(true);
            replace.setSelected(false);
        }
        else {
            replace.setSelected(true);
            xor.setSelected(false);
        }
        modGr.add(replace);
        modGr.add(xor);
        add(replace);
        add(xor);

        final JButton ok = new JButton("OK");
        ok.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                if(!height.getText().equals("")){
                    Params.modelHeight = Integer.parseInt(height.getText());
                }
                else {
                    JOptionPane.showMessageDialog(height, "Enter field Height");
                    return;
                }

                if(!width.getText().equals("")){
                    Params.modelWidth = Integer.parseInt(width.getText());
                }
                else {
                    JOptionPane.showMessageDialog(width, "Enter field width");
                    return;
                }

                if(!cellSize.getText().equals("")){
                    Params.cellSize = Integer.parseInt(cellSize.getText());
                }
                else {
                    JOptionPane.showMessageDialog(cellSize, "Enter Cell size");
                    return;
                }

                if(!crossLineSize.getText().equals("")){
                    Params.crossLineSize = Integer.parseInt(crossLineSize.getText());
                }
                else {
                    JOptionPane.showMessageDialog(crossLineSize,"Enter cross line size");
                    return;
                }

                if(liveBegin.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+") &&
                        liveEnd.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+") &&
                        birthBegin.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+") &&
                        birthEnd.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+")){
                    double lb = Double.parseDouble(liveBegin.getText());
                    double le = Double.parseDouble(liveBegin.getText());
                    double bb = Double.parseDouble(liveBegin.getText());
                    double be = Double.parseDouble(liveBegin.getText());
                    if(lb <= bb && bb <= be && be <= le){
                        Params.LIVE_BEGIN = lb;
                        Params.BIRTH_BEGIN = bb;
                        Params.LIVE_END = le;
                        Params.BIRTH_END = be;
                    }
                }
                else {
                    JOptionPane.showMessageDialog(crossLineSize, "Incorrect Environment");
                    return;
                }

                if(fstImpact.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+") &&
                        sndImpact.getText().matches("\\d+|\\d+,\\d+|\\d+.\\d+")){
                    Params.FST_IMPACT = Double.parseDouble(fstImpact.getText());
                    Params.SND_IMPACT = Double.parseDouble(sndImpact.getText());
                }
                else {
                    JOptionPane.showMessageDialog(crossLineSize, "Incorrect impacts");
                    return;
                }

                if(xor.isSelected()){
                    Params.clickMode = 1;
                    anInterface.xor.setSelected(true);
                    anInterface.replace.setSelected(false);
                }
                else {
                    Params.clickMode = 0;
                    anInterface.replace.setSelected(true);
                    anInterface.replace.setSelected(false);
                }
                dispose();
                anInterface.onInit();

            }
        });
        add(ok);
        JButton cancel = new JButton("Cancel");
        cancel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                dispose();
            }
        });
        add(cancel);
        pack();
        setVisible(true);
    }



}
