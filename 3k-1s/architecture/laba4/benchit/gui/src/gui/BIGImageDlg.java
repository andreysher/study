package gui;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;

import javax.imageio.ImageIO;
import javax.swing.*;

public class BIGImageDlg extends JFrame {
	private static final long serialVersionUID = 1L;

	class ImagePanel extends JPanel implements MouseWheelListener {
		private static final long serialVersionUID = 1L;
		private final BufferedImage img;
		private Image drawImg;
		public int addSize = 0;
		public ImagePanel(BufferedImage img) {
			this.img = img;
			addMouseWheelListener(this);
			resizeImg();
		}

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);
			try {
				if (drawImg != null && drawImg.getWidth(this) > 0 && drawImg.getHeight(this) > 0)
					g.drawImage(drawImg, 0, 0, this);
			} catch (Exception e) {}
		}

		public void resizeImg() {
			int w = Math.max(img.getWidth() + addSize, 1);
			drawImg = img.getScaledInstance(w, -1, Image.SCALE_SMOOTH);
			revalidate();
			repaint();
		}

		public void componentShown(ComponentEvent arg0) {}

		public void mouseWheelMoved(MouseWheelEvent e) {
			// Zoom by 5 percent
			int rot = (int) (e.getPreciseWheelRotation() * img.getWidth() * 5 / 100);
			addSize -= rot;
			resizeImg();
		}

		@Override
		public Dimension getPreferredSize() {
			int w = img.getWidth();
			int h = img.getHeight();
			double ratio = (double) h / w;
			w = Math.max(w + addSize, 0);
			h = (int) (w * ratio);
			return new Dimension(w, h);
		}

		public int getImgWidth() {
			return img.getWidth();
		}
	}

	private class MyScrollPane extends JScrollPane implements ComponentListener {
		private static final long serialVersionUID = 1L;
		private final ImagePanel imgPanel;
		private int oldW;

		public MyScrollPane(ImagePanel imgPanel) {
			super(imgPanel);
			// this.add(new JScrollPane(imgPanel));
			this.imgPanel = imgPanel;
			oldW = getWidth();
			imgPanel.addSize = getAddSize();
			this.addComponentListener(this);
		}

		private int getAddSize() {
			return getWidth() - imgPanel.getImgWidth() - getVerticalScrollBar().getMaximumSize().width
					- 4;
		}

		public void componentHidden(ComponentEvent e) {}

		public void componentMoved(ComponentEvent e) {}

		public void componentResized(ComponentEvent e) {
			imgPanel.addSize += getWidth() - oldW;
			oldW = getWidth();
			if (imgPanel.getPreferredSize().width == 0) {
				int addSize = getAddSize();
				oldW -= imgPanel.addSize - addSize;
				imgPanel.addSize = addSize;
			}
			imgPanel.resizeImg();
		}

		public void componentShown(ComponentEvent e) {}
	}

	public BIGImageDlg(String title, File imgFile) throws FileNotFoundException, IOException {
		if (!imgFile.exists())
			throw new FileNotFoundException(imgFile.getAbsolutePath() + "not found");
		Box box = Box.createVerticalBox();
		JLabel titleLbl = new JLabel(title);
		titleLbl.setAlignmentX(CENTER_ALIGNMENT);
		box.add(titleLbl);
		BufferedImage image = ImageIO.read(imgFile);
		MyScrollPane sp = new MyScrollPane(new ImagePanel(image));
		sp.setAlignmentX(CENTER_ALIGNMENT);
		box.add(sp);
		JButton btClose = new JButton("Close");
		btClose.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				BIGImageDlg.this.setVisible(false);
			}
		});
		btClose.setAlignmentX(CENTER_ALIGNMENT);
		box.add(btClose);
		add(box);
		setTitle(title);
		setSize(800, 600);
		setMinimumSize(new Dimension(400, 300));
		setLocationRelativeTo(null);
		setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		pack();
		setVisible(true);
	}
}
