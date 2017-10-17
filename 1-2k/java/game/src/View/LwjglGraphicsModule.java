package View;

/**
 * Created by andrey on 05.05.17.
 */

import Control.ErrorCatcher;
import Model.GameField;
import Model.ModelConfig;
import org.lwjgl.LWJGLException;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.opengl.GL11.glClearColor;


public class LwjglGraphicsModule implements GraphicsModule{
    private LwjglSpriteSystem spriteSystem;

    /**
     * Инициализирует графический движок и необходимые поля модуля.
     */
    public  LwjglGraphicsModule() {
        initOpengl();
        spriteSystem = new LwjglSpriteSystem();
    }

    private void initOpengl() {
        try {
            /* Задаём размер будущего окна */
            Display.setDisplayMode(new DisplayMode(ViewConfig.SCREEN_WIDTH, ViewConfig.SCREEN_HEIGHT));

            /* Задаём имя будущего окна */
            Display.setTitle(ViewConfig.SCREEN_NAME);

            /* Создаём окно */
            Display.create();
        } catch (LWJGLException e) {
            ErrorCatcher.graphicsFailure(e);
        }

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, ViewConfig.SCREEN_WIDTH,0, ViewConfig.SCREEN_HEIGHT,1,-1);
        glMatrixMode(GL_MODELVIEW);

		/* Для поддержки текстур */
        glEnable(GL_TEXTURE_2D);

		/* Для поддержки прозрачности */
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		/* Белый фоновый цвет */
        glClearColor(1,1,1,1);
    }

    /**
     * Отрисовывает переданное игровое поле
     *
     * @param field Игровое поле, которое необходимо отрисовать
     */
    @Override
    public void draw(GameField field) {
        glClear(GL_COLOR_BUFFER_BIT);

        for(int i = 0; i < ModelConfig.COUNT_CELLS_X; i++) {
            for (int j = 0; j < ModelConfig.COUNT_CELLS_Y; j++) {
                drawCell(ViewConfig.CELL_SIZE*i, ViewConfig.CELL_SIZE*j, field.getState(i,j));
            }
        }

        Display.update();
        Display.sync(60);
    }

    /**
     * Отрисовывает отдельную ячейку
     *
     * @param x Координата отрисовки X
     * @param y Координата отрисовки Y
     * @param state Состояние ячейки
     */
    private void drawCell(int x, int y, int state) {
        spriteSystem.getSpriteByNumber(state).getTexture().bind();

        glBegin(GL_QUADS);
        glTexCoord2f(0,0);
        glVertex2f(x,y+ ViewConfig.CELL_SIZE);
        glTexCoord2f(1,0);
        glVertex2f(x+ ViewConfig.CELL_SIZE,y+ ViewConfig.CELL_SIZE);
        glTexCoord2f(1,1);
        glVertex2f(x+ ViewConfig.CELL_SIZE, y);
        glTexCoord2f(0,1);
        glVertex2f(x, y);
        glEnd();
    }

    /**
     * @return Возвращает true, если в окне нажат "крестик"
     */
    @Override
    public boolean isCloseRequested() {
        return Display.isCloseRequested();
    }

    /**
     * Заключительные действия.
     * Принудительно уничтожает окно.
     */
    @Override
    public void destroy() {
        Display.destroy();
    }
}
