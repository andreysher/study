import org.junit.Assert;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by Андрей on 31.03.2017.
 */
public class POPTest {
    @Test
    public void execute() throws Exception {
        POP pop = new POP();
        List<String> conamdArgs = new LinkedList<String>();
        Context context = new Context();
        context.pushOnSteck(5);
        if (context.popFromSteck() != 5) {
            Assert.fail();
        }
    }
}