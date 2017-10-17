import org.junit.Assert;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by Андрей on 31.03.2017.
 */
public class SQRTTest {
    @Test
    public void execute() throws Exception {
        SQRT sqrt = new SQRT();
        List<String> commandArgs = new LinkedList<String>();
        Context context = new Context();
        context.pushOnSteck(5);
        sqrt.execute(commandArgs,context);
        if(context.popFromSteck() != Math.sqrt(5)){
            Assert.fail();
        }
    }
}