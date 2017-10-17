import org.junit.Assert;
import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by Андрей on 01.04.2017.
 */
public class DIVTest {
    @Test
    public void execute() throws Exception {
        DIV p = new DIV();
        List<String> commandArgs = new LinkedList<String>();
        Context context = new Context();
        context.pushOnSteck(5);
        context.pushOnSteck(10);
        p.execute(commandArgs, context);
        if(context.popFromSteck() != 2)
            Assert.fail();
        POP pop = new POP();
        commandArgs.add("sfsdf");
        try{
            pop.execute(commandArgs,context);
            p.execute(commandArgs,context);
        }
        catch (YourStackIsEmpty e){
            System.out.println(e.getMessage());
        }
        catch (InvalidCommandArgs ex){
            System.out.println(ex.getMessage());
        }
    }

}