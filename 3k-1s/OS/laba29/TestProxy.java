import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class TestProxy extends Thread
{
	private String url;

	public TestProxy(String url)
	{
		this.url = url;
	}

	@Override
	public void run()
	{
		try (PrintWriter writer = new PrintWriter(new Socket("localhost", 5680).getOutputStream()))
		{
			writer.print("GET " + url + " HTTP/1.0\r\n");
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}

	public static void main(String[] args)
	{
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in)))
		{
			while (true)
			{
				new TestProxy(reader.readLine()).start();
			}
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}
	}
}