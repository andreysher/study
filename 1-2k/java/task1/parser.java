package task1;
import java.io.*;
import java.util.*;
import java.util.Map.Entry;

class BuildRez {
	HashMap<String, Integer> map;
	int wordCounter;
}

class Word implements Comparable<Word>{
	String word;
	int count;

	public int compareTo(Word arg0) {
		return this.count - arg0.count;
	}
}

public class parser {
	public static void main(String args[]){		
		Reader file = open_file(args[1]);
		
		BuildRez br = build(file);
		
		List<Word> sorted_list = sort(br.map);
		
		Writer writer = openWriteFile();
		
		Show(sorted_list, br.wordCounter, writer);
		try{
			file.close();
			writer.close();
		}catch(IOException e){
			
		}
	}
	
	public static Reader open_file(String file_name){ 
		Reader rez = null;
		try{
			rez = new InputStreamReader(new FileInputStream(file_name));
		}
		catch(IOException e){
			System.err.println("Error while reading file:"+ e.getLocalizedMessage());
		}
		finally{
			if(null != rez){
				try{
					rez.close();
				}
				catch(IOException e){
					e.printStackTrace(System.err);
				}
			}
		}
		return rez;
	}
	
	public static BuildRez build(Reader reader) {
		char ch;
		BuildRez rez = new BuildRez();
		int data = 0;
		rez.map = new HashMap<String, Integer>();
		try {
			data = reader.read();
		} catch (IOException e) {

			e.printStackTrace();
		}
		StringBuilder temp = new StringBuilder();
			while(data != -1){
				ch = (char) data;
				if(Character.isLetterOrDigit(ch)){
					temp.append(ch);
					try {
						data = reader.read();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				else{
					break;
				}
			}
			rez.wordCounter++;
			if(rez.map.containsKey(temp)){
				rez.map.put(temp.toString(), rez.map.get(temp)+1);
			}
			else{
				rez.map.put(temp.toString(), 1);
			}
		return rez;
	}
	
	public static List<Word> sort(HashMap<String, Integer> map){
		List<Word> list = new ArrayList<Word>();
		for( Entry<String,Integer> entry : map.entrySet()){
			Word temp = new Word();
			temp.count = entry.getValue();
			temp.word = entry.getKey();
			list.add(temp);
			//(list.get(i)).count = entry.getValue();
			//(list.get(i)).word = entry.getKey();
		}
		
		Collections.sort(list);
		
		return list;
		
	}
	
	public static Writer openWriteFile(){
		Writer rez = null;
		try{
			rez = new OutputStreamWriter(new FileOutputStream("rezult.txt"));
		}
		catch(IOException e){
			System.err.println("Error while reading file:"+ e.getLocalizedMessage());
		}
		finally{
			if(null != rez){
				try{
					rez.close();
				}
				catch(IOException e){
					e.printStackTrace(System.err);
				}
			}
		}
		return rez;
	}
	
	public static void Show(List<Word> wl, int wordCount,  Writer writer) {
		double frequency = 0.0;
		String str1 = new String();
		String str2 = new String();
		for(int index = 0; index < wl.size(); index++) {
			
			frequency = (wl.get(index)).count / wordCount;
			str1 = String.format("%.5d", frequency);
			str2 = String.format("%.5d /%", frequency * 100);
			try{
			writer.write((wl.get(index)).word);
			writer.write(str1);
			writer.write(str2);
			}catch(IOException e){
		
			}
		}
	}
}
