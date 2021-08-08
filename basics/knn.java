package kNN;
import java.io.*;
import java.util.*;

public class kNearestNeighbor {
	
	static String[] classes;
	static double[] x;
	static double[] y;
	static int _nod = 0;
	static double[] distances;
	static HashMap<Double, String> distanceValues = new HashMap<Double, String>();
	
	
	static int getNumOfDatas(File f) throws FileNotFoundException {
		Scanner s= new Scanner(f);
		int nod=0;
		String str="";
         while(s.hasNextLine()) {
			str=s.nextLine();
			nod++;
			_nod++;
		}
         s.close();
		return nod;
	}
	
	static void getValues() throws FileNotFoundException {
		File f= new File("/home/ayyuce/Desktop/train.txt"); //path to your train.txt file
		int nod= getNumOfDatas(f);
		//System.out.println(nod);
		classes= new String[nod];
		x= new double[nod];
		y= new double[nod];
		Scanner s= new Scanner(f);
		int cnt=0;
		while(s.hasNextLine()) {
			if(cnt==nod) break;
			classes[cnt]=s.next();
			x[cnt]=s.nextDouble();
			y[cnt]=s.nextDouble();
			cnt++;
		}
		s.close();
		
		
	}
	static void getEuclideanDistance(double _x, double _y) {
		distances = new double[_nod];
		for(int i=0; i<_nod;i++) {
			distances[i]=Math.sqrt(Math.pow((_x-x[i]),2) + Math.pow((_y-y[i]), 2));
		}

	}
	
	static void putToHashMap(double _x, double _y) throws FileNotFoundException {
		getValues();
		getEuclideanDistance(_x , _y);
		
		for(int i=0; i<_nod;i++) {
			distanceValues.put(distances[i], classes[i]);
		}
	}
     static double[] bubbleSort(double[] arr) {
		
		int n = arr.length; 
        for (int i = 0; i < n-1; i++) 
            for (int j = 0; j < n-i-1; j++) 
                if (arr[j] > arr[j+1]) { 
                    double temp = arr[j]; 
                    arr[j] = arr[j+1]; 
                    arr[j+1] = temp; 
                } 
        return arr;
		
	}
     
     static void frequency(String[] arr) {
    	List asList = Arrays.asList(arr);
 		Set<String> mySet = new HashSet<String>(asList);
 		for(String s: mySet){
         
 		 System.out.println(s + " " +Collections.frequency(asList,s));

 		}
     }
     static void writeFrequencyToFile(String[] arr) throws FileNotFoundException {
    	 PrintWriter p = new PrintWriter("/home/ayyuce/Desktop/set.txt");
     	List asList = Arrays.asList(arr);
  		Set<String> mySet = new HashSet<String>(asList);
  		for(String s: mySet){
          
  		 p.println(s + " " +Collections.frequency(asList,s));
          
  		}
  		p.close();
      }
	
     static void kNN(double _x, double _y) throws FileNotFoundException {
    	 putToHashMap(_x , _y);
     }
     
     static void prediction(String[] arr, int k) throws FileNotFoundException {
    	 writeFrequencyToFile(arr);
    	 File f = new File("/home/ayyuce/Desktop/set.txt"); 
    	 Scanner sc= new Scanner(f);
    	 Scanner s= new Scanner(f);
    	 int count=0;
    	 String str="";
    	 while(s.hasNextLine()) {
    		 str=s.nextLine();
    		 count++;
    	 }
    	 s.close();
    	 
    	 String[] cls= new String[count];
    	 int[] num= new int[count];
    	 int c= 0;
    	 while(sc.hasNextLine()) {
    		 if(c==count) break;
    		 cls[c]=sc.next();
    		 num[c]=sc.nextInt();
    		 c++;
    	 }
    	 sc.close();
    	 int max=0;
    	 int maxFrequency=num[0];
    	 for(int i=0; i<count; i++) {
    		 if(num[i]>num[max]) {
    			 max=i;
    		 }
    		 if(num[i]>maxFrequency) {
    			 maxFrequency=num[i];
    		 }
    	 }
    	 
    	
    	 
    	 double prediction=((maxFrequency*100)/k);
    	 System.out.println(cls[max] + " "+ prediction +"%");
    	 
     }
     
	public static void main(String[] args) throws FileNotFoundException {
		//5.2 , 3.1
		int k = 7;
		kNN(5.2 , 3.1);
		double[] sortedDistances= bubbleSort(distances);
		String[] kData= new String[k];
		
		for(int i=0; i<k; i++) {
			kData[i]= distanceValues.get(sortedDistances[i]);
		}
		
		
		frequency(kData);
		
		System.out.println("-------------------------");
		prediction(kData, k);
		
	}

}
