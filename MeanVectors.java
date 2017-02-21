/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.json.JSONObject;

/**
 *
 * @author tsan
 */
public class MeanVectors {
    
    static int num_of_docs=10000;
    static int num_of_tests = 1000;
    static ArrayList<String> filesRead= new ArrayList<>();
    
    public static HashMap<String,String> readTest(String directory, int limit,String type) throws IOException {
        HashMap<String,String> documents = new HashMap<>();
        int count = 0;
        File dir = new File(directory);
        File[] files = dir.listFiles();
        for (File f : files) {
            if(filesRead.contains(f.getName()))
            {
                continue;
            }
            count++;
            if(count>limit && limit!=-1)
            {
                break;
            }
            String text = readFile(f.getAbsolutePath(), Charset.forName("UTF-8"));
            JSONObject jobj;
            try{
                jobj = new JSONObject(text); 
                String doc = jobj.getString("text");
                documents.put(doc, type);                
            } catch(Exception e)
            {
                
            }            
        }
        return documents;
    }
        public static ArrayList<String> readReviews(String directory, int limit) throws IOException {
        ArrayList<String> documents = new ArrayList<>();
        int count = 0;
        File dir = new File(directory);
        File[] files = dir.listFiles();

        for (File f : files) {
            count++;
            if(count>limit && limit!=-1)
            {
                break;
            }
            String text = readFile(f.getAbsolutePath(), Charset.forName("UTF-8"));
            JSONObject jobj;
            try{
                jobj = new JSONObject(text); 
                String doc = jobj.getString("text");
                documents.add(doc);
                filesRead.add(f.getName());
                
            } catch(Exception e)
            {
                
            }
            
        }

        return documents;
    }
        
    public static String readFile(String path, Charset encoding) throws java.io.IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }
    
    
    public static double jaccardSimilarity(HashSet<String> usedTerms,Map<String,Double>test,Map<String,Double>mean)
    {
        //Jaccard
        Double testrev_score = 0.0;
        Double sum_max_score = 0.0;
        Double final_score;
        //Weighted Jaccard for POSITIVE
        for(String s: usedTerms)
        {
            Double test_score = test.get(s);
            Double dict_score = mean.get(s);
            if(test_score<dict_score)
            {
                testrev_score += test_score;
                sum_max_score += dict_score;
            }
            else
            {
                testrev_score += dict_score;
                sum_max_score += test_score;
            }
        }
        final_score = testrev_score/sum_max_score;
        return final_score;
    }
    
    
    public static void createResults(Map<String,Double> mean_pos,Map<String,Double> mean_neg,Map<String,HashMap<String,Integer>> dict,Map<String,Integer> num_pos,Map<String,Integer> num_neg ) throws IOException
    {
        int i=0;
        BufferedWriter out = new BufferedWriter(new FileWriter("prediction.txt"));
        String base_dir = "";
        ArrayList<String> test_docs = readReviews(base_dir+"test/", -1);
        HashSet<String> usedTerms=new HashSet<>();
        Map<String,Double> test_keep =  new TreeMap<>();
        for (String key : dict.keySet()) {
           test_keep.put(key, 0.0);
        }
        for(String doc: test_docs)
        {
            Map<String,Double> test =  test_keep;
            usedTerms.clear();
            int count_terms = 0;
            String[] word_list = doc.split(" ");
            for(String w: word_list)
            {
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                //If the term is in the word dictionary
                if(dict.containsKey(w)) 
                {
                    count_terms++;
                    //Get frequency of term w
                    test.put(w, test.get(w)+1);
                    usedTerms.add(w);
                }
            }
            
            Map<String,Double> test_pos = test;
            Map<String,Double> test_neg = test;
            for(String s: usedTerms)
            {
                double count_s=test.get(s);
                if(count_s == 0)
                {
                    test_pos.put(s, 0.0);
                    test_neg.put(s, 0.0);
                }
                else
                {
                    Double pos_score = (count_s/count_terms)*(Math.log10(num_pos.get(s)/count_s));
                    Double neg_score = (count_s/count_terms)*(Math.log10(num_neg.get(s)/count_s));
                    if(pos_score==Double.NEGATIVE_INFINITY || pos_score == Double.POSITIVE_INFINITY ||
                       neg_score==Double.NEGATIVE_INFINITY || neg_score == Double.POSITIVE_INFINITY)
                    {
                        test_pos.put(s, 0.0);
                        test_neg.put(s, 0.0);
                    }
                    else
                    {
                        test_pos.put(s, pos_score); //TF*IDF
                        test_neg.put(s, neg_score); //TF*IDF
                    }
                }
            }
            
            Double pos_score = jaccardSimilarity(usedTerms,test_pos,mean_pos);
            Double neg_score = jaccardSimilarity(usedTerms,test_neg,mean_neg);
            if(pos_score>neg_score)
            {
                out.write((new DecimalFormat("00000").format(i)+" "+1+"\n"));
            }else
            {
                out.write((new DecimalFormat("00000").format(i)+" "+0+"\n"));
            }
            i++;
        }
        out.close();
        
    }
    
    public static void main(String[] args) throws IOException
    {
        String base_dir = "";
        //Read negative reviews
        ArrayList<String> neg_docs = readReviews(base_dir+"train/neg/", num_of_docs);
        //Read positive reviews
        ArrayList<String> pos_docs = readReviews(base_dir+"train/pos/", num_of_docs);
       
        HashMap<String,String> test_pos_docs = readTest(base_dir+"train/pos/", num_of_tests,"pos");
        HashMap<String,String> test_neg_docs = readTest(base_dir+"train/pos/", num_of_tests,"neg");
        HashMap<String,String> test_docs = new HashMap<>();
        test_docs.putAll(test_pos_docs);
        test_docs.putAll(test_neg_docs);
        //TreeMap gia na einai taksinomimena ws pros to key
        Map<String,HashMap<String,Integer>> dict =  new TreeMap<>();
        //Sto dict kratame times typou "leksi", [{"neg",INT},{"pos",INT}]
        //gia na kserw poses fores emfanizetai i leksi se thetika kai poses se arnitika reviews
        //Negative Reviews
        for (String review : neg_docs) {
            String[] word_list = review.split(" ");
            for(String w: word_list)
            {
                //Leave only unicode and numbers in the word. Eliminate symbols
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                if(dict.containsKey(w))
                {
                    int c = dict.get(w).get("neg");
                    c++;
                    HashMap<String,Integer> a = dict.get(w);
                    a.put("neg",c);
                    dict.put(w, a);
                }
                else
                {
                    HashMap<String,Integer> a = new HashMap<>();
                    a.put("neg",1);
                    a.put("pos",0);
                    dict.put(w, a);
                }
            }
        }
        //Positive Reviews
        for (String review : pos_docs) {
            String[] word_list = review.split(" ");
            for(String w: word_list)
            {
                //Edw tha prepei na vlepw oti i leksi mou einai mono grammata
                //prin tin valw sto Map
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                if(dict.containsKey(w))
                {
                    int c = dict.get(w).get("pos");
                    c++;
                    HashMap<String,Integer> a = dict.get(w);
                    a.put("pos",c);
                    dict.put(w, a);
                }
                else
                {
                    HashMap<String,Integer> a = new HashMap<>();
                    a.put("neg",0);
                    a.put("pos",1);
                    dict.put(w, a);
                }
            }
        }
        Map<String,Integer> num_pos = new TreeMap<>();
        Map<String,Integer> num_neg = new TreeMap<>();
        Map<String,Double> mean_pos =  new TreeMap<>();
        Map<String,Double> mean_neg =  new TreeMap<>();
        Map<String,Integer> temp =  new TreeMap<>();
        
        //Vale tous counters kai to leksiko sta antistoixa meso thetiko k meso arnitiko "dianysma"
        for(String word: dict.keySet())
        {
            int positive = dict.get(word).get("pos");
            int negative = dict.get(word).get("neg");
            num_pos.put(word, positive);
            num_neg.put(word, negative);
            mean_neg.put(word,0.0);
            mean_pos.put(word,0.0);
            temp.put(word,0);
        }
        
        
        //---------------------------------------------------------------------------
        //8a ftia3w meso 8etiko kai meso arnhtiko dianusma
        
        //pairnw ena-ena ola ta 8etika gia na ftia3w to meso 8etiko
        HashSet<String> usedTerms=new HashSet<>();// krataw ena hashset me tous orous pou exei to sugkekrimeno review pou elegxw.
        for (String review : pos_docs) {
            String[] word_list = review.split(" ");
            for(String w: word_list) //metraw poses fores emfanizetai o ka8e o oros mesa sto ka8e review.
            {
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                temp.put(w, temp.get(w)+1);
                usedTerms.add(w);
            }
            for(String s : usedTerms) //gia ton ka8e oro upogizw to varos tou me tupo W=(freq(x)/sumofterms)*log(N/freq(x))
            {
                double count_s=temp.get(s);
                mean_pos.put(s, mean_pos.get(s)+ ((count_s/word_list.length)*(Math.log10(num_pos.get(s)/count_s)))/num_of_docs);
                temp.put(s, 0);
            }
            usedTerms.clear();
        }
        //pairnw ena-ena ola ta arnhtika gia na ftia3w to meso arnhtika
        for (String review : neg_docs) {
            String[] word_list = review.split(" ");
            for(String w: word_list)
            {
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                temp.put(w, temp.get(w)+1);
                usedTerms.add(w);
            }
            for(String s : usedTerms)
            {
                double count_s=temp.get(s);
                mean_neg.put(s, mean_neg.get(s)+ ((count_s/word_list.length)*(Math.log10(num_neg.get(s)/count_s)))/num_of_docs); //TF*IDF
                temp.put(s, 0);
            }
            usedTerms.clear();
        }
        
        

        //Read each TEST review
        int correct = 0;
        int wrong = 0;
        Map<String,Double> test_keep =  new TreeMap<>();
        for (String key : dict.keySet()) {
           test_keep.put(key, 0.0);
        }
        for(String doc: test_docs.keySet())
        {
            Map<String,Double> test =  test_keep;
            usedTerms.clear();
            int count_terms = 0;
            String[] word_list = doc.split(" ");
            for(String w: word_list)
            {
                w = w.replaceAll("[^\\p{L}\\p{N}]+", "");
                if(w.equals("")) continue;
                //If the term is in the word dictionary
                if(dict.containsKey(w)) 
                {
                    count_terms++;
                    //Get frequency of term w
                    test.put(w, test.get(w)+1);
                    usedTerms.add(w);
                }
            }
            
            Map<String,Double> test_pos = test;
            Map<String,Double> test_neg = test;
            for(String s: usedTerms)
            {
                double count_s=test.get(s);
                if(count_s == 0)
                {
                    test_pos.put(s, 0.0);
                    test_neg.put(s, 0.0);
                }
                else
                {
                    Double pos_score = (count_s/count_terms)*(Math.log10(num_pos.get(s)/count_s));
                    Double neg_score = (count_s/count_terms)*(Math.log10(num_neg.get(s)/count_s));
                    if(pos_score==Double.NEGATIVE_INFINITY || pos_score == Double.POSITIVE_INFINITY ||
                       neg_score==Double.NEGATIVE_INFINITY || neg_score == Double.POSITIVE_INFINITY)
                    {
                        test_pos.put(s, 0.0);
                        test_neg.put(s, 0.0);
                    }
                    else
                    {
                        test_pos.put(s, pos_score); //TF*IDF
                        test_neg.put(s, neg_score); //TF*IDF
                    }
                }
            }
            
            Double pos_score = jaccardSimilarity(usedTerms,test_pos,mean_pos);
            Double neg_score = jaccardSimilarity(usedTerms,test_neg,mean_neg);
            if(pos_score>=neg_score)
            {
                String type= test_docs.get(doc);
                if(type.equals("pos"))
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
            }
            else
            {
                String type= test_docs.get(doc);
                if(type.equals("neg"))
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
            }
            
        }
        Double precision = (double)correct/(correct+wrong);
        //---------------------------------------------------------------------------   
        createResults(mean_pos,mean_neg,dict,num_pos, num_neg );
    }
}
