/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.json.JSONObject;

/**
 *
 * @author swagdam
 */
public class BayesClassifier {
    private static SimpleDateFormat ft = new SimpleDateFormat("hh:mm:ss");

    public static void main(String[] args) throws IOException {
        String base_dir = "processed_data";
        
        tlog(ft, "Reading reviews from disk.");
        
        ArrayList<Document> training_set = readReviews(base_dir + "/train/neg/", -1);
        training_set.addAll(readReviews(base_dir + "/train/pos/", -1));
        ArrayList<Document> query_set = readReviews(base_dir + "/test/", -1);
        
        tlog(ft, "Done reading from disk.");
        
        tlog(ft, "Starting.");
        //accuracyTest(training_set);
        predictQuerySet(training_set, query_set, "bayes_results.txt");
        tlog(ft, "Done.");
    }
    
    /**
     * Predicts the test set, sorts it with ascending file name order and writes to disk.
     * @param training_set The training set
     * @param query_set The test set
     * @param filename The name of the file which holds the results
     * @throws IOException 
     */
    public static void predictQuerySet(ArrayList<Document> training_set, ArrayList<Document> query_set, String filename) throws IOException {
        FileWriter outfile = new FileWriter(filename);
        
        //Predict and get predictions
        tlog(ft, "Predicting.");
        HashMap<String, Boolean> predictions = predict(training_set, query_set);
        tlog(ft, "Done predicting.");
        
        //Sort by file name, ascending
        tlog(ft, "Sorting.");
        ArrayList<String> files = new ArrayList<>(predictions.keySet());
        files.sort(new Comparator() {
            @Override
            public int compare(Object o1, Object o2) {
                String s1 = (String) o1;
                String s2 = (String) o2;
                
                return s1.compareTo(s2);
            } 
        });
        
        //Split path to get file name, then write file name and predicted class to disk.
        tlog(ft, "Done sorting, writing to file.");
        for(String s : files) {
            String[] str = s.split("/");
            String id = str[str.length - 1].split("\\.")[0];
            
            outfile.write(id + " " + boolToInt(predictions.get(s)) + System.lineSeparator());
        }
        
        outfile.close();
        
        tlog(ft, "All done. Results in: " + filename);
    }
    
    /**
     * 
     * @param val A boolean value
     * @return 1 if val == true, else 0
     */
    public static int boolToInt(boolean val) {
        if (val) {
            return 1;
        }
        return 0;
    }
    
    /**
     * Predicts the the test set based on the training set.
     * @param training_set The training set
     * @param test_set The test set
     * @return A hash map containing file paths and predicted class.
     */
    public static HashMap<String, Boolean> predict(ArrayList<Document> training_set, ArrayList<Document> test_set) {
        //HashMap containing the predictions.
        HashMap<String, Boolean> predictions = new HashMap<>();
        
        //Indexes containing words and occurences in positive and negative reviews respectively.
        HashMap<String, Integer> pos_index = new HashMap<>();
        HashMap<String, Integer> neg_index = new HashMap<>();
        
        //Number of positive and negative reviews.
        double pos_reviews = 0, neg_reviews = 0;
        
        tlog(ft, "Building word indexes.");
        for (Document doc : training_set) {
            boolean pos = doc.get("path").contains("pos");
            
            //Increment positive or negative review counter
            if(pos) pos_reviews++;
            else neg_reviews++;
            
            //Keep unique words
            HashSet<String> word_set = new HashSet<>();
            
            for(String word : doc.get("text").split(" ")) {
                if(!word.trim().equals("")) word_set.add(word);
            }
            
            //For each unique word, if it isn't in the index add it with 0 appearances
            //Else increment the counter in the index. Pos index for positive reviews, neg index for negative reviews.
            for (String word : word_set) {                
                if(!pos_index.containsKey(word)) pos_index.put(word, 0);
                if(!neg_index.containsKey(word)) neg_index.put(word, 0);

                if (pos) {
                    pos_index.put(word, pos_index.get(word)+1);
                } else {
                    neg_index.put(word, neg_index.get(word)+1);
                }                
            }
        }
        
        tlog(ft, "Done building word indexes. Predicting the test set.");
        
        //Probabilities for each review
        double pos_prob, neg_prob;
        
        //For each review in test set, calculate pos_prob and neg_prob, and choose the class of the bigger one.
        for(Document doc : test_set) {
            pos_prob = 1.0;
            neg_prob = 1.0;
            
            //For each word in the review, multiply the probability with the number of appearences divided by the number of reviews of the class.
            //Words with 0 appearances will make our probability zero, which is bad. Instead, we say it has 1 appearance.
            //Actually multiplies each probability with p(w_i|c_i)
            for(String word : doc.get("text").split(" ")) {
                if(pos_index.containsKey(word)) {
                    if(pos_index.get(word) != 0) pos_prob *= (double) pos_index.get(word)/pos_reviews;
                    else pos_prob *= (double) 1 / pos_reviews;
                }
                if(neg_index.containsKey(word)) {
                    if(neg_index.get(word) != 0) neg_prob *= (double) neg_index.get(word)/neg_reviews;
                    else neg_prob *= (double) 1 / neg_reviews;
                }
            }
             
            //Predict positive if pos_prob > neg_prob, else predict negative.
            if(pos_prob > neg_prob) {
                predictions.put(doc.get("path"), true);
            }
            else {
                predictions.put(doc.get("path"), false);
            }
        }
        
        return predictions;
    }
    
    /**
     * Uses 9/10 of the training set as the training set, and 1/10 as the test set, chosen randomly.
     * Prints the accuracy of the model at the end.
     * @param documents The training set, which will be divided to produce the test set 
     */
    public static void accuracyTest(ArrayList<Document> documents) {
        long seed = System.nanoTime();
        //Shuffle the documents, so that each time the training and test sets are different
        Collections.shuffle(documents, new Random(seed));
        
        ArrayList<Document> training_set = new ArrayList<>(documents.subList(0, documents.size() * 9 / 10));
        ArrayList<Document> test_set = new ArrayList<>(documents.subList(documents.size() * 9 / 10, documents.size()));

        //Metrics object to hold tp, tn, fp, fn.
        Metrics metrics = new Metrics();
        
        //Predict and get the predictions
        HashMap<String, Boolean> predictions = predict(training_set, test_set);
        
        //Sort by file name
        ArrayList<String> files = new ArrayList<>(predictions.keySet());
        files.sort(new Comparator() {
            @Override
            public int compare(Object o1, Object o2) {
                String s1 = (String) o1;
                String s2 = (String) o2;
                
                return s1.compareTo(s2);
            } 
        });
        
        //For each file, find if it is positive or negative and update the metrics object.
        for(String s : files) {
            if(predictions.get(s)) {
                if(s.contains("pos")) {
                    //Predicted positive, is positive
                    metrics.incTp();
                }
                else {
                    //Predicted positive, is negative
                    metrics.incFp();
                }
            }
            else {
                if(s.contains("pos")) {
                    //Predicted negative, is positive
                    metrics.incFn();
                }
                else {
                    //Predicted negative, is negative
                    metrics.incTn();
                }
            }
        }
        
        //Calculate accuracy and print.
        metrics.calculate();
        System.out.println("Accuracy: " + metrics.getAccuracy());
    }
    
    /**
     * Reads a file from the disk
     * @param path Path of file to read
     * @param encoding Encoding, UTF-8
     * @return A string containing the text of the read file
     * @throws java.io.IOException 
     */
    public static String readFile(String path, Charset encoding) throws java.io.IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }

    /**
     * Reads reviews from a given directory.
     * @param directory The directory of the documents
     * @param limit A limit on the number of documents read, -1 for as many as there are
     * @return A list with the read documents
     * @throws IOException 
     */
    public static ArrayList<Document> readReviews(String directory, int limit) throws IOException {
        ArrayList<Document> documents = new ArrayList<>();

        int i = 0;

        File dir = new File(directory);
        File[] files = dir.listFiles();

        for (File f : files) {
            String text = readFile(f.getAbsolutePath(), Charset.forName("UTF-8"));

            JSONObject jobj = new JSONObject(text);
            Document doc = new Document();
            doc.add(new StringField("path", f.getAbsolutePath(), Field.Store.YES));
            doc.add(new TextField("text", jobj.getString("text"), Field.Store.YES));
            doc.add(new DoublePoint("pos_score", jobj.getDouble("pos_score")));
            doc.add(new DoublePoint("neg_score", jobj.getDouble("neg_score")));

            documents.add(doc);
            i++;
            if (i == limit) {
                return documents;
            }
        }

        return documents;
    }
    
    /**
     * Prints a string with a timestamp.
     * @param ft A simple date format
     * @param str A string to print
     */
    public static void tlog(SimpleDateFormat ft, String str) {
        System.out.println("[" + ft.format(new Date()) + "] " + str);
    }
}
