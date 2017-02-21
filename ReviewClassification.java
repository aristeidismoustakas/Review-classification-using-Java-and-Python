/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.log;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.DoublePoint;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.json.JSONObject;

/**
 *
 * @author swagdam
 */
public class ReviewClassification {

    public static String replaceRegex = "[^a-zA-Z ]+";
    public static int KNN = 0;
    public static int KNN_SENTIMENT = 1;
    public static int COS_SCORE = 2;
    private static SimpleDateFormat ft = new SimpleDateFormat("hh:mm:ss");

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, ParseException, org.apache.lucene.queryparser.classic.ParseException {
        String base_dir = "processed_data";
        
        tlog(ft, "Reading reviews from disk.");
   
        ArrayList<Document> documents = readReviews(base_dir + "/train/neg/", -1);
        documents.addAll(readReviews(base_dir + "/train/pos/", -1));
        ArrayList<Document> query_set = readReviews(base_dir + "/test/", -1);
   
        tlog(ft, "Done reading from disk.");
        
        BooleanQuery.setMaxClauseCount(100000);
        
        tlog(ft, "Starting.");
        
        //accuracyTest(documents, 1, 200, "./accuracy_test_results.txt");
        predictTestSet(documents, query_set, 61, "./cos_score_results.txt");

        tlog(ft, "Done.");
    }

    /**
     * Makes predictions for a test set. Saves them in a hash map that is later written in a file, in ascending file name order. Uses cos_score by default.
     * @param training_set The training set
     * @param query_set The test set
     * @param threshold The threshold used for queries and predictions
     * @param filename The name of the file that holds the results
     * @throws org.apache.lucene.queryparser.classic.ParseException
     * @throws IOException 
     */
    public static void predictTestSet(ArrayList<Document> training_set, ArrayList<Document> query_set, int threshold, String filename) throws org.apache.lucene.queryparser.classic.ParseException, IOException {
        Similarity cos_sim = new ClassicSimilarity();
        FileWriter outfile = new FileWriter(filename);
        HashMap<String, Boolean> predictions = new HashMap<>();
        
        tlog(ft, "Bulding document index.");
        //Lucene stuff, building an analyzer, an index, creating a configuration, making an index writer, writing documents to the index, making a reader, and a searcher from the reader.
        //Setting cosine similarity as the similarity method in the configuration and the searcher.
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = new RAMDirectory();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setSimilarity(cos_sim);

        IndexWriter w = new IndexWriter(index, config);
        addDocuments(w, training_set);
        w.close();

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(cos_sim);
        
        tlog(ft, "Done bulding index. Predicting the test set.");
        
        //For each review in the test set, query the index, get the results, and predict with a given threshold.
        //Then add the prediction to the hash map. The key is the name of the file. We only have the path, so we split it, get the filename, and remove the extension.
        for (Document doc : query_set) {
            ArrayList<Result> results = query(doc, analyzer, searcher, threshold);
            boolean cos_pred = predict(results, doc, threshold, COS_SCORE);
            String[] str = doc.get("path").split("/");
            predictions.put(str[str.length - 1].split("\\.")[0], cos_pred);
        }
        
        //Sort files in file name ascending order
        
        tlog(ft, "Done predicting test set. Sorting files.");
        
        ArrayList<String> files = new ArrayList<>(predictions.keySet());

        files.sort(new Comparator() {
            @Override
            public int compare(Object o1, Object o2) {
                String s1 = (String) o1;
                String s2 = (String) o2;
                return s1.compareTo(s2);
            }
        });
        
        tlog(ft, "Done sorting files. Writing to disk.");
        
        //Write results to disk
        for (String s : files) {
            outfile.write(s + " " + boolToInt(predictions.get(s)) + System.lineSeparator());
        }
        outfile.close();
        
        tlog(ft, "Done writing to disk. Results in: " + filename);
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
     * Uses 9/10 of the training set as the training set, and 1/10 as the test set, chosen randomly.
     * Makes predictions for all 3 scoring methods and for multiple thresholds, to decide the best scoring method and the best threshold to use.
     * @param documents The training set, which will be divided to create a test set
     * @param threshold_start The minimum threshold
     * @param threshold_end The maximum threshold
     * @param filename The name of the file that holds the results
     * @throws IOException
     * @throws org.apache.lucene.queryparser.classic.ParseException 
     */
    public static void accuracyTest(ArrayList<Document> documents, int threshold_start, int threshold_end, String filename) throws IOException, org.apache.lucene.queryparser.classic.ParseException {
        long seed = System.nanoTime();
        Collections.shuffle(documents, new Random(seed));
        FileWriter outfile = new FileWriter(filename);

        //9/10 of the training set is used for training
        //The remaining 1/10 is used for testing
        ArrayList<Document> training_set = new ArrayList<>(documents.subList(0, documents.size() * 9 / 10));
        ArrayList<Document> test_set = new ArrayList<>(documents.subList(documents.size() * 9 / 10, documents.size()));

        //Metrics objects hold tp, fp, tn, and fn counters. We keep one for each threshold. We are testing with 3 scoring methods, so we need 3 lists of objects, 
        //which contain an object for each threshold.
        ArrayList<Integer> threshold_list = new ArrayList<>();
        ArrayList<Metrics> metrics_list_knn = new ArrayList<>();
        ArrayList<Metrics> metrics_list_knn_sentiment = new ArrayList<>();
        ArrayList<Metrics> metrics_list_cos_score = new ArrayList<>();

        //Initializing the metrics objects.
        for (int i = threshold_start; i <= threshold_end; i++) {
            threshold_list.add(i);
            metrics_list_knn.add(new Metrics());
            metrics_list_knn_sentiment.add(new Metrics());
            metrics_list_cos_score.add(new Metrics());
        }

        //Built-in cosine similarity method.
        Similarity cos_sim = new ClassicSimilarity();
        
        tlog(ft, "Bulding document index.");
        //Lucene stuff, building an analyzer, an index, creating a configuration, making an index writer, writing documents to the index, making a reader, and a searcher from the reader.
        //Setting cosine similarity as the similarity method in the configuration and the searcher.
        StandardAnalyzer analyzer = new StandardAnalyzer();
        Directory index = new RAMDirectory();
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setSimilarity(cos_sim);

        IndexWriter w = new IndexWriter(index, config);
        addDocuments(w, training_set);
        w.close();

        IndexReader reader = DirectoryReader.open(index);
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(cos_sim);

        tlog(ft, "Done bulding index. Predicting the test set.");

        //For each review in the test set, query the index, get the results, then predict with a given threshold.
        //Testing for multiple thresholds to find which one to use.
        for (Document doc : test_set) {
            ArrayList<Result> results = query(doc, analyzer, searcher, threshold_list.get(threshold_list.size() - 1));
            boolean query_class = doc.get("path").contains("pos");
            
            //We execute the query only once, then for each threshold count the results with the appropriate metrics object.
            for (int i = 0; i < threshold_list.size(); i++) {
                boolean knn_pred = predict(results, doc, threshold_list.get(i), KNN);
                boolean knn_senti_pred = predict(results, doc, threshold_list.get(i), KNN_SENTIMENT);
                boolean cos_pred = predict(results, doc, threshold_list.get(i), COS_SCORE);

                update_metrics(metrics_list_knn.get(i), query_class, knn_pred);
                update_metrics(metrics_list_knn_sentiment.get(i), query_class, knn_senti_pred);
                update_metrics(metrics_list_cos_score.get(i), query_class, cos_pred);
            }
        }

        tlog(ft, "Done predicting test set. Calculating accuracies and writing to file.");
        
        //For each metrics object we call calculate(), which calculates the accuracy, then write it to file.
        for (int i = 0; i < threshold_list.size(); i++) {
            metrics_list_knn.get(i).calculate();
            metrics_list_knn_sentiment.get(i).calculate();
            metrics_list_cos_score.get(i).calculate();
            outfile.write(threshold_list.get(i) + " " + metrics_list_knn.get(i).getAccuracy() + " " + metrics_list_knn_sentiment.get(i).getAccuracy() + " " + metrics_list_cos_score.get(i).getAccuracy() + System.lineSeparator());
        }

        outfile.close();
        
        tlog(ft, "Done writing to file. Results in: " + filename);
    }

    /**
     * Predicts the class of a query given its results.
     * @param results The results of the query.
     * @param doc The review we want to classify (the query doc).
     * @param threshold The threshold we will use.
     * @param prediction_method Indicates the prediction method used
     * @return True if we predict the class as positive, else false
     */
    public static boolean predict(ArrayList<Result> results, Document doc, int threshold, int prediction_method) {
        int pos = 0;
        double knn_score; // Method 1
        double query_score, pos_score = 0.0, neg_score = 0.0; // Method 2
        double cos_score = 0.0; //Method 3, most effective
        boolean predicted_positive = false;

        int limit = threshold;
        if(limit > results.size()) limit = results.size();
        
        //Pos is the number of positive results, used later to calculate the ratio
        //Cos score is a variable where we add the cosine similarity score of a positive review, or substract the cosine similarity score of a negative review.
        
        for (int j = 0; j < limit; j++) {
            //If the result is a positive review, we increment the pos variable.
            if (results.get(j).getDoc().get("path").contains("pos")) {
                pos++;
                cos_score += results.get(j).getScore();
            } else {
                cos_score -= results.get(j).getScore();
            }
        }

        //Calculating the score of the query based on the SentiWordNet dictionary (query_score)
        //As well as the ratio of positive reviews in the results (knn_score)
        pos_score = doc.getField("pos_score").numericValue().doubleValue();
        neg_score = doc.getField("neg_score").numericValue().doubleValue();
        query_score = (double) (pos_score - neg_score) / (pos_score + neg_score);

        //Converting knn_score from [0,1] to [-1,1]
        knn_score = 2 * ((double) pos / threshold) - 1;

        //Calculate score with given method
        if (prediction_method == COS_SCORE) {
            if (cos_score > 0) {
                predicted_positive = true;
            }
        } else if (prediction_method == KNN_SENTIMENT) {
            if (query_score * 0.2 + knn_score * 0.8 > 0) {
                predicted_positive = true;
            }
        } else if (prediction_method == KNN) {
            if (knn_score > 0) {
                predicted_positive = true;
            }
        }

        return predicted_positive;
    }

    /**
     * Updates a metrics object in the appropriate way, depending on the predicted and the actual class of a review.
     * @param metrics A metrics object, which will update according to the true
     * and predicted classes
     * @param true_class The true class of the query doc
     * @param predicted_class The predicted class of the query doc
     */
    public static void update_metrics(Metrics metrics, boolean true_class, boolean predicted_class) {
        if (predicted_class) {
            //Predicted positive
            if (!true_class) {
                //False positive
                metrics.incFp();
            } else {
                //True positive
                metrics.incTp();
            }
        } else {
            //Predicted negative
            if (!true_class) {
                //True negative
                metrics.incTn();
            } else {
                //False negative
                metrics.incFn();
            }
        }
    }

    /**
     * Executes a query and returns the results.
     * @param doc The review we want to classify
     * @param analyzer Lucene's analyzer, needed to construct a query
     * @param searcher Lucene's searcher, searches in the index
     * @param threshold K, for top-k results.
     * @return A list with the results of the query.
     * @throws org.apache.lucene.queryparser.classic.ParseException
     * @throws IOException
     */
    public static ArrayList<Result> query(Document doc, StandardAnalyzer analyzer, IndexSearcher searcher, int threshold) throws org.apache.lucene.queryparser.classic.ParseException, IOException {
        //Create a query after escaping some special characters
        String querystr = doc.get("text").toLowerCase().replaceAll(replaceRegex, "");
        Query q = new QueryParser("text", analyzer).parse(QueryParser.escape(querystr));

        //The returned list
        ArrayList<Result> results = new ArrayList<>();

        //Search for the query with a given threshold, and get the results in a ScoreDoc array
        TopDocs docs = searcher.search(q, threshold);
        ScoreDoc[] hits = docs.scoreDocs;

        //Add the results to the list and return
        for (int i = 0; i < hits.length; i++) {
            results.add(new Result(searcher.doc(hits[i].doc), hits[i].score));
        }

        return results;
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
     * Writes documents to the index.
     * @param w Lucene's index writer
     * @param documents Documents to write in the index
     * @throws IOException 
     */
    public static void addDocuments(IndexWriter w, ArrayList<Document> documents) throws IOException {
        for (Document doc : documents) {
            w.addDocument(doc);
        }
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

        //For each file in the directory, read the file, and create a Lucene document where we store
        //the text, the positive and negative scores, and the path.
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
