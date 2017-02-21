/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

import org.apache.lucene.document.Document;

/**
 *
 * @author swagdam
 */
public class Result {
    private Document doc;
    private float score;
    
    Result(Document doc, float score) {
        this.doc = doc;
        this.score = score;
    }

    public Document getDoc() {
        return doc;
    }

    public float getScore() {
        return score;
    }
    
}
