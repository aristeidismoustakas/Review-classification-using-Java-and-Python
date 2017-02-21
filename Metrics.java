/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package reviewclassification;

/**
 *
 * @author swagdam
 */
public class Metrics {
    private int tp, tn, fp, fn;
    private double accuracy;

    Metrics() {
        tp = 0;
        tn = 0;
        fp = 0;
        fn = 0;
    }

    public void incTp() {
        tp+=1;
    }
    
    public void incTn() {
        tn+=1;
    }
    
    public void incFp() {
        fp+=1;
    }
    
    public void incFn() {
        fn+=1;
    }
    
    public void calculate() {
        this.accuracy = (double) (this.tp+this.tn)/(this.tp+this.tn+this.fp+this.fn);
    }

    public double getAccuracy() {
        return accuracy;
    }

    public int getTp() {
        return tp;
    }

    public int getTn() {
        return tn;
    }

    public int getFp() {
        return fp;
    }

    public int getFn() {
        return fn;
    }
    
    
}
