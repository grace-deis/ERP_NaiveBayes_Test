package text;
import text.PorterStemmer;

public class NaiveBayesModel {
    public final double[] logPrior;
    public final double[][] logLikelihood;
    

    public NaiveBayesModel(double[] logPrior, double[][] logLikelihood) {
        this.logPrior = logPrior;
        this.logLikelihood = logLikelihood;
    }

    public int predict(double[] x) {
        double best = Double.NEGATIVE_INFINITY;
        int bestClass = -1;
        for (int c = 0; c < logPrior.length; c++) {
            double score = logPrior[c];
            for (int j = 0; j < x.length; j++) {
                score += x[j] * logLikelihood[c][j];
            }
            if (score > best) {
                best = score;
                bestClass = c;
            }
        }
        return bestClass;
    }

    public double[] predictProba(double[] x) {
        double[] logProbs = new double[logPrior.length];
        double maxLogProb = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < logPrior.length; c++) {
            double score = logPrior[c];
            for (int j = 0; j < x.length; j++) {
                score += x[j] * logLikelihood[c][j];
            }
            logProbs[c] = score;
            if (score > maxLogProb) maxLogProb = score;
        }
        // Softmax normalization for numerical stability
        double sum = 0.0;
        for (int c = 0; c < logPrior.length; c++) {
            logProbs[c] = Math.exp(logProbs[c] - maxLogProb);
            sum += logProbs[c];
        }
        for (int c = 0; c < logPrior.length; c++) {
            logProbs[c] /= sum;
        }
        return logProbs;
    }

    /**
     * Fisher's method for combining likelihoods into a single statistic.
     * Returns the Fisher statistic for each class.
     * 
     * @param x the feature vector
     * @return an array of Fisher statistics, one per class
     */
    
    public double[] fisherScore(double[] x) {
        double[] scores = new double[logPrior.length];
        for (int c = 0; c < logPrior.length; c++) {
            double fisher = 0.0;
            for (int j = 0; j < x.length; j++) {
                if (x[j] == 0) continue; // Only consider present features
                // Compute likelihood as exp(logLikelihood)
                double prob = Math.exp(logLikelihood[c][j]);
                // Avoid log(0)
                if (prob > 0) {
                    fisher += -2.0 * Math.log(prob);
                }
            }
            scores[c] = fisher;
        }
        return scores;
    }

    /**
     * Predict using Fisher's method: return the class with the lowest Fisher statistic.
     * 
     * @param x the feature vector
     * @return the predicted class index
     */
    
    public int predictFisher(double[] x) {
        double[] fisherScores = fisherScore(x);
        double best = Double.POSITIVE_INFINITY;
        int bestClass = -1;
        for (int c = 0; c < fisherScores.length; c++) {
            if (fisherScores[c] < best) {
                best = fisherScores[c];
                bestClass = c;
            }
        }
        return bestClass;
    }
}