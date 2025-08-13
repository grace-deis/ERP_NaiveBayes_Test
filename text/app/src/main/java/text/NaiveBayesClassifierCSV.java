package text;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import com.opencsv.CSVReader;

import java.nio.charset.StandardCharsets;

/**
 * Naive Bayes classifier that can be trained from a CSV (text,label).
 * Provides train, predict, and CSV utility methods.
 */
public class NaiveBayesClassifierCSV {

    /** Holds a training sample. */
    public static class Sample {
        public final String text;
        public final String label;
        public Sample(String text, String label) {
            this.text = text;
            this.label = label;
        }
    }

    /** Holds a trained classifier and all necessary mappings. */
    public static class TrainedClassifier {
        public final NaiveBayesModel model;
        public final List<String> vocab;
        public final Map<String, Integer> labelMap;
        public final Map<String, Integer> vocabDf;
        public final int trainDocCount;
        public TrainedClassifier(NaiveBayesModel model, List<String> vocab, Map<String, Integer> labelMap, Map<String, Integer> vocabDf, int trainDocCount) {
            this.model = model;
            this.vocab = vocab;
            this.labelMap = labelMap;
            this.vocabDf = vocabDf;
            this.trainDocCount = trainDocCount;
        }
    }

    // --- Core Training ---
    public static TrainedClassifier train(List<Sample> samples) {
        if (samples.size() < 2) throw new IllegalArgumentException("Need at least 2 samples");
        List<String> vocab = buildVocab(samples);
        Map<String, Integer> df = documentFrequency(samples, vocab);
        double[][] X = vectorizeTfIdf(samples, vocab, df);
        Map<String, Integer> labelMap = encodeLabels(samples);
        int[] y = encodeLabelsArray(samples, labelMap);
        NaiveBayesModel nb = trainNaiveBayes(X, y, labelMap.size());
        return new TrainedClassifier(nb, vocab, labelMap, df, samples.size());
    }

    // --- Prediction ---
    public static String predict(TrainedClassifier tc, String text) {
        double[] x = vectorizeTfIdf(text, tc.vocab, tc.vocabDf, tc.trainDocCount);
        int predIdx = tc.model.predict(x);
        return inverseLabelMap(tc.labelMap, predIdx);
    }

    // --- CSV helpers ---
    public static List<Sample> loadSamplesFromCsv(String csvPath) throws IOException, com.opencsv.exceptions.CsvValidationException {
        List<Sample> samples = new ArrayList<>();
        try (
            FileReader fr = new FileReader(csvPath, StandardCharsets.UTF_8);
            CSVReader reader = new CSVReader(fr)
        ) {
            String[] line;
            boolean isFirstLine = true;
            while ((line = reader.readNext()) != null) {
                if (isFirstLine) { // skip header
                    isFirstLine = false;
                    continue;
                }
                if (line.length < 2) continue; // skip incomplete rows
                String text = line[0];
                String label = line[1];
                if (label == null || label.trim().isEmpty()) continue; // skip blank label
                if (text == null || text.trim().isEmpty()) continue; // skip blank text
                samples.add(new Sample(text, label));
            }
        }
        return samples;
    }

    private static String[] parseCsvLine(String line) {
        List<String> result = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;
        for (char c : line.toCharArray()) {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes) {
                result.add(sb.toString());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        result.add(sb.toString());
        return result.toArray(new String[0]);
    }

    // --- Text vectorization ---
    private static List<String> buildVocab(List<Sample> samples) {
        Set<String> vocabSet = new HashSet<>();
        for (Sample sample : samples) {
            for (String token : tokenize(sample.text)) {
                vocabSet.add(token);
            }
        }
        List<String> vocab = new ArrayList<>(vocabSet);
        Collections.sort(vocab);
        return vocab;
    }

    private static Map<String, Integer> documentFrequency(List<Sample> samples, List<String> vocab) {
        Map<String, Integer> df = new HashMap<>();
        for (String v : vocab) df.put(v, 0);
        for (Sample sample : samples) {
            Set<String> tokensInDoc = new HashSet<>(Arrays.asList(tokenize(sample.text)));
            for (String token : tokensInDoc) {
                if (df.containsKey(token)) {
                    df.put(token, df.getOrDefault(token, 0) + 1);
                }
            }
        }
        return df;
    }

    public static double[][] vectorizeTfIdf(List<Sample> samples, List<String> vocab, Map<String, Integer> df) {
        int N = samples.size();
        double[][] X = new double[N][vocab.size()];
        for (int i = 0; i < N; i++) {
            String[] tokens = tokenize(samples.get(i).text);
            Map<String, Integer> tf = new HashMap<>();
            for (String token : tokens) tf.put(token, tf.getOrDefault(token, 0) + 1);
            for (int j = 0; j < vocab.size(); j++) {
                String term = vocab.get(j);
                int tfTerm = tf.getOrDefault(term, 0);
                int dfTerm = df.getOrDefault(term, 1);
                double idf = Math.log((double) N / (dfTerm));
                X[i][j] = tfTerm * idf;
            }
        }
        return X;
    }

    public static double[] vectorizeTfIdf(String sentence, List<String> vocab, Map<String, Integer> df, int N) {
        String[] tokens = tokenize(sentence);
        Map<String, Integer> tf = new HashMap<>();
        for (String token : tokens) tf.put(token, tf.getOrDefault(token, 0) + 1);
        double[] x = new double[vocab.size()];
        for (int j = 0; j < vocab.size(); j++) {
            String term = vocab.get(j);
            int tfTerm = tf.getOrDefault(term, 0);
            int dfTerm = df.getOrDefault(term, 1);
            double idf = Math.log((double) N / (dfTerm));
            x[j] = tfTerm * idf;
        }
        return x;
    }

    public static double predictLabelProbability(TrainedClassifier tc, String text, String label) {
        double[] x = vectorizeTfIdf(text, tc.vocab, tc.vocabDf, tc.trainDocCount);
        double[] probs = tc.model.predictProba(x);
        Integer idx = tc.labelMap.get(label);
        if (idx == null) return 0.0;
        return probs[idx];
    }

    public static String[] tokenize(String text) {
        String[] rawTokens = text.toLowerCase().replaceAll("[^a-z0-9 ]", " ").split("\\s+");
        PorterStemmer stemmer = new PorterStemmer();
        ArrayList<String> stems = new ArrayList<>();
        for (String token : rawTokens) {
            if (token.isEmpty()) continue;
            String stem = stemmer.stem(token);
            if (!stem.isEmpty()) stems.add(stem);
        }
        return stems.toArray(new String[0]);
    }

    // --- Label encoding ---
    private static Map<String, Integer> encodeLabels(List<Sample> samples) {
        Map<String, Integer> labelMap = new HashMap<>();
        int idx = 0;
        for (Sample s : samples) {
            if (!labelMap.containsKey(s.label)) {
                labelMap.put(s.label, idx++);
            }
        }
        return labelMap;
    }

    private static int[] encodeLabelsArray(List<Sample> samples, Map<String, Integer> labelMap) {
        int[] y = new int[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            y[i] = labelMap.get(samples.get(i).label);
        }
        return y;
    }

    public static String inverseLabelMap(Map<String, Integer> labelMap, int idx) {
        for (Map.Entry<String, Integer> entry : labelMap.entrySet()) {
            if (entry.getValue() == idx) return entry.getKey();
        }
        return null;
    }



    // --- Naive Bayes training ---
    public static NaiveBayesModel trainNaiveBayes(double[][] X, int[] y, int nClasses) {
        int N = X.length;
        int D = X[0].length;
        double alpha = 1.0; // Laplace smoothing

        double[] classCounts = new double[nClasses];
        double[][] featureSums = new double[nClasses][D];
        for (int i = 0; i < N; i++) {
            int cls = y[i];
            classCounts[cls] += 1;
            for (int j = 0; j < D; j++) {
                featureSums[cls][j] += X[i][j];
            }
        }

        double[] logPrior = new double[nClasses];
        double[][] logLikelihood = new double[nClasses][D];
        for (int c = 0; c < nClasses; c++) {
            logPrior[c] = Math.log(classCounts[c] / N);
            double sum = 0.0;
            for (int j = 0; j < D; j++)
                sum += featureSums[c][j] + alpha;
            for (int j = 0; j < D; j++) {
                logLikelihood[c][j] = Math.log((featureSums[c][j] + alpha) / sum);
            }
        }
        return new NaiveBayesModel(logPrior, logLikelihood);
    }
}