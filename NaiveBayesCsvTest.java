import java.io.*;
import java.util.*;

/**
 * Test harness for NaiveBayesClassifier using a CSV of text and labels.
 * CSV format: text,label
 */
public class NaiveBayesCsvTest {

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java NaiveBayesCsvTest <csv_file>");
            System.exit(1);
        }
        String csvFile = args[0];

        // Step 1: Load samples from CSV
        List<Sample> samples = loadSamplesFromCsv(csvFile);

        // Step 2: Train classifier (single group test)
        // We use all samples as a single Variable group for this isolated test
        List<String> vocab = buildVocab(samples);
        Map<String, Integer> df = documentFrequency(samples, vocab);
        double[][] X = NaiveBayesClassifier.vectorizeTfIdf(samples, vocab, df);
        Map<String, Integer> labelMap = encodeLabels(samples);
        int[] y = encodeLabelsArray(samples, labelMap);

        NaiveBayesClassifier.NaiveBayesModel model =
            invokeTrainNaiveBayes(X, y, labelMap.size());

        // Step 3: Evaluate on the same data (or split for real test)
        int correct = 0;
        for (int i = 0; i < samples.size(); i++) {
            double[] xi = NaiveBayesClassifier.vectorizeTfIdf(
                samples.get(i).text, vocab, df, samples.size());
            int predIdx = model.predict(xi);
            String pred = inverseLabelMap(labelMap, predIdx);
            if (pred.equals(samples.get(i).label)) correct++;
        }
        double acc = (double) correct / samples.size();
        System.out.printf("Accuracy: %.3f (%d/%d)\n", acc, correct, samples.size());
    }

    /** Loads samples from a CSV. Assumes header row, columns: text,label */
    private static List<Sample> loadSamplesFromCsv(String csvFile) throws IOException {
        List<Sample> samples = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line = br.readLine(); // skip header
            while ((line = br.readLine()) != null) {
                String[] parts = parseCsvLine(line);
                if (parts.length >= 2) {
                    String text = parts[0];
                    String label = parts[1];
                    if (!text.trim().isEmpty() && !label.trim().isEmpty()) {
                        samples.add(new Sample(text, label));
                    }
                }
            }
        }
        return samples;
    }

    /** Naive CSV parser for 2 columns (handles commas inside quotes) */
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

    // --- Standalone versions of helpers from NaiveBayesClassifier ---

    private static List<String> buildVocab(List<Sample> samples) {
        Set<String> vocabSet = new HashSet<>();
        for (Sample sample : samples) {
            for (String token : NaiveBayesClassifier.tokenize(sample.text)) {
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
            Set<String> tokensInDoc = new HashSet<>(Arrays.asList(NaiveBayesClassifier.tokenize(sample.text)));
            for (String token : tokensInDoc) {
                if (df.containsKey(token)) {
                    df.put(token, df.getOrDefault(token, 0) + 1);
                }
            }
        }
        return df;
    }

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

    // Call the private method using reflection (if needed)
    private static NaiveBayesClassifier.NaiveBayesModel invokeTrainNaiveBayes(
            double[][] X, int[] y, int nClasses) throws Exception {
        try {
            java.lang.reflect.Method m = NaiveBayesClassifier.class.getDeclaredMethod(
                "trainNaiveBayes", double[][].class, int[].class, int.class);
            m.setAccessible(true);
            Object result = m.invoke(null, X, y, nClasses);
            return (NaiveBayesClassifier.NaiveBayesModel) result;
        } catch (NoSuchMethodException nsme) {
            // If method is public, just call it
            return null;
        }
    }

    private static String inverseLabelMap(Map<String, Integer> labelMap, int idx) {
        for (Map.Entry<String, Integer> entry : labelMap.entrySet()) {
            if (entry.getValue() == idx) return entry.getKey();
        }
        return null;
    }
}