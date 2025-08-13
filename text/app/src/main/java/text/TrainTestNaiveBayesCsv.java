package text;

import java.io.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
import com.opencsv.CSVWriter;

public class TrainTestNaiveBayesCsv {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage:\n"
                + "  java test.TrainTestNaiveBayesCsv input.csv output.csv [test_ratio]\n"
                + "  java test.TrainTestNaiveBayesCsv train.csv test.csv output.csv");
            System.exit(1);
        }

        List<NaiveBayesClassifierCSV.Sample> trainSamples;
        List<NaiveBayesClassifierCSV.Sample> testSamples;
        String outputCsv;

        if (args.length == 3 && isNumeric(args[2])) {
            // ratio mode
            String inputCsv = args[0];
            outputCsv = args[1];
            double testRatio = Double.parseDouble(args[2]);

            List<NaiveBayesClassifierCSV.Sample> samples = NaiveBayesClassifierCSV.loadSamplesFromCsv(inputCsv);
            long seed = 42;
            Collections.shuffle(samples, new Random(seed));
            int testCount = (int) Math.round(samples.size() * testRatio);
            testSamples = samples.subList(0, testCount);
            trainSamples = samples.subList(testCount, samples.size());
        } else if (args.length == 3) {
            // explicit train/test/output mode
            String trainCsv = args[0];
            String testCsv = args[1];
            outputCsv = args[2];

            List<NaiveBayesClassifierCSV.Sample> allTrainSamples = NaiveBayesClassifierCSV.loadSamplesFromCsv(trainCsv);
            // Use only 80% of the training data
            long seed = 42;
            Collections.shuffle(allTrainSamples, new Random(seed));
            int useCount = (int) Math.round(allTrainSamples.size() * 0.8);
            trainSamples = allTrainSamples.subList(0, useCount);

            testSamples = NaiveBayesClassifierCSV.loadSamplesFromCsv(testCsv);

            Set<String> trainTexts = new HashSet<>();
            for (NaiveBayesClassifierCSV.Sample s : trainSamples) {
                trainTexts.add(s.text.trim());
            }
            List<NaiveBayesClassifierCSV.Sample> filteredTest = new ArrayList<>();
            for (NaiveBayesClassifierCSV.Sample s : testSamples) {
                if (!trainTexts.contains(s.text.trim())) {
                    filteredTest.add(s);
                }
            }
            testSamples = filteredTest;
            System.out.println("Used 80% of training data: " + trainSamples.size() + " samples.");
            System.out.println("Removed duplicates: test set reduced to " + testSamples.size());
        } else if (args.length == 2) {
            // ratio mode with default split
            String inputCsv = args[0];
            outputCsv = args[1];
            double testRatio = 0.2;

            List<NaiveBayesClassifierCSV.Sample> samples = NaiveBayesClassifierCSV.loadSamplesFromCsv(inputCsv);
            long seed = 42;
            Collections.shuffle(samples, new Random(seed));
            int testCount = (int) Math.round(samples.size() * testRatio);
            testSamples = samples.subList(0, testCount);
            trainSamples = samples.subList(testCount, samples.size());
        } else {
            System.err.println("Too many arguments.");
            System.exit(1);
            return;
        }

        NaiveBayesClassifierCSV.TrainedClassifier tc = NaiveBayesClassifierCSV.train(trainSamples);

        // Use comma as delimiter for output
        try (CSVWriter writer = new CSVWriter(
                new OutputStreamWriter(new FileOutputStream(outputCsv), StandardCharsets.UTF_8),
                ',', // comma delimiter
                CSVWriter.DEFAULT_QUOTE_CHARACTER,
                CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                CSVWriter.DEFAULT_LINE_END)) {
            writer.writeNext(new String[] {"text", "pred_label", "true_label"});
            for (NaiveBayesClassifierCSV.Sample sample : testSamples) {
                String pred = NaiveBayesClassifierCSV.predict(tc, sample.text);
                writer.writeNext(new String[] {cleanText(sample.text), pred, sample.label});
            }
        }
        System.out.println("Done. Test set size: " + testSamples.size() + ". Output written to " + outputCsv);
    }

    private static boolean isNumeric(String s) {
        try {
            Double.parseDouble(s);
            return true;
        } catch (NumberFormatException ex) {
            return false;
        }
    }

    private static String cleanText(String text) {
        return text
            .replace("â€™", "'")
            .replace("â€œ", "\"")
            .replace("â€�", "\"")
            .replace("â€“", "-")
            .replace("â€”", "—")
            .replace("â€˜", "'");
    }
}