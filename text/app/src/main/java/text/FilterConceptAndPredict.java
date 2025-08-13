package text;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Because the DNA model only returns variables other than concept, if the concept class probability is >= 0.7,
 * we need to filter the concept predictions and write them to a new CSV to get a better feeling of the accuracy of the results displayed in DNA.
 * 
 * This program:
 * Loads a CSV with columns: text, concept, organization (change to appropriate variable names if needed).
 * - Splits into train and test sets (80/20 split, random seed 42)
 * - Trains concept and organization models from train set (excluding concept == "not_statement")
 * - For each row in the test set, predicts concept label and probability
 * - Only writes rows to output if concept_predprob >= 0.7 AND concept != "not_statement"
 * - Outputs: text, concept, organization, pred_concept, concept_predprob, pred_organization
 */
public class FilterConceptAndPredict {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: java text.FilterConceptAndPredictOrganization input.csv output.csv");
            System.exit(1);
        }

        String inputCsv = args[0];
        String outputCsv = args[1];

        // Load all samples from input CSV
        List<NaiveBayesClassifierCSV.Sample> allConceptSamples = new ArrayList<>();
        List<NaiveBayesClassifierCSV.Sample> allOrganizationSamples = new ArrayList<>();
        List<String[]> allRows = new ArrayList<>();

        // Read CSV and build samples for both models
        try (CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(inputCsv), StandardCharsets.UTF_8))) {
            String[] header = reader.readNext();
            if (header == null) throw new IOException("No header in input CSV");
            int textCol = -1, conceptCol = -1, orgCol = -1;
            for (int i = 0; i < header.length; i++) {
                String h = header[i].trim().toLowerCase();
                if (h.equals("text")) textCol = i;
                if (h.equals("concept")) conceptCol = i;
                if (h.equals("right")) orgCol = i;
            }
            if (textCol == -1 || conceptCol == -1 || orgCol == -1) {
                throw new RuntimeException("Missing one of required columns: text, concept, right");
            }

            String[] row;
            while ((row = reader.readNext()) != null) {
                allRows.add(row);
                String text = row[textCol];
                String concept = row[conceptCol];
                String organization = row[orgCol];
                // Only use for training if concept != "not_statement"
                if (!"not_statement".equalsIgnoreCase(concept.trim())) {
                    allConceptSamples.add(new NaiveBayesClassifierCSV.Sample(text, concept));
                    allOrganizationSamples.add(new NaiveBayesClassifierCSV.Sample(text, organization));
                }
            }
        }

        // -------------- SPLIT INTO TRAIN AND TEST SETS -------------
        // Shuffle the indices for reproducibility
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < allConceptSamples.size(); i++) indices.add(i);
        Collections.shuffle(indices, new Random(42)); // fixed seed for reproducibility

        int trainSize = (int) (allConceptSamples.size() * 0.8);
        List<NaiveBayesClassifierCSV.Sample> trainConceptSamples = new ArrayList<>();
        List<NaiveBayesClassifierCSV.Sample> testConceptSamples = new ArrayList<>();
        List<NaiveBayesClassifierCSV.Sample> trainOrganizationSamples = new ArrayList<>();
        List<NaiveBayesClassifierCSV.Sample> testOrganizationSamples = new ArrayList<>();

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (i < trainSize) {
                trainConceptSamples.add(allConceptSamples.get(idx));
                trainOrganizationSamples.add(allOrganizationSamples.get(idx));
            } else {
                testConceptSamples.add(allConceptSamples.get(idx));
                testOrganizationSamples.add(allOrganizationSamples.get(idx));
            }
        }

        // Train both models
        NaiveBayesClassifierCSV.TrainedClassifier conceptTC = NaiveBayesClassifierCSV.train(trainConceptSamples);
        NaiveBayesClassifierCSV.TrainedClassifier orgTC = NaiveBayesClassifierCSV.train(trainOrganizationSamples);

        // -------------- EVALUATE ON TEST SET AND WRITE OUTPUT --------------
        try (CSVWriter writer = new CSVWriter(new OutputStreamWriter(new FileOutputStream(outputCsv), StandardCharsets.UTF_8))) {
            writer.writeNext(new String[]{"text", "concept", "right", "pred_concept", "concept_predprob", "pred_right"});
            for (int i = 0; i < testConceptSamples.size(); i++) {
                NaiveBayesClassifierCSV.Sample testConcept = testConceptSamples.get(i);
                NaiveBayesClassifierCSV.Sample testOrganization = testOrganizationSamples.get(i);
                String text = testConcept.text;
                String trueConcept = testConcept.label;
                String trueOrg = testOrganization.label;

                // Skip if true concept is "not_statement"
                if ("not_statement".equalsIgnoreCase(trueConcept.trim())) continue;

                // Predict concept
                String predConcept = NaiveBayesClassifierCSV.predict(conceptTC, text);
                double conceptProb = NaiveBayesClassifierCSV.predictLabelProbability(conceptTC, text, predConcept);

                if (conceptProb >= 0.7) {
                    String predOrg = NaiveBayesClassifierCSV.predict(orgTC, text);
                    writer.writeNext(new String[]{
                        text,
                        trueConcept,
                        trueOrg,
                        predConcept,
                        String.format(Locale.US, "%.4f", conceptProb),
                        predOrg
                    });
                }
                // Otherwise: skip row
            }
        }

        System.out.println("Done. Test set results written to " + outputCsv);
    }
}