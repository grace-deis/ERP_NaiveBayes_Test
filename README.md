This code tests the output of the Naive Bayes classifier I implemented in DNA software. It uses a java programme to make predictions on the data using Naive Bayes and outputs a csv of predictions. These csvs are then assessed for accuracy using the .ipynb.

I edited the code to give me several testing scenarios for the data. Firstly, training on annotated statements and testing how well the model could distinguish between statements and nonstatements.

Then, I edited the DNA code to train on statements and nonstatements, only showing predictions for fields other than concept when concept probability is greater than 0.7 and it is a statement. This is where the code "FilterConceptAndPredict" comes in handy. It returns a filtered file of only statements where concept probability is greater than 0.7 and the predicted labels for the specified groups.

I also tested a stemming method which did not ultimately add any improvement.
