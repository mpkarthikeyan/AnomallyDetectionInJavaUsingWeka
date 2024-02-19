package org.example;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

public class Main {

    private static int dataSetLength = 0;
    public static void main(String[] args) throws Exception{
        //failedMethod();

        Instances newData = loadDataFromArffFile("/Users/karthikeyanpandian/Documents/workspace/bayesSample/src/main/resources/trainingData.arff");

        Instances testData = loadDataFromArffFile("/Users/karthikeyanpandian/Documents/workspace/bayesSample/src/main/resources/testingData.arff");

        Classifier classifier = new NaiveBayes();

        classifier.buildClassifier(newData);



        Evaluation evaluation = new Evaluation(newData);
        evaluation.crossValidateModel(classifier, testData, 2, new Random(1));

        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());

        System.out.println("++++++++++++ test ++++++++++++");
        Instance newInstance = new DenseInstance(newData.numAttributes());
        newInstance.setDataset(newData);

        // Set attribute values for the new instance
     System.out.println("test started");

        // Step 4: Use the trained model to predict the class label
        double prediction = classifier.classifyInstance(newInstance);
        String predictedClass = newData.classAttribute().value((int) prediction);

        // Optionally, analyze prediction probability or confidence level
        double[] probabilities = classifier.distributionForInstance(newInstance);
        System.out.println("Predicted class: " + predictedClass);
        System.out.println("Prediction probabilities: " + java.util.Arrays.toString(probabilities));

        // Step 5: Analyze prediction to determine if it's an anomaly
        boolean isAnomaly = predictedClass.equals("anomaly");
        System.out.println("Is anomaly? " + isAnomaly);
        System.out.println("test ends");
    }


    private static void decorateTestDatainInstance(Instance newInstance, Instances newData){
        String method = "PUT";
        String url = "/api/update";
        String headers = "<script>";
        double payloadSize = 64;

        newInstance.setValue(newData.attribute("method"), method);
        newInstance.setValue(newData.attribute("url"), url);
        newInstance.setValue(newData.attribute("headers"), headers);
        newInstance.setValue(newData.attribute("payload_size"), payloadSize);


    }

    private static Instances loadDataFromArffFile(String fileNameWithPath) throws Exception {
        ArffLoader arffLoader = new ArffLoader();
        arffLoader.setFile(new File(fileNameWithPath));
        Instances dataSet = arffLoader.getDataSet();
        dataSetLength = dataSet.numAttributes();
        dataSet.setClassIndex(dataSet.numAttributes() -1);


        // Step 3: Convert string attributes to word vectors
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataSet);
        Instances newData = Filter.useFilter(dataSet, filter);
        return newData;
    }




    private static void failedMethod() throws Exception {
        System.out.println("Hello world!");

        BufferedReader reader = new BufferedReader(new FileReader("/Users/karthikeyanpandian/Documents/workspace/bayesSample/src/main/resources/trainingData.arff"));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
        Instances data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1); // Set the class attribute



        // Train classifier
        Classifier classifier = new NaiveBayes();
        classifier.buildClassifier(data);

        // Save the model to a file
        weka.core.SerializationHelper.write("trained_model.model", classifier);
    }
}