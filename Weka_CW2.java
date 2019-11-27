import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Utils;
import java.util.Random;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class Weka_CW2 {
	private Instances fullData;
	private Instances trainingData;
	private Instances testData;

	public Weka_CW2(Instances data) {
		try {

			fullData = data;

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws Exception {

		FileHandler file_handle;
		Logger lg = Logger.getLogger(Weka_CW2.class.getName());

		try {

			String file_attributes_csv = "C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\x_train_gr_smpl.csv";
			String file_class_csv = "C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\y_train_smpl.csv";
			String file_merged_arff = "C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\full_merged.arff";
			String file_training_arff = "C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\training.arff";
			String file_test_arff = "C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\test.arff";
			
			file_handle = new FileHandler("C:\\Users\\Jyotish Thomas\\eclipse-workspace\\F21DL_DMML_CW2\\data\\LogFile.txt");
			lg.addHandler(file_handle);
			SimpleFormatter form = new SimpleFormatter();
			file_handle.setFormatter(form);

			// read and merge attributes and class csv files
			Instances data_attributes = Read_CSV(file_attributes_csv);
			lg.info("Read attributes csv file");
			Instances data_class = Read_CSV(file_class_csv);
			lg.info("Read class csv file");
			Instances data = Instances.mergeInstances(data_attributes, data_class);
			// Setting class attribute
			data.setClassIndex(data.numAttributes() - 1);
			
			lg.info("Merged attributes and class to a single dataset");

			// apply instance randomise filter
			final Randomize Randfilter = new Randomize();
			Randfilter.setInputFormat(data);
			data = Filter.useFilter(data, Randfilter);
			lg.info("Applied Weka Randomizer filter");
			lg.info(Randfilter.toString());			

			Weka_CW2 wkCW2 = new Weka_CW2(data);

			// save arff file
			wkCW2.Create_Arff(file_merged_arff, data);
			
			lg.info("Converted merged instance into arff format (full_merged.arff) suitable for Weka");

			Instances Copy_of_data = wkCW2.NtoNFilter();
			lg.info("Applied Weka Numeric to Nominal filter");
			
			//split full data into training and test data (70/30)
			wkCW2.trainingData = wkCW2.Split_Dataset(30, false, Copy_of_data); // training dataset
			lg.info("Applied Weka RemovePercentage filter with 30% and invert=false to create 70% training dataset");
			wkCW2.Create_Arff(file_training_arff, wkCW2.trainingData);
			wkCW2.testData = wkCW2.Split_Dataset(30, true, Copy_of_data); // test dataset
			wkCW2.Create_Arff(file_test_arff, wkCW2.testData);
			lg.info("Applied Weka RemovePercentage filter with 30% and invert=true to create 30% test dataset");
			
			String Options = null;
			
			//RandomForest rndForest = wkCW2.Classifier_RandForest();
			
			Options = "-C 0.25 -M 2"; //Default Options
			//Options = "-C 0.25 -B -M 2"; //Options with Binary Splits
			
			J48 j48 = wkCW2.Classifier_J48(Options);
			
			Evaluation eval = new Evaluation(wkCW2.trainingData); // evaluating using training data
			// Evaluate J48 Model
			
			eval = wkCW2.Evaluate_Classifier_CrossValidate(eval, j48);
			//eval.crossValidateModel(j48, wkCW2.trainingData, 10, new Random(1));
			lg.info("Applied Weka Evaluation using J48 Classifier default model; training dataset and 10 fold cross validation");
			lg.info(j48.toString());
			lg.info(eval.toSummaryString());
			lg.info(eval.toMatrixString());
			
			/*
			// Test RandomForest Model
			
			lg.info("Applied Weka Evaluation using J48 Classifier based model; training dataset and 10 fold cross validation");
			lg.info(eval.toSummaryString());
			lg.info(eval.toMatrixString());
			lg.info(j48.toString());
			*/
		
		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	private void Create_Arff(String dest_file_arff, Instances data) throws Exception {
		try {

			ArffSaver saver = new ArffSaver();
			saver.setInstances(data);
			saver.setFile(new File(dest_file_arff));
			saver.writeBatch();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	// function to read csv file and return dataset.
	private static Instances Read_CSV(String file) throws Exception {
		Instances dataSet = null;
		try {

			CSVLoader loader = new CSVLoader();
			loader.setSource(new File(file));
			dataSet = loader.getDataSet();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return dataSet;
	}

	//function to get J48 Classifier
	private J48 Classifier_J48(String options) throws Exception {

		J48 j48 = new J48();

		try {
			
			// Create classifier
			j48.setOptions(Utils.splitOptions(options)); 
			j48.buildClassifier(trainingData);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		return j48;

	}
	
	//function to get RandomForest Classifier
	private RandomForest Classifier_RandForest() throws Exception {

		RandomForest rndForest = new RandomForest();

		try {
			
			// Create classifier
			rndForest.setOptions(Utils.splitOptions("-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1")); 
			rndForest.buildClassifier(trainingData);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		return rndForest;

	}

	//function to evaluate Classifier
	private Evaluation Evaluate_Classifier_CrossValidate(Evaluation eval, Classifier obj1 ) throws Exception {

	try {
			eval.crossValidateModel(obj1, trainingData, 10, new Random(1));
		} catch (Exception e) {
			e.printStackTrace();
		}
		return eval;
	}
	
	
	// function to apply Numeric to Nominal filter
	private Instances NtoNFilter() throws Exception {
		Instances newdata = null;
		try {

			// apply numeric to nominal filter on class attribute
			final NumericToNominal NtNfilter = new NumericToNominal();
			String options1[] = { "-R", "first-last" };
			NtNfilter.setOptions(options1);
			NtNfilter.setInputFormat(fullData);
			newdata = Filter.useFilter(fullData, NtNfilter);

		} catch (Exception e) {
			e.printStackTrace();
		}
		return newdata;
	}

	// function to split dataset by percentage
	private Instances Split_Dataset(double percentage, boolean invert, Instances data) throws Exception {
		Instances split_data = null;
		try {

			// split data into training and test data
			RemovePercentage RemPercentage = new RemovePercentage();
			RemPercentage.setInvertSelection(invert);
			RemPercentage.setPercentage(percentage);
			RemPercentage.setInputFormat(data);
			split_data = Filter.useFilter(data, RemPercentage);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return split_data;
	}

}