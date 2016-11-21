

import javax.management.relation.Role

import org.apache
import org.apache.hadoop.yarn.webapp.hamlet.HamletSpec.{A, B}
import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._


/**
  * kaggle https://www.kaggle.com/c/unimelb/  with spark 2.0.2
  * Created by kotobotov on 18.11.2016.
  */


object KaggleGrand {
    def main(args: Array[String]) = {

//preparing Spark envirenment

        val spark = SparkSession
                .builder()
                .appName("Kaggle example")
                .master("local[*]")
                .getOrCreate()


        import spark.implicits._

// loading data
                //from path according to your needs

        val df = spark.read
                .option("delimiter", "\t")
                .option("header", "true")
//                .option("inferSchema", "true")
                .csv("C:\\inetpub\\play\\kaggle\\src\\main\\resources\\grantsPeople.csv")

// prepare data to able use by spark

        val fituredData = df.withColumn("Ci", df("Role").equalTo("CHIEF_INVESTIGATOR").cast("Int"))
                .withColumn("phd", df("With_PHD").equalTo("Yes").cast("Int"))
                .withColumn("paperscore", df("A2") * 4 + df("A") * 3)


// final case class Grand(Grant_Application_ID: String,RFCD_Code:String, RFCD_Percentage:String,  SEO_Code:String, SEO_Percentage:String, Person_ID:String,                 Role:String, Year_of_Birth:String, Country_of_Birth:String, Home_Language:String,  Dept_No:String, Faculty_No:String, With_PHD: String, No_of_Years_in_Uni_at_Time_of_Grant: String, Number_of_Successful_Grant: String, Number_of_Unsuccessful_Grant: String,   A2: String,    A: String,    B: String,    C: String, Grant_Status: String, Sponsor_Code:String,  Contract_Value_Band:String, Grant_Category_Code:String)
// val ds = df.as[Grand]
// its better work with datastream instead dataframe, but today we will be use DF

        //prepare data to able use by spark

        val grants = fituredData.groupBy("Grant_Application_ID").agg(
            max("Grant_status").as("Grant_Status"),
            max("Grant_Category_Code").as("Category_Code"),
            max("Contract_Value_Band").as("Value_Band"),
            when(sum("phd").isNull, 0).otherwise(sum("phd")).as("PHDs"),
            when(max(expr("paperscore * CI")).isNull, 0)
                    .otherwise(max(expr("paperscore * CI"))).as("paperscore"),
            count("*").as("teamsize"),
            when(sum("Number_of_Successful_Grant").isNull, 0).
                    otherwise(sum("Number_of_Successful_Grant")).as("successes"),
            when(sum("Number_of_Unsuccessful_Grant").isNull, 0).
                    otherwise(sum("Number_of_Unsuccessful_Grant")).as("failures")
        )

//        grants.show(10)  // - to see what we have in data


// convert data to fiture vector which is able to use by estimator

        val value_band = new StringIndexer().setInputCol("Value_Band").setOutputCol("Value_Index").fit(grants)
        val category_indexer = new StringIndexer().setInputCol("Category_Code").setOutputCol("Category_Index").fit(grants)
        val label_indexer = new StringIndexer().setInputCol("Grant_Status").setOutputCol("status").fit(grants)
        val assembler = new VectorAssembler().setInputCols(Array("Value_Index", "Category_Index","PHDs",  "paperscore", "teamsize", "successes", "failures")).setOutputCol("assembled")

//        set up our ML model
        val model =new RandomForestClassifier().setFeaturesCol("assembled").setLabelCol("status").setSeed(42)

        // set up our piplene

        val pipline = new Pipeline().setStages(Array(value_band, category_indexer, label_indexer, assembler, model))

        // set up evaluater

        val auc_eval = new BinaryClassificationEvaluator().setLabelCol("status").setRawPredictionCol("rawPrediction")

        System.out.println(auc_eval.getMetricName)

//        split traning i data set
        val training = grants.filter("Grant_Application_ID < 6635")
        val test = grants.filter("Grant_Application_ID >= 6635")

//        start our pipline, train model and transform (predict) data

        val pipline_result = pipline.fit(training)
                .transform(test)


grants.show(30)

        System.out.println("evaluating our result as area onder ROC curve : ")
        System.out.println(auc_eval.evaluate(pipline_result))

    }
}
