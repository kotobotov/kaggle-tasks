import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.SparkSqlParser
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.functions._
import sun.security.krb5.internal.Ticket

/**
  * Created by dimasik on 23.11.2016.
  * kaggle https://www.kaggle.com/c/titanic/  with spark 2.0.2
  */
object KaggleTitanic {
case class Passager(PassengerId: Int,Survived: Int, Pclass: Int, Name:String, Sex: String,Age: Double, SibSp:Int, Parch:Int,   Ticket:String, Fare:Double, Cabin:String = "A", Embarked:String)
    def main(args: Array[String])={
        val schema = StructType(Array(
            StructField("PassengerId",       DataTypes.IntegerType),
            StructField("Survived",       DataTypes.IntegerType),
            StructField("Pclass",       DataTypes.IntegerType),
            StructField("Name",       DataTypes.StringType),
            StructField("Sex",       DataTypes.StringType),
            StructField("Age",       DataTypes.DoubleType),
            StructField("SibSp",       DataTypes.IntegerType),
            StructField("Parch",       DataTypes.IntegerType),
            StructField("Ticket",       DataTypes.StringType),
            StructField("Fare",       DataTypes.DoubleType),
            StructField("Cabin",       DataTypes.StringType),
            StructField("Embarked",       DataTypes.StringType)
        ))


val sc = SparkSession.builder().appName("kaggle_titanic").master("local[*]").getOrCreate()
      val row_data = sc.read
              .schema(schema)
                      .option("header", true)
//              .option("inferSchema", "true")
        .csv("C:\\inetpub\\play\\kaggle\\src\\main\\resources\\train_titanic.csv")
        row_data.printSchema()

//              .map(attributes => Person(attributes(0), attributes(1).trim.toInt))


        import sc.implicits._
val trasTicket = udf{(tkt:String)=> tkt.trim
            .replaceAll("""[\d*]""", "")
}

        val trasCabin = udf{(tkt:String)=> tkt.charAt(0).toString
//                .replaceAll("""[\d*]""", "")
        }
        val trasAge = udf{(name:String) =>
            if (name.contains("Mr.")) "30.0"
            if (name.contains("Miss.")) "18.0"
            if (name.contains("Mrs.")) "30.0"
else "20.0"
//
        }

        val ds = row_data.as[Passager]
//                .filter(_.Survived==1)
//                .filter(_.Age.isEmpty)
//                .where("Cabin")
                .withColumn("TiketF",  trasTicket($"Ticket"))
                .withColumn("AgeF", when(($"Age").isNull, trasAge($"Name")).otherwise($"Age"))
                .withColumn("CabinF", when(($"Cabin").isNull, "").otherwise(trasCabin($"Cabin")))
//                .sort("Fare")
//                .show(800)

//        val stringAge = new StringIndexer().setInputCol("AgeF").setOutputCol("prepAge").fit(row_data)
        val stringSex = new StringIndexer().setInputCol("Sex").setOutputCol("prepSex").fit(row_data)

        val fitureVector = new VectorAssembler().setInputCols(Array("prepAge")).setOutputCol("fiture")
        val model = new RandomForestClassifier().setSeed(42).setFeaturesCol("fiture").setLabelCol("Survived")
        val evaluator = new BinaryClassificationEvaluator().setLabelCol("Survived").setRawPredictionCol("rowPridiction")

        val pipeline = new Pipeline().setStages(Array(stringAge, stringSex,  fitureVector, model))


        pipeline.fit(ds)
       System.out.println( ds.count())
//        ds.map(item =>"name"+item.Name)
    }

}
