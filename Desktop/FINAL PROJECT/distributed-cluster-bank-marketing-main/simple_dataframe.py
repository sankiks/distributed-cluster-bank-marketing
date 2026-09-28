from pyspark.sql import SparkSession

if __name__=="__main__":
    # Initialize a spark session
    spark = SparkSession.builder.appName("Simple Spark DataFrame Example").getOrCreate()

    # Create a DataFrame from a list of numbers
    numbers_list= [1,2,3,4,5]
    numbers_df=spark.createDataFrame([(t,) for t in numbers_list], ["number"])

    # Show the DataFrame
    numbers_df.show()
    # Stop the Spark session
    spark.stop()