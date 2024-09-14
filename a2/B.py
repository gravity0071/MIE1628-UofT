from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# instantiate the spark session
spark = SparkSession.builder.appName("movies").getOrCreate()
# set the shuffle partition same as number of cpu cores to improve performance
spark.conf.set("spark.sql.shuffle.partitions", 4)
df = spark.read.format("csv").option("inferSchema", True).option("header", True).option("sep", ',').option("path", "movies.csv").load()

# 1
print("part 1")
top12_movies = df.groupby('movieId').mean().orderBy("avg(rating)", ascending = False).select('movieId', "avg(rating)").limit(12)
top12_movies.show()
top12_users = df.groupby('userId').mean().orderBy('avg(rating)', ascending = False).select('userId', "avg(rating)").limit(12)
top12_users.show()

#2
train7_3, test7_3 = df.randomSplit([0.7, 0.3], seed=10)# split the data to 70/30
train9_1, test9_1 = df.randomSplit([0.9, 0.1], seed=10)# split the data to 90/10
train9_1_o = train9_1# the original data
test9_1_o = test9_1
als7_3 = ALS(userCol="userId", itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")
als9_1 = ALS(userCol="userId", itemCol='movieId', ratingCol='rating', coldStartStrategy="drop")

#3
print()
print("part 3")
model7_3 = als7_3.fit(train7_3)
pred7_3 = model7_3.transform(test7_3)#predicted data of 70/30
model9_1 = als9_1.fit(train9_1)
pred9_1 = model9_1.transform(test9_1)#predicted data of 60/40
eval_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

mae7_3 = eval_mae.evaluate(pred7_3)
mae9_1 = eval_mae.evaluate(pred9_1)
print("mae: 70/30: ", mae7_3)
print("mae: 90/10: ", mae9_1)

#4
print()
print("part 4")
paramGrid = ParamGridBuilder() \
    .addGrid(ALS.rank, [5, 10, 15]) \
    .addGrid(ALS.regParam, [0.01, 0.1, 0.5, 2]) \
    .build()

als_tuned = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop"
)

evaluator_tuned = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="mae")
tvs = TrainValidationSplit(estimator=als_tuned, estimatorParamMaps=paramGrid, evaluator=evaluator_tuned, parallelism=1, seed=12)

# Fit the tuned model
tuned_model = tvs.fit(train9_1)
predictions = tuned_model.bestModel.transform(test9_1)
error = evaluator_tuned.evaluate(predictions)

print("for 90/10:")
print("Best Rank: ", tuned_model.bestModel._java_obj.parent().getRank())
print("Max Iteration: ", tuned_model.bestModel._java_obj.parent().getMaxIter())
print("Best RegParam: ", tuned_model.bestModel._java_obj.parent().getRegParam())
print("MAE of test set: ", error)

paramGrid = ParamGridBuilder() \
    .addGrid(ALS.rank, [5, 10, 15]) \
    .addGrid(ALS.regParam, [0.01, 0.1, 0.5, 2]) \
    .build()

als_tuned2 = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop"
)

evaluator_tuned2 = RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="mae")
tvs2 = TrainValidationSplit(estimator=als_tuned2, estimatorParamMaps=paramGrid, evaluator=evaluator_tuned2, parallelism=1, seed=12)

# Fit the tuned model
tuned_model2 = tvs2.fit(train7_3)
predictions2 = tuned_model2.bestModel.transform(test7_3)
error2 = evaluator_tuned2.evaluate(predictions2)
print()
print("for 70/30:")
print("Best Rank: ", tuned_model2.bestModel._java_obj.parent().getRank())
print("Max Iteration: ", tuned_model2.bestModel._java_obj.parent().getMaxIter())
print("Best RegParam: ", tuned_model2.bestModel._java_obj.parent().getRegParam())
print("MAE of test set: ", error2)

#5
recom7_3 = tuned_model2.bestModel.recommendForAllUsers(12)
recom7_3.filter(col('userID') == 10).show(truncate=False)
recom7_3.filter(col('userID') == 12).show(truncate=False)
#for 90/10
recom9_1 = tuned_model.bestModel.recommendForAllUsers(12)
recom9_1.filter(col('userID') == 10).show(truncate=False)
recom9_1.filter(col('userID') == 12).show(truncate=False)
