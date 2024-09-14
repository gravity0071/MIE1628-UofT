from pyspark.sql import SparkSession
import re

# instantiate the spark session
spark = SparkSession.builder.appName("Part_A").getOrCreate()

#1
print("part1:")
intlistRDD = spark.sparkContext.textFile("integer.txt") # may need to replace to the other position
even = intlistRDD.filter(lambda x: int(x) % 2 == 0)
print('Num of evens: {}'.format(even.count()))
odd = intlistRDD.filter(lambda x: int(x) % 2 == 1)
print('Num of odds: {}\n'.format(odd.count()))

#2
print("part2:")
salaryRDD = spark.sparkContext.textFile("salary.txt")
arrayRDD = salaryRDD.map(lambda x: x.split(" "))
kvRDD = arrayRDD.map(lambda x: (x[0], int(x[1])))
sumRDD = kvRDD.reduceByKey(lambda x, y: x+y)
print(sumRDD.collect())

#3
print("part3:")
RDD = spark.sparkContext.textFile("shakespeare-1.txt")
tRDD = RDD.flatMap(lambda x: re.split('\W+', x)) # split the file to words
words = ["Shakespeare", "When", "Lord", "Library", "GUTENBERG", "WILLIAM", "COLLEGE", "WORLD"]

wordsRDD = tRDD.filter(lambda x: x in words).map(lambda x: (x, 1))
# print("Filtered Words RDD:")
# print(wordsRDD.collect())
sumRDD = wordsRDD.reduceByKey(lambda x, y: x + y)
print(sumRDD.collect())

#4
print("part4:")
w15RDD = tRDD.map(lambda x: (x, 1))
sumRDD = w15RDD.reduceByKey(lambda x, y: x+y)
sumRDD = sumRDD.filter(lambda x: x[0] != '')
topRDD = sumRDD.sortBy(lambda x: x[1], ascending=False)
topelements = topRDD.take(15)
print('top 15:')
for i, element in enumerate(topelements, start=1):
    print('{}: {}'.format(i, element))

lesRDD = sumRDD.sortBy(lambda x: x[1], ascending=True)
leselements = lesRDD.take(15)
print('Least 15:')
for i, element in enumerate(leselements, start=1):
    print('{}: {}'.format(i, element))