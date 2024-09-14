package kmeans;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans8 extends Configured implements Tool {
    private static int maxIter = 15; //maximum iterations = 15
    private static int K = 8; //default k = 5

    public static class Point {
        private double x;
        private double y;
        public Point() {
        }
        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
        public Point(String line) {
            String[] Position = line.split(",");
            this.x = Double.parseDouble(Position[0]);
            this.y = Double.parseDouble(Position[1]);
        }
        public double getX() { return this.x; }
        public double getY() { return this.y; }
        public static double calcDistance(Point x, Point y) {
            return Math.sqrt(Math.pow((x.getX() - y.getX()), 2) + Math.pow((x.getY() - y.getY()), 2));
        }
        public String toString() { return this.x + "," + this.y; }
    }

    public static class FileHandler {
        public static void writeToFile(List<Point> points, Configuration conf) throws IOException {
            Path centroidPath = new Path(conf.get("centroid.path"));
            FileSystem fs = FileSystem.get(conf);

            if (fs.exists(centroidPath)) {
                fs.delete(centroidPath, true);
            }

            final SequenceFile.Writer centerWriter = SequenceFile.createWriter(fs, conf, centroidPath, Text.class, IntWritable.class);
            final IntWritable value = new IntWritable(0);
            for (Point point : points) {
                centerWriter.append(new Text(point.toString()), value);
            }
            centerWriter.close();
        }

        public static List<Point> readFromFile(Configuration conf) throws IOException {
            Path centerPath = new Path(conf.get("centroid.path"));
            ArrayList<Point> pointList = new ArrayList<>();

            FileSystem fs = FileSystem.get(conf);
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, centerPath, conf);
            Text key = new Text();
            IntWritable value = new IntWritable();
            while (reader.next(key, value)) {
                pointList.add(new Point(key.toString()));
            }
            reader.close();
            return pointList;
        }

        public static boolean compareCenter(List<Point> oldCenter, List<Point> newCenter) {
            double sum = 0;
            for (int i = 0; i < oldCenter.size(); i++) {
                double tmp = Point.calcDistance(oldCenter.get(i), newCenter.get(i));
                sum += tmp;
            }
            if (sum < 0.1) {
                System.out.println("-----------converged------------");
                return true;
            } else {
                System.out.println("-----------distance sum is: " + sum + "-----------");
                return false;
            }
        }

        private static List<Point> randomCenterInit() {
            List<Point> initList = new ArrayList<>();
            Random random = new Random();
            double min = 0.0;
            double max = 100.0;

            for (int i = 0; i < K; i++) {
                double num1 = min + (max - min) * random.nextDouble();
                double num2 = min + (max - min) * random.nextDouble();
                initList.add(new Point(num1, num2));
            }
            return initList;
        }
    }

    public static class PointsMapper extends Mapper<LongWritable, Text, Text, Text> {
        public List<Point> centers = new ArrayList<>();

        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            ArrayList<Point> pointList = new ArrayList<>();
            Configuration conf = context.getConfiguration();

            Path centerPath = new Path(conf.get("centroid.path"));
            FileSystem fs = FileSystem.get(conf);
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, centerPath, conf);

            Text key = new Text();
            IntWritable value = new IntWritable();
            while (reader.next(key, value)) {
                pointList.add(new Point(key.toString()));
            }
            reader.close();
            this.centers = pointList;
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Point thisPoint = new Point(value.toString());
            int id = 0;
            double min = Double.MAX_VALUE;
            double dis = 0;

            for (int i = 0; i < centers.size(); i++) {
                dis = Point.calcDistance(thisPoint, centers.get(i));
                if (dis < min) {
                    min = dis;
                    id = i;
                }
            }
            context.write(new Text(Integer.toString(id)), new Text(thisPoint.toString()));
        }
    }

    public static class PointsReducer extends Reducer<Text, Text, Text, Text> {
        public List<Point> new_centers = new ArrayList<>();

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0;
            double sumY = 0;
            int pc = 0;

            for (Text value : values) {
                String line = value.toString();
                Point point = new Point(line);
                sumX += point.getX();
                sumY += point.getY();
                pc++;
            }
            double Xs = sumX / pc;
            double Ys = sumY / pc;
            Point newCenter = new Point(Xs, Ys);
            new_centers.add(newCenter);
            context.write(key, new Text(newCenter.toString()));
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            super.cleanup(context);
            Configuration config = context.getConfiguration();
            FileHandler.writeToFile(this.new_centers, config);
        }
    }

    public static void main(String[] args) throws Exception {
        int times = 1;
        Configuration conf = new Configuration();

        Path center_path = new Path("hdfs://127.0.0.1:9000/centroid/cen.seq");
        conf.set("centroid.path", center_path.toString());

        long start = System.currentTimeMillis();

        List<Point> old_centers = new ArrayList<>();
        List<Point> new_centers = new ArrayList<>();

        FileHandler.writeToFile(FileHandler.randomCenterInit(), conf);

        while (times <= maxIter) {
            old_centers = FileHandler.readFromFile(conf);
            ToolRunner.run(conf, new KMeans5(), args);
            new_centers = FileHandler.readFromFile(conf);

            if (FileHandler.compareCenter(old_centers, new_centers)) {
                break;
            } else {
                System.out.println("---iterate for: ---" + times);
                times++;
            }
        }
        long end = System.currentTimeMillis();
        long duration = (end - start) / 1000;
        System.out.println("Total running time: " + duration);
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        Job job = Job.getInstance(conf, "KMeans");

        FileSystem fs = FileSystem.get(conf);
        Path out = new Path(args[1]);

        job.setJarByClass(KMeans5.class);
        job.setMapperClass(PointsMapper.class);
        job.setReducerClass(PointsReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        if (fs.exists(out)) {
            fs.delete(out, true);
        }
        FileOutputFormat.setOutputPath(job, out);

        return job.waitForCompletion(true) ? 0 : 1;
    }
}
