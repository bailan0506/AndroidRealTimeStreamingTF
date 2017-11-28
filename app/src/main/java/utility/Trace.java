package utility;


import android.util.JsonReader;
import android.util.JsonWriter;
import android.util.Log;

import java.io.Serializable;
import java.io.StringReader;
import java.io.StringWriter;

public class Trace implements Serializable {

    public long time;
    public double [] values = null;
    public int dim;
    public String type = "none";

    private static String TAG = Trace.class.getSimpleName();

    public static String ACCELEROMETER = "accelerometer";
    public static String GYROSCOPE = "gyroscope";
    public static String MAGNETOMETER = "magnetometer";

    public static String ROTATION_MATRIX = "rotation_matrix";
    public static String GPS = "gps";


    public Trace() {

    }

    public Trace(int d) {
        time = 0;
        dim = d;
        values = new double [dim];
    }

    public void setValues(double x, double y, double z) {
        values[0] = x;
        values[1] = y;
        values[2] = z;
        dim = 3;
    }


    public void copyTrace(Trace trace) {
        this.time = trace.time;
        this.dim = trace.dim;
        this.values = new double[dim];
        for(int i = 0; i < dim; ++i) {
            this.values[i] = trace.values[i];
        }
    }

    public String toJson() {
        StringWriter sw = new StringWriter();
        JsonWriter writer = new JsonWriter(sw);

        try {
            writer.beginObject();
            writer.name("type").value(type);
            writer.name("timeStamp").value(time);
            writer.name("dim").value(dim);
            for (int i = 0; i < dim; ++i) {
                writer.name("x" + String.valueOf(i)).value(values[i]);
            }
            writer.endObject();
            writer.flush();
        } catch (Exception e) {
            Log.d(TAG, "convert to json failed");
        }
        return sw.toString();
    }

    public void fromJson(String json) {
        StringReader sr = new StringReader(json);
        JsonReader reader = new JsonReader(sr);
        try {
            reader.beginObject();
            while (reader.hasNext()) {
                String name = reader.nextName();
                if (name.equals("type")) {
                    type = reader.nextString();
                } else if (name.equals("timeStamp")) {
                    time = reader.nextLong();
                } else if (name.equals("dim")) {
                    dim = reader.nextInt();
                    values = new double[dim];
                } else if (name.contains("x")) {
                    int index = Integer.valueOf(name.substring(1)).intValue();
                    values[index] = (float)reader.nextDouble();
                } else {
                    reader.skipValue();
                }
            }
            reader.endObject();
        } catch (Exception e) {
            Log.d(TAG, "read to json failed");
        }
    }


}