package utility;

import android.os.Environment;

/**
 * Created by lkang on 3/29/16.
 */
public class Constants {

    /*for GPS*/
    public static final double kSmallEPSILON = 1e-8;
    public static final double kEarthRadius = 6371 * 1000; /*m*/

    public static final double kMeterToMile = 0.000621371;
    public static final double kMeterPSToMilePH = 2.23694;
    public static final double kKmPHToMPH = 0.621371;
    public static final double kKmPHToMeterPS = 0.277778;

    public static final double kMileToMeters = 1609.34;

    public static final String kInputSeperator = "\t";
    public static final String kOutputSeperator = "\t";

    public static final double kRecordingInterval = 100;


    public static final String kPackageName = "streaming";
    //public static final String kDBFolder = "/data/data/" + kPackageName + "/databases/";
    public static final String kDBFolder = Environment.getExternalStorageDirectory().getAbsolutePath().toString()+ "/databases/";
    //public static final String kVideoFolder = "/data/data/" + kPackageName + "/videos/";
    public static final String kVideoFolder = Environment.getExternalStorageDirectory().getAbsolutePath().toString()+ "/videos/";

}
