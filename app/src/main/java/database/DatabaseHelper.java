package database;


import android.content.ContentValues;
import android.database.sqlite.SQLiteDatabase;
import android.util.Log;


import utility.Constants;
import utility.FrameData;
import utility.Trace;


public class DatabaseHelper {

    // Logcat tag
    private static final String TAG = "DatabaseHelper";

    private SQLiteDatabase db_ = null;


    // Table Names
    private static final String TABLE_ACCELEROMETER = "accelerometer";
    private static final String TABLE_GYROSCOPE = "gyroscope";
    private static final String TABLE_MAGNETOMETER = "magnetometer";
    private static final String TABLE_ROTATION_MATRIX = "rotation_matrix";
    private static final String TABLE_GPS = "gps";
    private static final String TABLE_LATENCY = "latency";


    private static final String KEY_TIME = "time";


    private static final String frameSendTime = "frameSendTime";
    private static final String transmitSequence = "transmitSequence";
    private static final String roundLatency = "roundLatency";
    private static final String originalSize = "originalSize";
    private static final String serverTime = "serverTime";
    private static final String compressedDataSize = "compressedDataSize";
    private static final String isIFrame = "isIFrame";
    private static final String rawFrameIndex = "rawFrameIndex";


    /*rotation matrix*/
    private static final String KEY_VALUES[] = {"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"};


    // Table Create Statements

    private static final String CREATE_TABLE_GPS = "CREATE TABLE IF NOT EXISTS "
            + TABLE_GPS + "(" + KEY_TIME + " INTEGER PRIMARY KEY," + KEY_VALUES[0] + " REAL,"
            + KEY_VALUES[1] + " REAL," +  KEY_VALUES[2] + " REAL" + ");";
    private static final String CREATE_TABLE_ACCELEROMETER = "CREATE TABLE IF NOT EXISTS "
            + TABLE_ACCELEROMETER + "(" + KEY_TIME + " INTEGER PRIMARY KEY," + KEY_VALUES[0]
            + " REAL," + KEY_VALUES[1] + " REAL," +  KEY_VALUES[2] + " REAL" + ");";
    private static final String CREATE_TABLE_GYROSCOPE = "CREATE TABLE IF NOT EXISTS "
            + TABLE_GYROSCOPE + "(" + KEY_TIME + " INTEGER PRIMARY KEY," + KEY_VALUES[0]
            + " REAL," + KEY_VALUES[1] + " REAL," +  KEY_VALUES[2] + " REAL" + ");";
    private static final String CREATE_TABLE_MAGNETOMETER = "CREATE TABLE IF NOT EXISTS "
            + TABLE_MAGNETOMETER + "(" + KEY_TIME + " INTEGER PRIMARY KEY," + KEY_VALUES[0]
            + " REAL," + KEY_VALUES[1] + " REAL," +  KEY_VALUES[2] + " REAL" + ");";


    private static final String CREATE_TABLE_LATENCY = "CREATE TABLE IF NOT EXISTS "
            + TABLE_LATENCY + "(" + transmitSequence + " INTEGER PRIMARY KEY,"
            + frameSendTime + " INTEGER," +  roundLatency + " REAL,"
            + originalSize + " INTEGER," + serverTime + " INTEGER," +  compressedDataSize + " INTEGER,"
            + isIFrame + " INTEGER, " + rawFrameIndex + " INTEGER )";

    private static final String CREATE_TABLE_ROTATION_MATRIX = "CREATE TABLE IF NOT EXISTS "
            + TABLE_ROTATION_MATRIX + "(" + KEY_TIME + " INTEGER PRIMARY KEY,"
            + KEY_VALUES[0] + " REAL," + KEY_VALUES[1] + " REAL," +  KEY_VALUES[2] + " REAL,"
            + KEY_VALUES[3] + " REAL," + KEY_VALUES[4] + " REAL," +  KEY_VALUES[5] + " REAL,"
            + KEY_VALUES[6] + " REAL," + KEY_VALUES[7] + " REAL," +  KEY_VALUES[8] + " REAL"
            + ")";

    private boolean opened = false;
    // public interfaces
    public DatabaseHelper() {
        this.opened = true;
    }


    //open and close for each trip
    public void createDatabase(long t) {
        this.opened = true;
        db_ = SQLiteDatabase.openOrCreateDatabase(Constants.kDBFolder + String.valueOf(t).concat(".db"), null, null);
        db_.execSQL(CREATE_TABLE_ACCELEROMETER);
        db_.execSQL(CREATE_TABLE_GYROSCOPE);
        db_.execSQL(CREATE_TABLE_MAGNETOMETER);
        db_.execSQL(CREATE_TABLE_GPS);
        db_.execSQL(CREATE_TABLE_ROTATION_MATRIX);
        db_.execSQL(CREATE_TABLE_LATENCY);
    }


    public void closeDatabase() {
        this.opened = false;
        if(db_ != null && db_.isOpen()) {
            db_.close();
        }
    }
    public boolean isOpen() {
        return this.opened;
    }

    //private String[] dataName = {"frameSendTime","transmitSequence","roundLatency","oraginalSize"};

    public void insertFrameData(FrameData frameData) {
            ContentValues values = new ContentValues();
            values.put(frameSendTime, frameData.getFrameSendTime());
            values.put(transmitSequence, frameData.transmitSequence);
            values.put(roundLatency, frameData.roundLatency);
            values.put(originalSize, frameData.originalDataSize);
            values.put(serverTime, frameData.serverTime);
            values.put(compressedDataSize, frameData.compressedDataSize);
            values.put(isIFrame, frameData.isIFrame);
            values.put(rawFrameIndex, frameData.rawFrameIndex);
            db_.insert(TABLE_LATENCY, null, values);
    }


    public int updateFrameData(FrameData updatedFrameData) {
        // Log.d(TAG, "updateFrameData");

        ContentValues args = new ContentValues();
        args.put(roundLatency, updatedFrameData.roundLatency);
        //Log.d(TAG,"round latency:" + roundLatency);
        args.put(serverTime,updatedFrameData.serverTime);

        String where = " transmitSequence = ? ";
        String[] whereArgs = {String.valueOf(updatedFrameData.transmitSequence)};
        return db_.update(TABLE_LATENCY, args, where, whereArgs);
    }

    public void insertSensorData(Trace trace) {
        String type = trace.type;
        ContentValues values = new ContentValues();
        values.put(KEY_TIME, trace.time);
        for(int i = 0; i < trace.dim; ++i) {
            values.put(KEY_VALUES[i], trace.values[i]);
        }

        if (type.equals(Trace.ROTATION_MATRIX)) {
            db_.insert(TABLE_ROTATION_MATRIX, null, values);
        } else if (type.equals(Trace.ACCELEROMETER)) {
            db_.insert(TABLE_ACCELEROMETER, null, values);
        } else if (type.equals(Trace.GYROSCOPE)) {
            db_.insert(TABLE_GYROSCOPE, null, values);
        } else if (type.equals(Trace.MAGNETOMETER)) {
            db_.insert(TABLE_MAGNETOMETER, null, values);
        } else if (type.equals(Trace.GPS)) {
            db_.insert(TABLE_GPS, null, values);
        }  else {
            assert 0 == 1;
        }
    }




}
