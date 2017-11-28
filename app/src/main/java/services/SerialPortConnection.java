package services;

import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;

/**
 * Created by wei on 2/23/17.
 */

public class SerialPortConnection implements ServiceConnection {
    private SerialPortService.SerialBinder binder = null;
    private static final String TAG = "SenrialPortConnection";

    public void onServiceConnected(ComponentName className, IBinder service) {
        Log.d(TAG, "connected");
        binder = ((SerialPortService.SerialBinder) service);

    }
    public void onServiceDisconnected(ComponentName className) {
        binder = null;
        Log.d(TAG, "distconnected");
    }

    //send command from main to serialPort
    public int sendCommandFunction(String cmd) {
        if(binder!=null) {
            return binder.sendCommand(cmd);
        } else {
            return -1;
        }
    }
};
