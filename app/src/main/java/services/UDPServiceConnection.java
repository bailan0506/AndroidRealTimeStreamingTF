package services;

import android.content.ComponentName;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.util.Log;

import java.net.InetAddress;

import utility.FrameData;

/**
 * Created by wei on 4/4/17.
 */

public class UDPServiceConnection implements ServiceConnection {

    private UDPService.UDPBinder binder = null;
    private static final String TAG = "UDPServiceConnection";


    public void onServiceConnected(ComponentName className, IBinder service) {
        Log.d(TAG, "connected");
        binder = ((UDPService.UDPBinder) service);
    }
    public void onServiceDisconnected(ComponentName className) {
        binder = null;
        Log.d(TAG, "distconnected");
    }
    public boolean isRunning() {
        return binder.isRunning();
    }
    public void sendData(FrameData data, InetAddress remoteIPAddress, int remotePort) {
        if (data != null && remoteIPAddress!= null && String.valueOf(remotePort) != null){
            binder.sendData(data, remoteIPAddress, remotePort);
        }
    }

}
