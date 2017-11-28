package services;

import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbManager;
import android.os.Binder;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;

import com.felhr.usbserial.UsbSerialDevice;
import com.felhr.usbserial.UsbSerialInterface;
import com.google.gson.Gson;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import utility.ControlCommand;


/**
 * Created by wei on 2/23/17.
 */

/**
 * rotation   0.0 - 1.0 from left to right
 * rotation(0.5): straight
 * rotation(0.4): left slightly
 * rotation(0.6): right slighly
 *
 * throttle    (0.0)1.0 -  1.2
 * 0.0 is stop
 * 1.0 is starting to move, lowest throttle
 * 1.2 is the assigned highest (can be higher) throttle
 */
public class SerialPortService extends Service implements Runnable {

    private final String TAG = "Serial Port Service";
    public final String ACTION_USB_PERMISSION = "wisc.selfdriving.arduinousb.USB_PERMISSION";
    UsbManager usbManager;
    UsbDevice device;
    UsbSerialDevice serialPort;
    UsbDeviceConnection connection;
    int rotationNumber = 0;
    double previousTime = 0.00;

    //sync mode: user syncWrite, syncRead, syncOpen and SyncClose
    boolean serialSync = true;

    private final Binder binder_ = new SerialBinder();
    private AtomicBoolean isRunning_ = new AtomicBoolean(false);

    public class SerialBinder extends Binder {
        public SerialPortService getService() {
            return SerialPortService.this;
        }
        //public int sendCommand(String cmd) {
        public int sendCommand(String cmd) {
            if (serialPort != null) {
                cmd += "\n";
                int res = 1;
                if(serialSync) {
                    res = serialPort.syncWrite(cmd.getBytes(), 0);
                } else {
                    serialPort.write(cmd.getBytes());
                }
                return res;
            } else {
                return -1;
            }
        }
    }

    //register the USB and broadcastReceiver, get permission and attached the device
    public void registerReceiver() {
        IntentFilter filter = new IntentFilter();
        filter.addAction(ACTION_USB_PERMISSION);
        filter.addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED);
        filter.addAction(UsbManager.ACTION_USB_DEVICE_DETACHED);
        registerReceiver(broadcastReceiver, filter);
        Log.d(TAG,"registerReceiver");
    }

    //auto start
    public int onStartCommand(Intent intent, int flags, int startId) {
        Log.d(TAG, "Start command called on Service");
        startService();
        return START_STICKY;
    }

    //auto desotry
    public void onDestroy() {
        Log.d(TAG, "stop service");
        if (broadcastReceiver != null) {
            unregisterReceiver(broadcastReceiver);
        } else {
            Log.d(TAG,"broadcastReceiver is null");
        }

        if (serialPort != null) {
            if(serialSync) {
                serialPort.syncClose();
            } else {
                serialPort.close();
            }
        }
        isRunning_.set(false);
        stopSelf();
    }

    //connect the UsbDevice with the port
    private void startService() {
        Log.d(TAG, "start service");
        usbManager = (UsbManager) getSystemService(Context.USB_SERVICE);
        HashMap<String, UsbDevice> usbDevices = usbManager.getDeviceList();
        Log.d(TAG, String.valueOf(usbDevices.size()));
        if (!usbDevices.isEmpty()) {
            Log.d(TAG,"usbDevices.is size:" + usbDevices.size());
            for (Map.Entry<String, UsbDevice> entry : usbDevices.entrySet()) {
                device = entry.getValue();
                PendingIntent pi = PendingIntent.getBroadcast(this, 0, new Intent(ACTION_USB_PERMISSION), 0);
                usbManager.requestPermission(device, pi);
                break;
            }
        } else {
            Log.e(TAG, "usb device list is empty");
        }
        isRunning_.set(true);

        if(serialSync) {
            Thread curThread = new Thread(this);
            curThread.start();
        }
        registerReceiver();
    }

    //only run under sync mode
    public void run() {
        String buffer = "";
        while(isRunning_.get() && serialSync) {
            byte[] tmp = new byte[1024];
            if(serialPort == null) {
                continue;
            }
            int recv =  serialPort.syncRead(tmp, 0);
            if(recv == -1) {
                continue;
            }
            String cur = new String(tmp);
            buffer += cur.substring(0, recv);
            while(buffer.contains("\n")) {
                int newline = buffer.indexOf("\n");
                String command = buffer.substring(0, newline);
                buffer = buffer.substring(newline + 1);
                parseSerialMessage(command);
            }
        }
    }

    //Defining a Callback which triggers whenever data is read.
    UsbSerialInterface.UsbReadCallback mCallback = new UsbSerialInterface.UsbReadCallback() {
        String buffer = "";
        @Override
        public void onReceivedData(byte[] arg0) {
            try {
                String data = new String(arg0, "UTF-8");
                Log.d(TAG, "receive from serial port:" + data);
                buffer += data;
                while(buffer.contains("\n")) {
                    int newline = buffer.indexOf("\n");
                    String command = buffer.substring(0, newline);
                    buffer = buffer.substring(newline + 1);
                    parseSerialMessage(command);
                }
            } catch (UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    };

    //Receive data from serialPort
    private BroadcastReceiver broadcastReceiver = new BroadcastReceiver() { //Broadcast Receiver to automatically start and stop the Serial connection.
        @Override
        public void onReceive(Context context, Intent intent) {
            //
            if (intent.getAction().equals(ACTION_USB_PERMISSION)) {
                boolean granted = intent.getExtras().getBoolean(UsbManager.EXTRA_PERMISSION_GRANTED);
                if (granted) {
                    connection = usbManager.openDevice(device);
                    serialPort = UsbSerialDevice.createUsbSerialDevice(device, connection);
                    if (serialPort != null) {
                        Log.d(TAG,"serialPort is not null in onReceive ");
                        if (serialSync?serialPort.syncOpen():serialPort.open()) { //Set Serial Connection Parameters.
                            serialPort.setBaudRate(115200);
                            serialPort.setDataBits(UsbSerialInterface.DATA_BITS_8);
                            serialPort.setStopBits(UsbSerialInterface.STOP_BITS_1);
                            serialPort.setParity(UsbSerialInterface.PARITY_NONE);
                            serialPort.setFlowControl(UsbSerialInterface.FLOW_CONTROL_OFF);
                            serialPort.read(mCallback);
                            Log.d(TAG, "Serial Connection Opened!\n");

                        } else {
                            Log.d(TAG, "PORT NOT OPEN");
                        }
                    } else {
                        Log.d(TAG, "PORT IS NULL");
                    }
                } else {
                    Log.d(TAG, "PERM NOT GRANTED");
                }
            }
        };
    };

    /**
     *
     * @param data
     * @return
     */
    //detect the Halldata and current rotation number
    private void parseSerialMessage(String data){
        if (data.contains("rotation(1.0)")) {
            rotationNumber++;
            double speed = calculateSpeed(rotationNumber);
            Log.d(TAG, "No."+rotationNumber + " rotation detected");
            Log.d(TAG, "Speed of this rotation is " + String.valueOf(speed));

            long time = System.currentTimeMillis();
            //sendSerialMessage(throttle,rotationNumber, timeStamp);
        } else if(data.contains("time")) {
            int s = data.indexOf('(');
            int e = data.indexOf(')');
            long time = Long.valueOf(data.substring(s + 1, e));
            double speed = calculateSpeed(rotationNumber);
            sendSerialMessage(speed, rotationNumber, time);
        } else {
            Log.e(TAG, "Invalid Serial Message:" + data);
        }
    }

    //calculate the average throttle for each rotation
    private double calculateSpeed(int rotation){;
        double currenttime= System.currentTimeMillis();
        double speed = 1000.00/(currenttime-previousTime);
        previousTime = currenttime;
        return speed;
    }

    //send the Halldata(current rotation number and throttle) to main
    private void sendSerialMessage(double speed, int rotation, long time) {

        ControlCommand obj = new ControlCommand(speed, rotation, time);
        Gson gson = new Gson();
        String json = gson.toJson(obj);

        long now = System.currentTimeMillis();
        long rrt =  now - time;
        Log.d(TAG, "now: " + now + " RRT:" + rrt);
        Log.d(TAG, json);
        Intent intent = new Intent("SerialPort");
        intent.putExtra("serialMessage", json);
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent);
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder_;
    }

}
