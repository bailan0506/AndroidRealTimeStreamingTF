package main;

import android.content.Context;
import android.content.SharedPreferences;
import android.hardware.Camera;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.PreferenceFragment;
import android.preference.PreferenceManager;
import android.provider.Settings;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.LongSparseArray;

import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import selfdriving.streaming.R;

import static java.lang.String.valueOf;


public class SettingsActivity extends AppCompatActivity {
    private static String TAG = "SettingActivity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getFragmentManager().beginTransaction()
                .replace(android.R.id.content, new SettingsFragment())
                .commit();
        setupActionBar();
    }
    /**
     *
     * @param context
     * @return
     */
    public static String getRemoteIP(Context context) {
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(context);
        String ip = sharedPref.getString("pref_remote_ip", "192.168.11.2");
        return ip;
    }

    /**
     *
     */
    private void setupActionBar() {
        ActionBar actionBar = getSupportActionBar();
        if (actionBar != null) {
            // Show the Up button in the action bar.
            actionBar.setDisplayHomeAsUpEnabled(true);
            actionBar.setDisplayShowHomeEnabled(true);
        }
    }
    /**
     *
     * @param context
     * @return
     */
    public static List<Integer> getResolution(Context context) {
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(context);
        String resolution = sharedPref.getString("pref_resolution", "640x480");

        String numbers[]= resolution.split("x", 2);
        List<Integer> res = new ArrayList<Integer>();
        res.add(Integer.valueOf(numbers[0]));
        res.add(Integer.valueOf(numbers[1]));
        return res;
    }

    /**
     *
     * @param context
     * @return
     */
    public static List<Double> getBitRate(Context context) {
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(context);
        String bitRate = sharedPref.getString("pref_bitrate", "1.0 mbps");

        String numbers[]= bitRate.split(" ", 2);
        List<Double> res = new ArrayList<Double>();
        res.add(Double.valueOf(numbers[0]));
        Log.d(TAG,"bit rate:" + res);
        return res;
    }

    /**
     *
     */
    public static class SettingsFragment extends PreferenceFragment {
        private String TAG = "SettingFragment";

        public void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            addPreferencesFromResource(R.xml.preferences);

            generateResolutionPreference();
            generateLocalIP();
            generateBitRate();
        }

        private void generateLocalIP() {
            final EditTextPreference ipPref = (EditTextPreference)findPreference("pref_remote_ip");
            String ip = "192.168.0.1";
            try {
                Enumeration<NetworkInterface> enumNetworkInterfaces = NetworkInterface.getNetworkInterfaces();
                while (enumNetworkInterfaces.hasMoreElements()) {
                    NetworkInterface networkInterface = enumNetworkInterfaces.nextElement();
                    Enumeration<InetAddress> enumInetAddress = networkInterface.getInetAddresses();
                    while (enumInetAddress.hasMoreElements()) {
                        InetAddress inetAddress = enumInetAddress.nextElement();
                        String sAddr = inetAddress.getHostAddress();
                        boolean isIPv4 = sAddr.indexOf(':') < 0;
                        if (!inetAddress.isLoopbackAddress() && isIPv4) {
                            ip = inetAddress.getHostAddress();
                        }
                    }
                }
            } catch (SocketException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }

            ipPref.setText(ip);
            ipPref.setSummary(ip);
            ipPref.setOnPreferenceChangeListener(new Preference.OnPreferenceChangeListener() {
                @Override
                public boolean onPreferenceChange(Preference preference, Object newValue) {
                    ipPref.setSummary(newValue.toString());
                    ipPref.setText(newValue.toString());
                    ipPref.setDefaultValue(newValue);
                    return true;
                }
            });
        }


        private void generateBitRate() {
            double [] rates = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
            CharSequence[] choiceStrItems = new String[rates.length];
            CharSequence[] choiceValues = new String[rates.length];

            String unit = "mbps";
            String defaultValue = "1.0 mbps";
            boolean hasDefault = false;
            for (int i = 0; i < rates.length; ++i) {
                double bitRateValue = rates[i];
                choiceStrItems[i] = valueOf(bitRateValue + " " + unit);
                choiceValues[i] = valueOf(bitRateValue);
                if (bitRateValue == 1.0) {
                    Log.d(TAG,"bitRateValue" + String.valueOf(bitRateValue));
                    hasDefault = true;
                }
            }
            //List preference under the category
            final ListPreference listPref = (ListPreference)findPreference("pref_bitrate");

            listPref.setEntries(choiceStrItems);
            listPref.setEntryValues(choiceStrItems);

            if (hasDefault) {
                listPref.setSummary(defaultValue);
                listPref.setValue(defaultValue);
            } else {
                Log.e(TAG, "does not support default bit rate");
            }

            listPref.setOnPreferenceChangeListener(new Preference.OnPreferenceChangeListener() {
                @Override
                public boolean onPreferenceChange(Preference preference, Object newValue) {
                    listPref.setSummary(newValue.toString());
                    listPref.setValue(newValue.toString());
                    return true;
                }
            });
        }

        /**
         *
         */
        private void generateResolutionPreference() {
            Camera tmpCam = Camera.open();
            Camera.Parameters params = tmpCam.getParameters();
            final List<Camera.Size> prevSizes = params.getSupportedPreviewSizes();
            CharSequence[] choiceStrItems = new String[prevSizes.size()];

            String defaultValue = "640x480";
            boolean hasDefault = false;
            for (int i = 0; i < prevSizes.size(); ++i) {
                Camera.Size cur = prevSizes.get(i);
                choiceStrItems[i] = valueOf(cur.width + "x" + cur.height);
                if (cur.width == 640 && cur.height == 480) {
                    hasDefault = true;
                }
            }

            // List preference under the category
            final ListPreference listPref = (ListPreference)findPreference("pref_resolution");

            listPref.setEntries(choiceStrItems);
            listPref.setEntryValues(choiceStrItems);


            if (hasDefault) {
                listPref.setSummary(defaultValue);
                listPref.setValue(defaultValue);
            } else {
                Log.e(TAG, "does not support default resolution");
            }

            listPref.setOnPreferenceChangeListener(new Preference.OnPreferenceChangeListener() {
                @Override
                public boolean onPreferenceChange(Preference preference, Object newValue) {
                    listPref.setSummary(newValue.toString());
                    listPref.setValue(newValue.toString());
                    return true;
                }
            });

            tmpCam.release();
            tmpCam = null;
        }
    }

}
