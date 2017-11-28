package main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ActivityInfo;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.widget.Button;

import selfdriving.streaming.R;
import com.google.gson.Gson;

import database.DatabaseHelper;
import services.SerialPortConnection;
import services.SerialPortService;
import services.UDPServiceConnection;
import services.UDPService;
import services.SensorService;
import utility.Constants;
import utility.FrameData;
import utility.ControlCommand;
import utility.Trace;

import static java.lang.Math.abs;

////
// import android.content.pm.PackageManager;
// import android.hardware.camera2.CameraAccessException;
// import android.hardware.camera2.CameraCharacteristics;
// import android.hardware.camera2.CameraManager;
// import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Handler;
import android.os.HandlerThread;
// import android.os.Trace;
import android.util.Size;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.WindowManager;
import android.widget.Toast;
import java.nio.ByteBuffer;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
// import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.



import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.Surface;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import main.OverlayView.DrawCallback;
import android.util.Log;


public class MainActivity extends Activity implements SurfaceHolder.Callback, Camera.PreviewCallback{
	private final static String TAG = MainActivity.class.getSimpleName();

	// skype frame rate 5-30
	// skype bit rate 30kbps - 950kbps
	// skype resolution 	640*480, 320*240, 160*120
	private final static int DEFAULT_FRAME_RATE = 10;
	private static int frame_bitrate = (int)1e6; // 1mbps
	// 0.5mbps 1mpbs 1.5mpbs 2mbps 2.5mbps 3mbps

	Camera camera;
	SurfaceHolder previewHolder;
	byte[] previewBuffer;
	boolean isStreaming = false;
	AvcEncoder encoder;
    boolean consistentControl = false;

	private String ip = "192.168.11.2";

	public InetAddress address;
	public final int port = 55555;

	List<FrameData> encDataList = new LinkedList<FrameData>();
	List<ControlCommand> encControlCommandList = new LinkedList<ControlCommand>();
	LatencyMonitor latencyMonitor;

	private static Intent mSensor = null;
	private DatabaseHelper dbHelper_ = null;
	private FileOutputStream fOut_ = null;

	// width* height = 640 * 480 or 320 * 240
	private int width = 640;
	private int height = 480;
//////////////////////////////////////
	private Runnable postInferenceCallback;
	private Runnable imageConverter;
	private byte[] lastPreviewFrame;
	private static final Logger LOGGER = new Logger();
	
	private static final int PERMISSIONS_REQUEST = 1;

//	private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
//	private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

	private boolean debug = false;

	private Handler handler;
	private HandlerThread handlerThread;
	private boolean isProcessingFrame = false;
	private byte[][] yuvBytes = new byte[3][];
	private int[] rgbBytes = null;
	private int yRowStride;
/////////////////////////////////////////////////////////

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO;
  }
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

//////////////////////////////////////////

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		Log.v("onCreate", "main activity start");
		super.onCreate(savedInstanceState);
		this.requestWindowFeature(Window.FEATURE_NO_TITLE);
		this.setContentView(R.layout.activity_main);

		if (Build.MODEL.equals("Nexus 5X")){
			//Nexus 5X's screen is reversed, ridiculous! the image sensor does not fit in correct orientation
			setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE);
		} else {
			setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
		}


		this.findViewById(R.id.btntest).setOnClickListener(
				new View.OnClickListener() {
					@Override
					public void onClick(View v) {
						Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
						startActivity(intent);
					}
				});

		this.findViewById(R.id.btnstart).setOnClickListener(
				new View.OnClickListener() {
					@Override
					public void onClick(View v) {
						if (isStreaming) {
							((Button) v).setText("Start");
							stopServices();
							stopStream();
						} else {
							startStream();
							startServices();
						}
					}
				});


		setupFolders();

		SurfaceView svCameraPreview = (SurfaceView) this.findViewById(R.id.svCameraPreview);
		this.previewHolder = svCameraPreview.getHolder();
		this.previewHolder.addCallback(this);
	}


	private void startServices() {
		startSerialService();
		startUDPService();
		mSensor = new Intent(this, SensorService.class);
		startService(mSensor);
		LocalBroadcastManager.getInstance(this).registerReceiver(mMessageReceiver, new IntentFilter("sensor"));
		LocalBroadcastManager.getInstance(this).registerReceiver(mMessageReceiver, new IntentFilter("udp"));
		LocalBroadcastManager.getInstance(this).registerReceiver(mMessageReceiver, new IntentFilter("control"));

		long time = System.currentTimeMillis();
		dbHelper_ = new DatabaseHelper();
		dbHelper_.createDatabase(time);

		try {
			File file = new File(Constants.kVideoFolder.concat(String.valueOf(time)).concat(".raw"));
			this.fOut_ = new FileOutputStream(file, true);
		} catch (Exception e) {
			e.printStackTrace();
		}

		latencyMonitor = new LatencyMonitor();
	}

	private void setupFolders () {
		File dbDir = new File(Constants.kDBFolder);
		File videoDir = new File(Constants.kVideoFolder);
		if (!dbDir.exists()) {
			dbDir.mkdirs();
		}
		if(!videoDir.exists()) {
			videoDir.mkdir();
		}
	}

	// protected void onDestroy() {
	// 	super.onDestroy();
	// 	stopServices();
	// }

	private void stopServices() {
		stopUDPService();
		stopSerialService();
		if (mSensor!= null){
			stopService(mSensor);
			mSensor = null;
		}
		if (dbHelper_!= null) {
			dbHelper_.closeDatabase();
		}
		if (fOut_ != null) {
			try {
				fOut_.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		if (mMessageReceiver!= null) {
			LocalBroadcastManager.getInstance(this).unregisterReceiver(mMessageReceiver);
		}
	}

	// @Override
	// protected void onPause() {
	// 	this.stopStream();
	// 	if (encoder != null)
	// 		encoder.close();

	// 	super.onPause();
	// }

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		int id = item.getItemId();
		if (id == R.id.action_settings)
			return true;
		return super.onOptionsItemSelected(item);
	}


	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		startCamera();
	}

	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
		Log.d(TAG, "surface changed");
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder holder) {
		stopCamera();
	}


	private void loadPreferences() {
		List<Integer> resolution = SettingsActivity.getResolution(MainActivity.this);
		this.width = resolution.get(0);
		this.height = resolution.get(1);

		this.ip = SettingsActivity.getRemoteIP(MainActivity.this);
		Log.d(TAG, "Resolution:" + this.width + "x" + this.height);

		List<Double> bitRate = SettingsActivity.getBitRate(MainActivity.this);
		if (bitRate.get(0) != null) {
			double temp = bitRate.get(0);
			this.frame_bitrate =  (int) (temp * 1000000.0);
		}
	}


	private void startStream() {

		loadPreferences();

		stopCamera();
		startCamera();

		this.encoder = new AvcEncoder();
		this.encoder.init(width, height, DEFAULT_FRAME_RATE, frame_bitrate);
		try {
			this.address = InetAddress.getByName(ip);
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}

		this.isStreaming = true;

		Thread streamingThread = new Thread(senderRun);
		streamingThread.start();
		Thread controlThread = new Thread(controlMessageThread);
		controlThread.start();

		((Button) this.findViewById(R.id.btnstart)).setText("Stop");
		this.findViewById(R.id.btntest).setEnabled(false);

		Log.d(TAG,"frame_bitrate is:" + frame_bitrate);

	}

	private void stopStream() {
		this.isStreaming = false;

		if (this.encoder != null)
			this.encoder.close();
		this.encoder = null;

		this.findViewById(R.id.btntest).setEnabled(true);
	}

	private void startCamera() {


		Log.d(TAG, "width: " + width + " height:" + height);


		this.previewHolder.setFixedSize(width, height);

		int stride = (int) Math.ceil(width / 16.0f) * 16;
		int cStride = (int) Math.ceil(width / 32.0f) * 16;
		final int frameSize = stride * height;
		final int qFrameSize = cStride * height / 2;

		this.previewBuffer = new byte[frameSize + qFrameSize * 2];

		try {
			camera = Camera.open();
			camera.setPreviewDisplay(this.previewHolder);

			Camera.Parameters params = camera.getParameters();
			params.setPreviewSize(width, height);
			params.setPreviewFormat(ImageFormat.YV12);
			camera.setParameters(params);
			camera.setPreviewCallbackWithBuffer(this);
			camera.addCallbackBuffer(previewBuffer);
			camera.startPreview();

			// adjust the orientation
			camera.setDisplayOrientation(0);
		} catch (IOException e) {
			//TODO:
		} catch (RuntimeException e) {
			//TODO:
		}
	}


	private void stopCamera() {
		if (camera != null) {
			camera.setPreviewCallback(null);
			camera.stopPreview();
			camera.release();
			camera = null;
		}
	}

	/**
	 * we cannot store the video when we use onPreviewFrame
	 * @param data
	 * @param camera
	 */
	@Override
	public void onPreviewFrame(final byte[] data, final Camera camera) {
		Log.v("main","onPreviewFrame");
		if (isProcessingFrame) {
			Log.v("Main","IsProcessingFrame");
		}

		else {

			try {
				// Initialize the storage bitmaps once when the resolution is known.
				Log.v("main","Initialize the storage bitmaps once when the resolution is known.");
				if (rgbBytes == null) {
					Camera.Size previewSize = camera.getParameters().getPreviewSize();
				//	height = previewSize.height;
				//	width = previewSize.width;
					rgbBytes = new int[width * height];
					onPreviewSizeChosen(new Size(width, height), 270);
				}
			} catch (final Exception e) {
				return;
			}


			isProcessingFrame = true;
			lastPreviewFrame = data;
			yuvBytes[0] = data;
			yRowStride = width;

			LOGGER.d("image converter");
			imageConverter =
					new Runnable() {
						@Override
						public void run() {
							ImageUtils.convertYUV420SPToARGB8888(data, width, height, rgbBytes);
						}
					};

			postInferenceCallback =
					new Runnable() {
						@Override
						public void run() {
							camera.addCallbackBuffer(data);
							isProcessingFrame = false;
						}
					};
			LOGGER.d("image processing start");
			processImage();
			Log.v("Main", "image processed");
		}
		camera.addCallbackBuffer(previewBuffer);

		if (isStreaming) {
			/*
			if (FrameData.sequenceIndex%2 == 0) {
				encoder.forceIFrame();
			}
			*/
			// long time = System.currentTimeMillis();
			FrameData frameData = encoder.offerEncoder(data);
			// Log.d(TAG, String.valueOf(System.currentTimeMillis() - time));

            List<FrameData> frames = frameData.split();
			for (int i = 0; i < frames.size(); ++i) {
				FrameData frame = frames.get(i);
				dbHelper_.insertFrameData(frame);
				if (frame.getDataSize() > 0) {
					synchronized (encDataList) {
						encDataList.add(frame);
					}
				}
			}

		}
	}


	private void appendToVideoFile(byte [] data) {
		try {
			int datalen = data.length;
			String strlen = String.valueOf(datalen);
			int encodelen = strlen.length();
			byte [] header = new byte[encodelen + 1];
			for (int i = 0; i < encodelen; ++ i) {
				header[i] = (byte)strlen.charAt(i);
			}
			header[encodelen] = '\n';
			this.fOut_.write(header, 0, encodelen + 1);
			this.fOut_.write(data, 0, data.length);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	//initial UDPConnetion
	private static Intent mUDPService = null;
	private static UDPServiceConnection mUDPConnection = null;

	private void startUDPService() {
		Log.d(TAG, "startUDPService");
		mUDPService = new Intent(this, UDPService.class);
		mUDPConnection = new UDPServiceConnection();
		bindService(mUDPService, mUDPConnection, Context.BIND_AUTO_CREATE);
		startService(mUDPService);
	}

	private void stopUDPService() {
		if (mUDPService != null && mUDPConnection != null) {
			unbindService(mUDPConnection);
			stopService(mUDPService);
			mUDPService = null;
			mUDPConnection = null;
		}
	}

	//initial SerialPortConnection
	private static Intent mSerial = null;
	private static SerialPortConnection mSerialPortConnection = null;

	private void startSerialService() {
		Log.d(TAG, "start serial service");
		mSerial = new Intent(this, SerialPortService.class);
		mSerialPortConnection = new SerialPortConnection();
		bindService(mSerial, mSerialPortConnection, Context.BIND_AUTO_CREATE);
		startService(mSerial);
	}

	private void stopSerialService() {

		Log.d(TAG, "stop serial service");
		if(mSerial != null && mSerialPortConnection != null) {
			mSerialPortConnection.sendCommandFunction("throttle(0.0)");
			mSerialPortConnection.sendCommandFunction("steering(0.5)");

			unbindService(mSerialPortConnection);
			stopService(mSerial);
			mSerial = null;
			mSerialPortConnection = null;
		}
	}


	private BroadcastReceiver mMessageReceiver = new BroadcastReceiver(){
		@Override
		public void onReceive(Context context, Intent intent) {

			if (intent.getAction().equals("sensor")) {
				String message = intent.getStringExtra("trace");
				Trace trace = new Trace();
				trace.fromJson(message);

				if (dbHelper_.isOpen()) {
					dbHelper_.insertSensorData(trace);
				}
				// Log.d(TAG, "sensor data: " + message);
			} else if(intent.getAction().equals("udp")) {
				String message = intent.getStringExtra("latency");

				Gson gson = new Gson();
				FrameData frameData = gson.fromJson(message, FrameData.class);

				if (dbHelper_.isOpen()) {
					dbHelper_.updateFrameData(frameData);
				}
				//Log.d(TAG, "frame data update: " + dbHelper_.updateFrameData(frameData));

			} else if (intent.getAction().equals("control")){
				String receivedCommand = intent.getStringExtra("control");
				Gson gson = new Gson();
				// Log.d(TAG,"control: " + receivedCommand);
				ControlCommand controlCommand = gson.fromJson(receivedCommand, ControlCommand.class);
				if (controlCommand != null) {
				    latencyMonitor.recordOneWayLatency(System.currentTimeMillis() - controlCommand.timeStamp);
					synchronized (encControlCommandList) {
						encControlCommandList.add(controlCommand);
					}
				}
			} else {
				Log.d(TAG, "unknown intent: " + intent.getAction());
			}
		}

	};


	/**
	 * push data to sender
	 */
	Runnable senderRun = new Runnable() {
		@Override
		public void run() {
			while (isStreaming) {
				boolean empty = false;
				FrameData frameData = null;

				synchronized (encDataList) {
					if (encDataList.size() == 0) {
						empty = true;
					} else
						frameData = encDataList.remove(0);
				}
				if (empty) {
					try {
						Thread.sleep(1);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					continue;
				}
				//we can start 2 thread, one is with timeStamp header send to one server and get timeStamp back
				// the other thread will send without header and directly show the video.
				appendToVideoFile(frameData.rawFrameData);
				if (mUDPConnection != null && mUDPConnection.isRunning()) {
					mUDPConnection.sendData(frameData, address, port);
				}
			}
		}
	};

	/**
	 * push data to sender
	 */
	Runnable controlMessageThread = new Runnable() {
		@Override
		public void run() {
			while (isStreaming) {
				boolean empty = false;
				ControlCommand controlCommand = null;

				synchronized (encControlCommandList) {
					if (encControlCommandList.size() == 0) {
						empty = true;
					} else
						controlCommand = encControlCommandList.remove(0);
				}
				if (empty) {
					try {
						Thread.sleep(1);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
					continue;
				}
				/*
				* delay for consistence control
				* */
                long timeDiff = System.currentTimeMillis() - controlCommand.timeStamp;
                if (consistentControl) {
                    long diff = timeDiff - latencyMonitor.getAverageOneWayLatency();
                    if (diff < 0) {
                        try {
                            Thread.sleep(Math.abs(diff));
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
				if (mSerialPortConnection != null) {
					double throttle = (float)0.0;
					double steering = controlCommand.steering;
					if(controlCommand.throttle > 0.5) {
						throttle = (float) ((controlCommand.throttle-0.5) * 0.4 + 1.0);
					}
					mSerialPortConnection.sendCommandFunction("throttle(" + String.valueOf(throttle) + ")");
					mSerialPortConnection.sendCommandFunction("steering(" + String.valueOf(steering) + ")");
				}
			}
		}
	};




//	public void onImageAvailable(final ImageReader reader) {
//		//We need wait until we have some size from onPreviewSizeChosen
//		// if (width == 0 || height == 0) {
//		//   return;
//		// }
//		if (rgbBytes == null) {
//		  rgbBytes = new int[width * height];
//		}
//		try {
//		  final Image image = reader.acquireLatestImage();
//
//		  if (image == null) {
//			return;
//		  }
//
//		  if (isProcessingFrame) {
//			image.close();
//			return;
//		  }
//		  isProcessingFrame = true;
//		//   Trace.beginSection("imageAvailable");
//		  final Plane[] planes = image.getPlanes();
//		  fillBytes(planes, yuvBytes);
//		  yRowStride = planes[0].getRowStride();
//		  final int uvRowStride = planes[1].getRowStride();
//		  final int uvPixelStride = planes[1].getPixelStride();
//
//		  imageConverter =
//			  new Runnable() {
//				@Override
//				public void run() {
//				  ImageUtils.convertYUV420ToARGB8888(
//					  yuvBytes[0],
//					  yuvBytes[1],
//					  yuvBytes[2],
//					  width,
//					  height,
//					  yRowStride,
//					  uvRowStride,
//					  uvPixelStride,
//					  rgbBytes);
//				}
//			  };
//
//		  postInferenceCallback =
//			  new Runnable() {
//				@Override
//				public void run() {
//				  image.close();
//				  isProcessingFrame = false;
//				}
//			  };
//
//		  processImage();
//		} catch (final Exception e) {
//		//   LOGGER.e(e, "Exception!");
//		//   Trace.endSection();
//		  return;
//		}
//		// Trace.endSection();
//	}


	@Override
	public synchronized void onStart() {
	  super.onStart();
	}
  
	@Override
	public synchronized void onResume() {
	  super.onResume();
  
	  handlerThread = new HandlerThread("inference");
	  handlerThread.start();
	  handler = new Handler(handlerThread.getLooper());
	}
  
	@Override
	public synchronized void onPause() {
  
	  if (!isFinishing()) {

		finish();
	  }
  
	  handlerThread.quitSafely();
	  try {
		handlerThread.join();
		handlerThread = null;
		handler = null;
	  } catch (final InterruptedException e) {
		
	  }

	  this.stopStream();
	  if (encoder != null)
		  encoder.close();
  
	  super.onPause();
	}
  
	@Override
	public synchronized void onStop() {
	  super.onStop();
	}
  
	@Override
	public synchronized void onDestroy() {
	  super.onDestroy();
	  stopServices();
	}
  
	protected synchronized void runInBackground(final Runnable r) {
	  if (handler != null) {
		handler.post(r);
	  }
	}
  
//	protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
//		// Because of the variable row stride it's not possible to know in
//		// advance the actual necessary dimensions of the yuv planes.
//		for (int i = 0; i < planes.length; ++i) {
//		  final ByteBuffer buffer = planes[i].getBuffer();
//		  if (yuvBytes[i] == null) {
//		  	Log.v("fillbytes","Initializing buffer " );
//			yuvBytes[i] = new byte[buffer.capacity()];
//		  }
//		  buffer.get(yuvBytes[i]);
//		}
//	}
//
	public boolean isDebug() {
		return debug;
	}
	
	public void addCallback(final OverlayView.DrawCallback callback) {
		final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
		if (overlay != null) {
		  overlay.addCallback(callback);
		}
	}



  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    width = size.getWidth();
    height = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", width, height);
    rgbFrameBitmap = Bitmap.createBitmap(width, height, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            width, height,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    LOGGER.i("Initializing tracking overlay");
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    LOGGER.i("Initialized Tracking overlay");
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    LOGGER.i("Onpreviewsizechosen");
    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + width + "x" + height);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

 
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        width,
        height,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, width, 0, 0, width, height);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
	        Log.v("main","Running detection on image " );
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
              case MULTIBOX:
                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                break;
              case YOLO:
                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            requestRender();
            computingDetection = false;
          }
        });
  }


	protected int getScreenOrientation() {
		switch (getWindowManager().getDefaultDisplay().getRotation()) {
			case Surface.ROTATION_270:
				return 270;
			case Surface.ROTATION_180:
				return 180;
			case Surface.ROTATION_90:
				return 90;
			default:
				return 0;
		}
	}


	protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

 
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }



	protected int[] getRgbBytes() {
		imageConverter.run();
		return rgbBytes;
	}

	protected int getLuminanceStride() {
		return yRowStride;
	}

	protected byte[] getLuminance() {
		return yuvBytes[0];
	}

	protected void readyForNextImage() {
		if (postInferenceCallback != null) {
			postInferenceCallback.run();
		}
	}

	public void requestRender() {
		final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
		if (overlay != null) {
			overlay.postInvalidate();
		}
	}
}
