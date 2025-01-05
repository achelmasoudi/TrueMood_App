package com.achel.truemood;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.annotation.NonNull;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class EmotionCameraActivity extends CameraActivity {

    private static final String TAG="MainActivity"; // Tag for logging purposes
    private Mat mRgba; // To hold the RGBA image frame (color image)
    private Mat mGray; // To hold the grayscale image frame
    private CameraBridgeViewBase mOpenCvCameraView; // Camera view to display video feed
    private FacialExpressionRecognition facialExpressionRecognition; // Object for emotion recognition

    private ImageView flipBtn;
    private int cameraIndex = 0; // Index of the camera (0 for front-facing, 1 for back-facing)
    public EmotionCameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    // Called when the activity is first created
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Remove title bar and keep the screen on during camera use
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_emotion_camera); // Set the content view layout

        getPermission(); // Request camera permissions if not already granted

        // Initialize OpenCV camera view
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cameraViewId);

        // Initialize the flip button
        flipBtn = findViewById(R.id.flipBtnId);

        // Set the listener for camera frames
        mOpenCvCameraView.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            // Called when the camera view starts
            @Override
            public void onCameraViewStarted(int width, int height) {
                mRgba = new Mat(height, width, CvType.CV_8UC4); // Create a matrix to store RGBA frame
                mGray = new Mat(height, width, CvType.CV_8UC1); // Create a matrix to store grayscale frame
            }

            // Called when the camera view stops
            @Override
            public void onCameraViewStopped() {
                mRgba.release(); // Release the resources when camera view stops
            }

            // Called to process each frame captured by the camera
            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                mRgba = inputFrame.rgba(); // Get the RGBA image from the camera
                mGray = inputFrame.gray(); // Get the grayscale image from the camera

                if (cameraIndex == 1) {
                    Core.flip(mRgba, mRgba, 1); // Flip the image if camera index is 1 (back-facing camera)
                    Core.flip(mGray , mGray , 1); // Flip the grayscale image if camera index is 1 (back-facing camera)
                }

                // Perform facial expression recognition on the current frame
                mRgba = facialExpressionRecognition.recognizeImage(mRgba, EmotionCameraActivity.this);

                return mRgba; // Return the processed RGBA frame
            }
        });

        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            mOpenCvCameraView.enableView(); // Enable the camera view
        }

        // Load the emotion recognition model and initialize the classifier
        modelPrecess();

        flipBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraIndex = cameraIndex^1; // Toggle the camera index (0 to 1, 1 to 0)
                mOpenCvCameraView.disableView();

                mOpenCvCameraView.setCameraIndex(cameraIndex); // Set the camera index
                mOpenCvCameraView.enableView(); // Enable the camera view
            }
        });
    }

    // Method to load the emotion recognition model and initialize the classifier
    private void modelPrecess() {
        try {
            int inputSize = 48; // Model input size (48x48 image)
            facialExpressionRecognition = new FacialExpressionRecognition(
                    getAssets(), // Get assets from the app
                    EmotionCameraActivity.this, // Context of the activity
                    "emotion_mod.tflite", // Path to the emotion recognition model (TensorFlow Lite)
                    inputSize); // Size of the input image for the model
        } catch (IOException e) {
            e.printStackTrace(); // Catch and print the error if the model loading fails
        }
    }

    // Return a list of camera views (this activity only has one camera view)
    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    // Request camera permission if it is not granted
    private void getPermission() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 101); // Request permission for the camera
        }
    }

    // Handle the result of the camera permission request
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission(); // If permission is not granted, request it again
        }
    }

    // Called when the activity is resumed (visible to the user)
    @Override
    protected void onResume() {
        super.onResume();
        mOpenCvCameraView.enableView(); // Enable the camera view when the activity resumes
    }

    // Called when the activity is paused (not visible to the user)
    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView(); // Disable the camera view to release resources
        }
    }

    // Called when the activity is destroyed (finished)
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView(); // Disable the camera view to release resources
        }
    }
}