package com.achel.truemood;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.util.Log;
import android.widget.TextView;

import androidx.core.content.res.ResourcesCompat;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FacialExpressionRecognition {

    // Interpreter for TensorFlow Lite model to recognize emotions
    private Interpreter interpreter;

    // Input size for the model
    private int INPUT_SIZE;

    // Dimensions of the input image (height and width)
    private int height = 0;
    private int width = 0;

    // GpuDelegate for GPU acceleration
    private GpuDelegate gpuDelegate = null;

    // CascadeClassifier for face detection
    private CascadeClassifier faceDetector;

    // Constructor to initialize the model and the face detector
    public FacialExpressionRecognition(AssetManager assetManager , Context context , String modelPath , int inputSize ) throws IOException {
        INPUT_SIZE = inputSize;

        // Set up the options for TensorFlow Lite interpreter
        Interpreter.Options options = new Interpreter.Options();

        // Initialize GPU delegate for accelerated inference
        gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);  // Add GPU delegate to the options

        // Set the number of threads for interpreter (this can be tuned based on the device)
        options.setNumThreads(4);

        // Load the model into the interpreter
        interpreter = new Interpreter(loadModelFile(assetManager , modelPath) , options);

        // Log a success message after the model is loaded
        Log.d("Facial Expression" , "Model loaded successfully");

        // Initialize the face detection classifier
        loadFaceDetector(context);
    }

    // Loads the TensorFlow Lite model from assets
    private MappedByteBuffer loadModelFile(AssetManager assetManager , String modelPath) throws IOException {
        // Open the model file from assets
        AssetFileDescriptor assetFileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputSteam = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputSteam.getChannel();

        // Get the start offset and declared length of the file
        long startOffset = assetFileDescriptor.getStartOffset();
        long declaredLength = assetFileDescriptor.getDeclaredLength();

        // Map the file into memory and return
        return fileChannel.map(FileChannel.MapMode.READ_ONLY , startOffset , declaredLength);
    }

    // Loads the face detection classifier (Haar Cascade)
    private void loadFaceDetector(Context context) {
        try {
            // Open the resource for Haar Cascade face detection model
            InputStream inputStream = context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);

            // Create a directory to store the classifier file
            File cascadeDir = context.getDir("cascade" , Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir , "haarcascade_frontalface_alt");

            // Open an output stream to write the classifier data to the file
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);

            // Buffer to read the classifier file in chunks
            byte[] buffer = new byte[4096];
            int bytesRead;

            // Copy data from input stream to output stream
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer , 0 , bytesRead);
            }

            // Close streams after reading
            inputStream.close();
            outputStream.close();

            // Initialize the face detector with the classifier file
            faceDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());

            // Log a success message if the classifier is loaded
            Log.d("Facial Expression" , "Face detector loaded successfully");
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Recognizes faces and emotions in the input image (Mat)
    public Mat recognizeImage(Mat matImage , Context context) {

        // Convert the input image to grayscale for face detection
        Mat grayScaleImage = new Mat();
        Imgproc.cvtColor(matImage, grayScaleImage, Imgproc.COLOR_BGR2GRAY);

        // Set the height and width based on the grayscale image
        height = grayScaleImage.height();
        width = grayScaleImage.width();

        // Set the minimum face size threshold based on image height
        int absoluteFaceSize = (int) (height * 0.1);

        // Create a MatOfRect to store the detected faces
        MatOfRect faces = new MatOfRect();

        // Detect faces if the face detector is available
        if (faceDetector != null) {
            faceDetector.detectMultiScale(
                    grayScaleImage,
                    faces,
                    1.1,
                    2,
                    2,
                    new Size(absoluteFaceSize, absoluteFaceSize),
                    new Size()
            );
        }

        // Convert the MatOfRect to a Rect array for easier processing
        Rect[] faceArray = faces.toArray();

        // Load custom font from resources to display emotion text
        Typeface customTypeface = ResourcesCompat.getFont(context, R.font.aldrich);
        Paint paint = new Paint();
        paint.setTypeface(customTypeface);  // Set the custom font
        paint.setTextSize(50);              // Set the text size
        paint.setColor(0xFFFF0000);         // Set the text color (Red)
        paint.setAntiAlias(true);           // Enable anti-aliasing for smooth text rendering

        // Loop through all the detected faces
        for (int i = 0; i < faceArray.length; i++) {
            // Draw a rectangle around each detected face
            Imgproc.rectangle(
                    matImage,
                    faceArray[i].tl(),
                    faceArray[i].br(),
                    new Scalar(0, 255, 0),  // Green color for the rectangle
                    3
            );

            // Crop the detected face from the image for emotion recognition
            Rect roi = new Rect(
                    (int) faceArray[i].tl().x,
                    (int) faceArray[i].tl().y,
                    (int) faceArray[i].br().x - (int) faceArray[i].tl().x,
                    (int) faceArray[i].br().y - (int) faceArray[i].tl().y
            );

            // Create a Mat for the cropped region of interest (ROI)
            Mat croppedRgba = new Mat(matImage, roi);

            // Convert the cropped face to Bitmap format for processing
            Bitmap bitmap = Bitmap.createBitmap(
                    croppedRgba.cols(),
                    croppedRgba.rows(),
                    Bitmap.Config.ARGB_8888
            );
            Utils.matToBitmap(croppedRgba, bitmap);

            // Resize the Bitmap to match the model's input size
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

            // Convert the Bitmap to a ByteBuffer for input into the TensorFlow Lite model
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

            // Use the model to predict the emotion
            float[][] emotion = new float[1][1];
            interpreter.run(byteBuffer, emotion);

            // Log the predicted emotion value
            float emotionValue = (float) Array.get(Array.get(emotion, 0), 0);
            Log.d("Facial Expression", "Output: " + emotionValue);

            // Get the emotion label based on the predicted value
            String emotionText = getEmotionText(emotionValue);

            // Convert the Mat to Bitmap for drawing the emotion text on the image
            Bitmap matBitmap = Bitmap.createBitmap(matImage.width(), matImage.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matImage, matBitmap);

            // Create a Canvas to draw on the Bitmap
            Canvas canvas = new Canvas(matBitmap);

            // Draw the emotion text above the detected face
            canvas.drawText(
                    emotionText,
                    (float) (faceArray[i].tl().x + 10),  // X position
                    (float) (faceArray[i].tl().y - 35),  // Y position
                    paint
            );

            // Convert the Bitmap back to Mat after drawing the text
            Utils.bitmapToMat(matBitmap, matImage);
        }

        return matImage;
    }

    // Converts the emotion value to a human-readable emotion label
    private String getEmotionText(float emotionValue) {
        String val = "";

        // Map the emotion value to an emotion label
        if(emotionValue >= 0 & emotionValue < 0.5){
            val = "Surprise";
        } else if(emotionValue >= 0.5 & emotionValue < 1.5){
            val = "Fear";
        } else if(emotionValue >= 1.5 & emotionValue < 2.5){
            val = "Angry";
        } else if(emotionValue >= 2.5 & emotionValue < 3.5){
            val = "Neutral";
        } else if(emotionValue >= 3.5 & emotionValue < 4.5){
            val = "Sad";
        } else if(emotionValue >= 4.5 & emotionValue < 5.5){
            val = "Disgust";
        } else {
            val = "Happy";
        }

        return val;
    }

    // Converts the Bitmap image to ByteBuffer for input to the TensorFlow Lite model
    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int sizeImage = INPUT_SIZE; // Input size of the image (e.g., 48x48)

        byteBuffer = ByteBuffer.allocateDirect(4 * 1 * sizeImage * sizeImage * 3);
        byteBuffer.order(ByteOrder.nativeOrder());  // Set byte order for the buffer

        // Get pixel values from the Bitmap
        int[] intValues = new int[sizeImage * sizeImage];
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0, scaledBitmap.getWidth(), scaledBitmap.getHeight());

        int pixel = 0;
        // Loop through all pixels and convert each RGB value to float and add to ByteBuffer
        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
                byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
                byteBuffer.putFloat((val & 0xFF) / 255.0f);
            }
        }

        return byteBuffer;
    }

}