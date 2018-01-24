/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 *  edited by Andreas Wilhelm and Alice Bollenmiller
 *  for the purposes of the lecture Deep Learning
 *
 *
 */

package com.dl.app.dogbreedanalyzerm;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Size;
import android.view.Display;

import java.util.List;

//import com.dl.app.dogbreedanalyzer.BorderedText;

/*

    Main Activity of mobile App which implements the Camera and starts the Classification

 */

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {

    // Input shapes
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    // Input Layer
    private static final String INPUT_NAME = "input";

    // Output Layer
    //private static final String OUTPUT_NAME = "MobilenetV1/Predictions/Softmax";
    private static final String OUTPUT_NAME = "final_result";

    // Model Name from Assets
    //private static final String MODEL_FILE = "file:///android_asset/graph.pb";
    private static final String MODEL_FILE = "file:///android_asset/opt4_retrained_dog_graph_mobilenet_0.50_224_700_0.007.pb";


    // Label Name from Assets
    private static final String LABEL_FILE = "file:///android_asset/retrained_dog_labels_mobilenet_0.50_224_700_0.007.txt";

    // Bitmap Format of Image given by the camera
    private static final boolean SAVE_PREVIEW_BITMAP = false;

    // Maintain_aspect - true -> Image is scaled, false -> Image is taken in original format
    private static final boolean MAINTAIN_ASPECT = true;

    // Standard Image Size
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 580); //480

    // Classifier created by the implementation of TensorFlowImageClassifier
    private Classifier classifier;

    // Screen orientation
    private Integer sensorOrientation;

    // Resized Image in Bitmap format
    private Bitmap cropCopyBitmap;

    // Miscellaneous settings for the input of an image
    private int previewWidth = 0;
    private int previewHeight = 0;
    private byte[][] yuvBytes;
    private int[] rgbBytes = null;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private boolean computing = false;

    // Matrices for Cropping an Image (Bitmap)
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    // View for display results including object name and probability
    private ResultsView resultsView;

    // Processing time for statistical purpose
    private long lastProcessingTimeMs;

    //
    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {

        // Init Classifier
        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        // Set Results panel
        resultsView = (ResultsView) findViewById(R.id.results);
        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        // Configure Screen and Display Orientation
        final Display display = getWindowManager().getDefaultDisplay();
        final int screenOrientation = display.getRotation();
        sensorOrientation = rotation + screenOrientation;

        // Crop Image and store as Bitmap
        rgbBytes = new int[previewWidth * previewHeight];
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        INPUT_SIZE, INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        yuvBytes = new byte[3][];

    }

    // Read Image Stream
    @Override
    public void onImageAvailable(final ImageReader reader) {
        Image image = null;

        try {
            image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (computing) {
                image.close();
                return;
            }
            computing = true;

            Trace.beginSection("imageAvailable");

            final Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);

            final int yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            ImageUtils.convertYUV420ToARGB8888(
                    yuvBytes[0],
                    yuvBytes[1],
                    yuvBytes[2],
                    previewWidth,
                    previewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes);

            image.close();
        } catch (final Exception e) {
            if (image != null) {
                image.close();
            }
            Trace.endSection();
            return;
        }

        // Image as Bitmap
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        // Classify the cropped Image in background and set results
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = classifier.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        resultsView.setResults(results);
                        requestRender();
                        computing = false;
                    }
                });

        Trace.endSection();
    }

    // Camera Fragment
    @Override
    protected int getLayoutId() {
        return R.layout.camera_fragment;
    }

    // Standard Image Size
    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

}
