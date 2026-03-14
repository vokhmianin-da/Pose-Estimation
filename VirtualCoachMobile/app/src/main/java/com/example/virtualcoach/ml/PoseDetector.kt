package com.example.virtualcoach.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.example.virtualcoach.data.PoseData
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import kotlin.math.abs

class PoseDetector(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    init {
        try {
            val modelPath = assetFilePath(context, "keypoint_rcnn.onnx")
            session = env.createSession(modelPath)

            val inputInfo = session?.inputInfo
            val outputInfo = session?.outputInfo

//            Log.d("PoseDetector", "=== Model Info ===")
//            Log.d("PoseDetector", "Input names: ${inputInfo?.keys}")
//            Log.d("PoseDetector", "Output names: ${outputInfo?.keys}")

        } catch (e: OrtException) {
            Log.e("PoseDetector", "ONNX Runtime error", e)
        } catch (e: IOException) {
            Log.e("PoseDetector", "Model file error", e)
        }
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            throw RuntimeException("Не удалось скопировать модель из assets", e)
        }
    }

    fun detectPose(bitmap: Bitmap): PoseData? {
        val sess = session ?: return null

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        val pixels = IntArray(640 * 640)
        resizedBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val inputBuffer = FloatBuffer.allocate(1 * 3 * 640 * 640)
        for (i in pixels.indices) {
            val r = ((pixels[i] shr 16) and 0xFF) / 255f
            val g = ((pixels[i] shr 8) and 0xFF) / 255f
            val b = (pixels[i] and 0xFF) / 255f
            inputBuffer.put((r - mean[0]) / std[0])
            inputBuffer.put((g - mean[1]) / std[1])
            inputBuffer.put((b - mean[2]) / std[2])
        }
        inputBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, 640, 640))
        val inputs = mapOf("input" to inputTensor)
        val results = sess.run(inputs)

        val scoresTensor = results.get("scores")?.get() as? OnnxTensor
        val keypointsTensor = results.get("keypoints")?.get() as? OnnxTensor
        val keypointsScoresTensor = results.get("keypoints_scores")?.get() as? OnnxTensor

        if (scoresTensor == null) {
            Log.e("PoseDetector", "scoresTensor is null")
            return null
        }
        val scoresArray = scoresTensor.floatBuffer.array()
        Log.d("PoseDetector", "scores size = ${scoresArray.size}")
        if (scoresArray.isNotEmpty()) {
//            Log.d("PoseDetector", "scores (first 10) = ${scoresArray.take(10).joinToString()}")
//            Log.d("PoseDetector", "max score = ${scoresArray.maxOrNull()}")
        } else {
            Log.e("PoseDetector", "scoresArray is empty")
            return null
        }

        val threshold = 0.1f // временно низкий порог
        val bestIdx = scoresArray.indices.find { scoresArray[it] > threshold }
        if (bestIdx == null) {
            Log.w("PoseDetector", "No detection above threshold $threshold")
            return null
        }
//        Log.d("PoseDetector", "bestIdx = $bestIdx, score = ${scoresArray[bestIdx]}")

        if (keypointsTensor == null || keypointsScoresTensor == null) {
            Log.e("PoseDetector", "keypointsTensor or keypointsScoresTensor is null")
            return null
        }
        val keypointsArray = keypointsTensor.floatBuffer.array()
        val kpScoresArray = keypointsScoresTensor.floatBuffer.array()
//        Log.d("PoseDetector", "kpScoresArray size = ${kpScoresArray.size}")
        if (kpScoresArray.isNotEmpty()) {
            Log.d("PoseDetector", "First 5 kpScores: ${kpScoresArray.take(5).joinToString()}")
        }

        val expectedKpSize = scoresArray.size * 17 * 3
        if (keypointsArray.size != expectedKpSize) {
            Log.e("PoseDetector", "keypoints size mismatch: expected $expectedKpSize, got ${keypointsArray.size}")
            return null
        }
        val expectedKpScoreSize = scoresArray.size * 17
        if (kpScoresArray.size != expectedKpScoreSize) {
            Log.e("PoseDetector", "kpScores size mismatch: expected $expectedKpScoreSize, got ${kpScoresArray.size}")
            return null
        }

        val kp = Array(17) { i ->
            val baseIdx = bestIdx * 17 * 3 + i * 3
            val x = keypointsArray[baseIdx]
            val y = keypointsArray[baseIdx + 1]
            val scaleX = bitmap.width.toFloat() / 640f
            val scaleY = bitmap.height.toFloat() / 640f
            PointF(x * scaleX, y * scaleY)
        }

        val conf = FloatArray(17) { i ->
            kpScoresArray[bestIdx * 17 + i]
        }

        val positiveConf = conf.map { abs(it) }.toFloatArray()

//        Log.d("PoseDetector", "conf (first 5): ${conf.take(5).joinToString()}")

        return PoseData(kp, positiveConf, 0.0)
    }
}