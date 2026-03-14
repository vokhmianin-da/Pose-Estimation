package com.example.virtualcoach.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.example.virtualcoach.data.PoseData
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class PoseDetector(private val context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null

    init {
        session = env.createSession(assetFilePath(context, "keypoint_rcnn.onnx"))
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

        // Ресайз до 640x640
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        // Преобразование в тензор (1,3,640,640) с нормализацией ImageNet
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

        // Извлекаем выходные тензоры
        val scores = results.get("scores").get() as OnnxTensor
        val keypoints = results.get("keypoints").get() as OnnxTensor
        val keypointsScores = results.get("keypoints_scores").get() as OnnxTensor

        val scoresArray = scores.floatBuffer.array()
        val keypointsArray = keypoints.floatBuffer.array()
        val kpScoresArray = keypointsScores.floatBuffer.array()

        val threshold = 0.9f
        val bestIdx = scoresArray.indices.find { scoresArray[it] > threshold } ?: return null

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

        return PoseData(kp, conf, 0.0)
    }
}