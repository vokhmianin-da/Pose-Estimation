package com.example.virtualcoach.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.example.virtualcoach.data.PoseData
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class PoseDetector(private val context: Context) {

    private var module: Module? = null

    init {
        try {
            System.loadLibrary("torchvision_ops")
        } catch (e: UnsatisfiedLinkError) {
            e.printStackTrace()
            // Если библиотеки нет, здесь будет исключение
        }
        module = Module.load(assetFilePath(context, "keypoint_rcnn_scripted.pt"))
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
        val model = module ?: return null

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val output = model.forward(IValue.from(inputTensor)).toTuple()
        val detectionsList = output[1].toList()
        if (detectionsList.isEmpty()) return null

        val detectionDict = detectionsList[0].toDictStringKey()

        val scores = detectionDict["scores"]?.toTensor()?.dataAsFloatArray ?: return null
        val keypointsTensor = detectionDict["keypoints"]?.toTensor() ?: return null
        val keypointsScoresTensor = detectionDict["keypoints_scores"]?.toTensor() ?: return null

        val threshold = 0.9f
        val bestIdx = scores.indices.find { index -> scores[index] > threshold } ?: return null

        val keypointsArray = keypointsTensor.dataAsFloatArray
        val kpScoresArray = keypointsScoresTensor.dataAsFloatArray

        val keypoints = Array(17) { i ->
            val baseIdx = bestIdx * 17 * 3 + i * 3
            val x = keypointsArray[baseIdx]
            val y = keypointsArray[baseIdx + 1]
            val scaleX = bitmap.width.toFloat() / 640f
            val scaleY = bitmap.height.toFloat() / 640f
            PointF(x * scaleX, y * scaleY)
        }

        val confidences = FloatArray(17) { i ->
            kpScoresArray[bestIdx * 17 + i]
        }

        return PoseData(keypoints, confidences, 0.0)
    }
}