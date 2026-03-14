package com.example.virtualcoach

import android.content.Intent
import android.graphics.PointF
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.virtualcoach.databinding.ActivityProcessingBinding
import com.example.virtualcoach.data.PoseData
import com.example.virtualcoach.ml.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.first
import java.io.File

class ProcessingActivity : AppCompatActivity() {

    private lateinit var binding: ActivityProcessingBinding
    private lateinit var refUri: Uri
    private lateinit var userUri: Uri
    private var showSkeleton: Boolean = false


    data class ComparisonResult(
        val meanCos: Double,
        val meanWdist: Double,
        val grade: String,
        val frameDataMap: Map<Int, VideoExporter.FrameData>? = null,
        var outputVideoPath: String? = null
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityProcessingBinding.inflate(layoutInflater)
        setContentView(binding.root)

        refUri = Uri.parse(intent.getStringExtra("REF_URI"))
        userUri = Uri.parse(intent.getStringExtra("USER_URI"))
        showSkeleton = intent.getBooleanExtra("SHOW_SKELETON", true)

        lifecycleScope.launch {
            processVideos()
        }
    }

    private suspend fun processVideos() = withContext(Dispatchers.IO) {
        val processor = VideoProcessor(this@ProcessingActivity)

        withContext(Dispatchers.Main) {
            updateProgress(0, "Извлечение ключевых точек из эталонного видео...")
        }

        val refPoses = processor.extractPoses(refUri, frameStep = 10) { percent ->
            runOnUiThread { binding.progressBar.progress = percent / 2 }
        }.first()

        withContext(Dispatchers.Main) {
            updateProgress(50, "Извлечение ключевых точек из вашего видео...")
        }

        val userPoses = processor.extractPoses(userUri, frameStep = 10) { percent ->
            runOnUiThread { binding.progressBar.progress = 50 + percent / 2 }
        }.first()

        if (refPoses.isEmpty() || userPoses.isEmpty()) {
            withContext(Dispatchers.Main) {
                Toast.makeText(this@ProcessingActivity, "Не удалось обнаружить человека в одном из видео", Toast.LENGTH_LONG).show()
                finish()
            }
            return@withContext
        }

        withContext(Dispatchers.Main) {
            updateProgress(100, "Сравнение и создание видео...")
        }

        val result = comparePoses(refPoses, userPoses)

        var videoPath: String? = null
        if (result.frameDataMap != null) {
            try {
                videoPath = exportResultVideo(result)
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@ProcessingActivity, "Ошибка создания видео: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }

        val finalVideoPath = videoPath
        withContext(Dispatchers.Main) {
            val intent = Intent(this@ProcessingActivity, ResultActivity::class.java).apply {
                putExtra("MEAN_COS", result.meanCos)
                putExtra("MEAN_WDIST", result.meanWdist)
                putExtra("GRADE", result.grade)
                putExtra("OUTPUT_VIDEO", finalVideoPath)
            }
            startActivity(intent)
            finish()
        }
    }

    private suspend fun comparePoses(refPoses: List<PoseData>, userPoses: List<PoseData>): ComparisonResult {
        val refTimes = refPoses.map { it.timestamp }
        val userTimes = userPoses.map { it.timestamp }

        val fps = 30.0
        val maxTime = minOf(refTimes.last(), userTimes.last())
        val commonTimestamps = (0..(maxTime * fps).toInt()).map { it / fps }

        val refPoints = refPoses.map { it.keypoints }
        val userPoints = userPoses.map { it.keypoints }
        val userConf = userPoses.map { it.confidences }

        val refInterp = ComparisonEngine.interpolatePoints(refPoints, refTimes, commonTimestamps)
        val userInterp = ComparisonEngine.interpolatePoints(userPoints, userTimes, commonTimestamps)
        val userConfInterp = ComparisonEngine.interpolateConfidences(userConf, userTimes, commonTimestamps)

        val cosScores = mutableListOf<Double>()
        val wDistScores = mutableListOf<Double>()
        val transformsRefToUser = mutableListOf<org.apache.commons.math3.linear.RealMatrix>()

        for (i in commonTimestamps.indices) {
            val ref = refInterp[i]
            val user = userInterp[i]
            val conf = userConfInterp[i]

            val A = ComparisonEngine.getTransform(user.toList(), ref.toList())
            val userAligned = ComparisonEngine.applyTransform(user.toList(), A)
            val userFlat = ComparisonEngine.pointsToFlatArray(userAligned.toTypedArray())
            val refFlat = ComparisonEngine.pointsToFlatArray(ref)
            cosScores.add(ComparisonEngine.cosineDistance(userFlat, refFlat))
            wDistScores.add(ComparisonEngine.weightedDistance(userFlat, refFlat, conf))

            val B = ComparisonEngine.getTransform(ref.toList(), user.toList())
            transformsRefToUser.add(B)
        }

        val meanCos = cosScores.average()
        val meanWdist = wDistScores.average()
        val grade = getGrade(meanCos, meanWdist)

        // Создаём map с VideoExporter.FrameData
        val frameDataMap = mutableMapOf<Int, VideoExporter.FrameData>()
        for (i in commonTimestamps.indices) {
            val frameIdx = (commonTimestamps[i] * fps).toInt()
            val refOnUser = ComparisonEngine.applyTransform(refInterp[i].toList(), transformsRefToUser[i])
            frameDataMap[frameIdx] = VideoExporter.FrameData(
                refPoints = refOnUser.toTypedArray(),
                cos = cosScores[i],
                wdist = wDistScores[i]
            )
        }

        return ComparisonResult(
            meanCos = meanCos,
            meanWdist = meanWdist,
            grade = grade,
            frameDataMap = frameDataMap
        )
    }

    private suspend fun exportResultVideo(result: ComparisonResult): String {
        val outputFile = File(cacheDir, "result_${System.currentTimeMillis()}.mp4")
        val exporter = VideoExporter(this@ProcessingActivity)
        exporter.exportVideo(
            userVideoUri = userUri,
            fps = 30.0,
            frameDataMap = result.frameDataMap ?: emptyMap(), // теперь типы совпадают
            outputPath = outputFile.absolutePath,
            showSkeleton = showSkeleton
        ) { current, total ->
            runOnUiThread {
                // Можно обновлять прогресс создания видео
                binding.progressBar.progress = 100 // или использовать второй прогресс-бар
            }
        }
        return outputFile.absolutePath
    }

    private fun getGrade(meanCos: Double, meanWdist: Double): String {
        return when {
            meanCos > 0.98 && meanWdist < 20 -> "Excellent"
            meanCos > 0.95 && meanWdist < 30 -> "Good"
            meanCos > 0.9 && meanWdist < 50 -> "Average"
            else -> "Poor"
        }
    }

    private fun updateProgress(percent: Int, status: String) {
        binding.progressBar.progress = percent
        binding.tvProgress.text = "$percent%"
        binding.tvStatus.text = status
    }
}