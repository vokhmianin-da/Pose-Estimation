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

    // Локальный data class для хранения данных кадра
    data class FrameData(
        val refPoints: Array<PointF>,
        val cos: Double,
        val wdist: Double
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as FrameData

            if (!refPoints.contentEquals(other.refPoints)) return false
            if (cos != other.cos) return false
            if (wdist != other.wdist) return false

            return true
        }

        override fun hashCode(): Int {
            var result = refPoints.contentHashCode()
            result = 31 * result + cos.hashCode()
            result = 31 * result + wdist.hashCode()
            return result
        }
    }

    data class ComparisonResult(
        val meanCos: Double,
        val meanWdist: Double,
        val grade: String,
        val frameDataMap: Map<Int, FrameData>,
        var outputVideoPath: String = ""
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

    private suspend fun processVideos() = withContext(Dispatchers.Main) {
        updateProgress(0, "Извлечение ключевых точек из эталонного видео...")
        val processor = VideoProcessor(this@ProcessingActivity)

        // Извлекаем позы из референсного видео (используем first() для получения одного списка)
        val refPosesFlow = processor.extractPoses(refUri, frameStep = 10) { percent ->
            runOnUiThread { binding.progressBar.progress = percent / 2 }
        }
        val refPoses = refPosesFlow.first()

        updateProgress(50, "Извлечение ключевых точек из вашего видео...")
        val userPosesFlow = processor.extractPoses(userUri, frameStep = 10) { percent ->
            runOnUiThread { binding.progressBar.progress = 50 + percent / 2 }
        }
        val userPoses = userPosesFlow.first()

        if (refPoses.isEmpty() || userPoses.isEmpty()) {
            Toast.makeText(this@ProcessingActivity, "Не удалось обнаружить человека в одном из видео", Toast.LENGTH_LONG).show()
            finish()
            return@withContext
        }

        updateProgress(100, "Сравнение...")

        // Синхронизация и вычисление метрик
        val result = comparePoses(refPoses, userPoses)

        // Переход к результату (без экспорта видео пока)
        val intent = Intent(this@ProcessingActivity, ResultActivity::class.java).apply {
            putExtra("MEAN_COS", result.meanCos)
            putExtra("MEAN_WDIST", result.meanWdist)
            putExtra("GRADE", result.grade)
            // Видео не передаём, в ResultActivity покажем только метрики
        }
        startActivity(intent)
        finish()
    }

    private suspend fun comparePoses(refPoses: List<PoseData>, userPoses: List<PoseData>): ComparisonResult {
        // Извлекаем временные метки
        val refTimes = refPoses.map { it.timestamp }
        val userTimes = userPoses.map { it.timestamp }

        // Общая временная шкала (используем FPS пользователя)
        val fps = 30.0 // можно получить из видео, но для простоты предположим
        val maxTime = minOf(refTimes.last(), userTimes.last())
        val commonTimestamps = (0..(maxTime * fps).toInt()).map { it / fps }

        // Интерполяция
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

            // Преобразование user -> ref для метрик
            val A = ComparisonEngine.getTransform(user.toList(), ref.toList())
            val userAligned = ComparisonEngine.applyTransform(user.toList(), A)
            val userFlat = ComparisonEngine.pointsToFlatArray(userAligned.toTypedArray())
            val refFlat = ComparisonEngine.pointsToFlatArray(ref)
            cosScores.add(ComparisonEngine.cosineDistance(userFlat, refFlat))
            wDistScores.add(ComparisonEngine.weightedDistance(userFlat, refFlat, conf))

            // Преобразование ref -> user для наложения скелета (пока не используется, но сохраняем)
            val B = ComparisonEngine.getTransform(ref.toList(), user.toList())
            transformsRefToUser.add(B)
        }

        val meanCos = cosScores.average()
        val meanWdist = wDistScores.average()
        val grade = getGrade(meanCos, meanWdist)

        // Строим карту кадров пользователя (по индексам кадров) — пока не используется, но оставим
        val frameDataMap = mutableMapOf<Int, FrameData>()
        for (i in commonTimestamps.indices) {
            val frameIdx = (commonTimestamps[i] * fps).toInt()
            val refOnUser = ComparisonEngine.applyTransform(refInterp[i].toList(), transformsRefToUser[i])
            frameDataMap[frameIdx] = FrameData(
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