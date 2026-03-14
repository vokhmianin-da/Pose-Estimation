package com.example.virtualcoach

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.virtualcoach.databinding.ActivityResultBinding
import java.io.File

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private var videoPath: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val meanCos = intent.getDoubleExtra("MEAN_COS", 0.0)
        val meanWdist = intent.getDoubleExtra("MEAN_WDIST", 0.0)
        val grade = intent.getStringExtra("GRADE") ?: ""
        videoPath = intent.getStringExtra("OUTPUT_VIDEO")

        binding.tvMeanCos.text = "Среднее косинусное сходство: %.4f".format(meanCos)
        binding.tvMeanWdist.text = "Среднее взвешенное расстояние: %.2f".format(meanWdist)
        binding.tvGrade.text = "Оценка: $grade"

        binding.btnPlayVideo.setOnClickListener {
            playVideo()
        }

        binding.btnShare.setOnClickListener {
            shareVideo()
        }

        if (videoPath == null || !File(videoPath!!).exists()) {
            binding.btnPlayVideo.isEnabled = false
            binding.btnShare.isEnabled = false
            binding.btnPlayVideo.text = "Видео не сохранено"
        }
    }

    private fun playVideo() {
        val file = File(videoPath ?: return)
        if (!file.exists()) {
            Toast.makeText(this, "Видео не найдено", Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "video/mp4")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(intent)
    }

    private fun shareVideo() {
        val file = File(videoPath ?: return)
        if (!file.exists()) {
            Toast.makeText(this, "Видео не найдено", Toast.LENGTH_SHORT).show()
            return
        }
        val uri = FileProvider.getUriForFile(this, "${packageName}.fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "video/mp4"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, "Поделиться видео"))
    }
}