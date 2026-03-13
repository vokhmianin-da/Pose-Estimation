package com.example.virtualcoach

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.virtualcoach.databinding.ActivityMainBinding
import com.github.dhaval2404.imagepicker.ImagePicker

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var refVideoUri: Uri? = null
    private var userVideoUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnRefVideo.setOnClickListener { pickVideo(REQUEST_REF) }
        binding.btnUserVideo.setOnClickListener { pickVideo(REQUEST_USER) }
        binding.btnCompare.setOnClickListener {
            if (refVideoUri != null && userVideoUri != null) {
                startProcessing()
            }
        }
    }

    private fun pickVideo(requestCode: Int) {
        ImagePicker.with(this)
            .galleryMimeTypes(  // Ограничиваем выбор видеофайлами
                mimeTypes = arrayOf(
                    "video/mp4",
                    "video/3gpp",
                    "video/x-msvideo",
                    "video/quicktime"
                )
            )
            .start(requestCode)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            REQUEST_REF, REQUEST_USER -> {
                if (resultCode == RESULT_OK && data != null) {
                    val uri = data.data ?: return
                    if (requestCode == REQUEST_REF) {
                        refVideoUri = uri
                        binding.tvRefFileName.text = "Видео выбрано"
                    } else {
                        userVideoUri = uri
                        binding.tvUserFileName.text = "Видео выбрано"
                    }
                    binding.btnCompare.isEnabled = refVideoUri != null && userVideoUri != null
                } else if (resultCode == ImagePicker.RESULT_ERROR) {
                    val error = ImagePicker.getError(data)
                    Toast.makeText(this, "Ошибка: $error", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Выбор отменён", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun startProcessing() {
        val intent = Intent(this, ProcessingActivity::class.java).apply {
            putExtra("REF_URI", refVideoUri.toString())
            putExtra("USER_URI", userVideoUri.toString())
            putExtra("SHOW_SKELETON", binding.cbShowSkeleton.isChecked)
        }
        startActivity(intent)
    }

    companion object {
        private const val REQUEST_REF = 1001
        private const val REQUEST_USER = 1002
    }
}