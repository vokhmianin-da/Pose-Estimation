package com.example.virtualcoach.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.example.virtualcoach.data.PoseData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class VideoProcessor(private val context: Context) {

    private val poseDetector = PoseDetector(context)

    fun extractPoses(
        videoUri: Uri,
        frameStep: Int = 10,
        onProgress: (Int) -> Unit
    ): Flow<List<PoseData>> = flow {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(context, videoUri)
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
            // Используем фиксированную частоту кадров, так как METADATA_KEY_FRAMERATE может отсутствовать
            val fps = 30.0
            val totalFrames = (durationMs * fps / 1000).toInt()

            val poses = mutableListOf<PoseData>()
            for (frameIndex in 0 until totalFrames step frameStep) {
                val timeUs = (frameIndex * 1_000_000L / fps).toLong()
                val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if (bitmap != null) {
                    val rotatedBitmap = rotateBitmap(bitmap, rotation)
                    val pose = poseDetector.detectPose(rotatedBitmap)
                    if (pose != null) {
                        val timestamp = frameIndex / fps
                        poses.add(pose.copy(timestamp = timestamp))
                    }
                }
                onProgress((frameIndex * 100) / totalFrames)
            }
            emit(poses)
        } finally {
            retriever.release()
        }
    }.flowOn(Dispatchers.IO)

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}