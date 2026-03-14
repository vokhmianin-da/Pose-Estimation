package com.example.virtualcoach.ml

import android.content.Context
import android.graphics.*
import android.media.*
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer

class VideoExporter(private val context: Context) {

    private val limbs = listOf(
        Pair(0, 1), Pair(0, 2), Pair(1, 3), Pair(2, 4),
        Pair(5, 7), Pair(7, 9), Pair(6, 8), Pair(8, 10),
        Pair(11, 13), Pair(13, 15), Pair(12, 14), Pair(14, 16),
        Pair(5, 6), Pair(11, 12), Pair(5, 11), Pair(6, 12)
    )

    data class FrameData(
        val refPoints: Array<PointF>,
        val cos: Double,
        val wdist: Double
    )

    suspend fun exportVideo(
        userVideoUri: Uri,
        fps: Double,
        frameDataMap: Map<Int, FrameData>,
        outputPath: String,
        showSkeleton: Boolean,
        progressCallback: (Int, Int) -> Unit
    ) = withContext(Dispatchers.IO) {
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, userVideoUri)

        val videoWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 640
        val videoHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 480
        val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
        val durationUs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong()?.times(1000) ?: 0
        val totalFrames = (durationUs * fps / 1_000_000).toInt()

        val outputFile = File(outputPath)
        val mediaMuxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val mediaFormat = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, videoWidth, videoHeight).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, 2_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, fps.toInt())
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }

        val mediaCodec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        mediaCodec.configure(mediaFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val inputSurface = mediaCodec.createInputSurface()
        mediaCodec.start()

        var trackIndex = -1
        val bufferInfo = MediaCodec.BufferInfo()
        var presentationTimeUs = 0L
        val frameDurationUs = (1_000_000 / fps).toLong()

        fun encodeFrame(bitmap: Bitmap) {
            val canvas = inputSurface?.lockCanvas(null)
            if (canvas != null) {
                canvas.drawBitmap(bitmap, 0f, 0f, null)
                inputSurface.unlockCanvasAndPost(canvas)
            }

            while (true) {
                val outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 10_000L)
                when {
                    outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (trackIndex == -1) {
                            trackIndex = mediaMuxer.addTrack(mediaCodec.outputFormat)
                            mediaMuxer.start()
                        }
                    }
                    outputBufferIndex >= 0 -> {
                        val outputBuffer = mediaCodec.getOutputBuffer(outputBufferIndex)
                        if (outputBuffer != null && bufferInfo.size > 0) {
                            outputBuffer.position(bufferInfo.offset)
                            outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                            val sampleData = ByteBuffer.allocate(bufferInfo.size)
                            sampleData.put(outputBuffer)
                            sampleData.flip()
                            mediaMuxer.writeSampleData(trackIndex, sampleData, bufferInfo)
                        }
                        mediaCodec.releaseOutputBuffer(outputBufferIndex, false)
                    }
                    outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> break
                }
            }
        }

        for (frameIdx in 0 until totalFrames) {
            val timeUs = (frameIdx * 1_000_000L / fps).toLong()
            val bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: continue
            val rotatedBitmap = rotateBitmap(bitmap, rotation)

            val workingBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(workingBitmap)

            val data = frameDataMap[frameIdx]
            if (data != null) {
                if (showSkeleton) {
                    drawSkeleton(canvas, data.refPoints, Color.GREEN)
                }
                drawText(canvas, data)
            }

            encodeFrame(workingBitmap)
            presentationTimeUs += frameDurationUs

            progressCallback(frameIdx + 1, totalFrames)
        }

        retriever.release()
        mediaCodec.signalEndOfInputStream()

        while (true) {
            val outputBufferIndex = mediaCodec.dequeueOutputBuffer(bufferInfo, 10_000L)
            when {
                outputBufferIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* уже обработано */ }
                outputBufferIndex >= 0 -> {
                    val outputBuffer = mediaCodec.getOutputBuffer(outputBufferIndex)
                    if (outputBuffer != null && bufferInfo.size > 0) {
                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)
                        val sampleData = ByteBuffer.allocate(bufferInfo.size)
                        sampleData.put(outputBuffer)
                        sampleData.flip()
                        mediaMuxer.writeSampleData(trackIndex, sampleData, bufferInfo)
                    }
                    mediaCodec.releaseOutputBuffer(outputBufferIndex, false)
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
                outputBufferIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        }

        val summaryBitmap = createSummaryFrame(videoWidth, videoHeight, frameDataMap)
        for (i in 0 until (fps * 3).toInt()) {
            encodeFrame(summaryBitmap)
            presentationTimeUs += frameDurationUs
        }

        mediaCodec.stop()
        mediaCodec.release()
        if (trackIndex != -1) {
            mediaMuxer.stop()
        }
        mediaMuxer.release()
        inputSurface?.release()
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(degrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun drawSkeleton(canvas: Canvas, keypoints: Array<PointF>, color: Int) {
        val paint = Paint().apply {
            this.color = color
            strokeWidth = 8f
            style = Paint.Style.STROKE
        }
        val pointPaint = Paint().apply {
            this.color = color
            style = Paint.Style.FILL
        }
        for ((i, j) in limbs) {
            val p1 = keypoints[i]
            val p2 = keypoints[j]
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, paint)
        }
        for (pt in keypoints) {
            canvas.drawCircle(pt.x, pt.y, 10f, pointPaint)
        }
    }

    private fun drawText(canvas: Canvas, data: FrameData) {
        val paintGreen = Paint().apply {
            color = Color.GREEN
            textSize = 40f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val paintRed = Paint().apply {
            color = Color.RED
            textSize = 40f
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        canvas.drawText("Cos: %.3f".format(data.cos), 50f, 100f, paintGreen)
        canvas.drawText("WDist: %.2f".format(data.wdist), 50f, 160f, paintRed)
    }

    private fun createSummaryFrame(width: Int, height: Int, frameDataMap: Map<Int, FrameData>): Bitmap {
        val cosValues = frameDataMap.values.map { it.cos }
        val wdistValues = frameDataMap.values.map { it.wdist }
        val meanCos = cosValues.average()
        val meanWdist = wdistValues.average()
        val grade = getGrade(meanCos, meanWdist)

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.BLACK)

        val paint = Paint().apply {
            color = Color.WHITE
            textSize = 60f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            textAlign = Paint.Align.CENTER
        }

        val centerX = width / 2f
        canvas.drawText("Mean Cosine: %.4f".format(meanCos), centerX, height / 2f - 100, paint)
        canvas.drawText("Mean WDist: %.2f".format(meanWdist), centerX, height / 2f - 20, paint)
        canvas.drawText("Grade: $grade", centerX, height / 2f + 60, paint)

        return bitmap
    }

    private fun getGrade(meanCos: Double, meanWdist: Double): String {
        return when {
            meanCos > 0.98 && meanWdist < 20 -> "Excellent"
            meanCos > 0.95 && meanWdist < 30 -> "Good"
            meanCos > 0.9 && meanWdist < 50 -> "Average"
            else -> "Poor"
        }
    }
}