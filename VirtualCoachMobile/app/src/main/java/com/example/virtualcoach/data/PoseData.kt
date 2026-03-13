package com.example.virtualcoach.data

import android.graphics.PointF

data class PoseData(
    val keypoints: Array<PointF>,      // 17 точек COCO (x, y)
    val confidences: FloatArray,       // 17 значений достоверности
    val timestamp: Double              // время в секундах
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PoseData

        if (!keypoints.contentEquals(other.keypoints)) return false
        if (!confidences.contentEquals(other.confidences)) return false
        if (timestamp != other.timestamp) return false

        return true
    }

    override fun hashCode(): Int {
        var result = keypoints.contentHashCode()
        result = 31 * result + confidences.contentHashCode()
        result = 31 * result + timestamp.hashCode()
        return result
    }
}