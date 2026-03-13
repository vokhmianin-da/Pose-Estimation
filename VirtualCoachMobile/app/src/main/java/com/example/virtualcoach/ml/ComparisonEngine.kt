package com.example.virtualcoach.ml

import android.graphics.PointF
import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.LUDecomposition
import org.apache.commons.math3.linear.RealMatrix
import kotlin.math.abs
import kotlin.math.sqrt

object ComparisonEngine {

    fun interpolatePoints(
        ptsList: List<Array<PointF>>,
        timestamps: List<Double>,
        newTimestamps: List<Double>
    ): List<Array<PointF>> {
        if (ptsList.isEmpty()) return emptyList()
        val result = mutableListOf<Array<PointF>>()
        for (t in newTimestamps) {
            when {
                t <= timestamps.first() -> result.add(ptsList.first())
                t >= timestamps.last() -> result.add(ptsList.last())
                else -> {
                    val idx = timestamps.indexOfFirst { it > t }
                    val t0 = timestamps[idx - 1]
                    val t1 = timestamps[idx]
                    val pt0 = ptsList[idx - 1]
                    val pt1 = ptsList[idx]
                    val alpha = ((t - t0) / (t1 - t0)).toFloat()
                    val interp = Array(17) { i ->
                        PointF(
                            pt0[i].x + alpha * (pt1[i].x - pt0[i].x),
                            pt0[i].y + alpha * (pt1[i].y - pt0[i].y)
                        )
                    }
                    result.add(interp)
                }
            }
        }
        return result
    }

    fun interpolateConfidences(
        confList: List<FloatArray>,
        timestamps: List<Double>,
        newTimestamps: List<Double>
    ): List<FloatArray> {
        if (confList.isEmpty()) return emptyList()
        val result = mutableListOf<FloatArray>()
        for (t in newTimestamps) {
            when {
                t <= timestamps.first() -> result.add(confList.first())
                t >= timestamps.last() -> result.add(confList.last())
                else -> {
                    val idx = timestamps.indexOfFirst { it > t }
                    val t0 = timestamps[idx - 1]
                    val t1 = timestamps[idx]
                    if (t - t0 < t1 - t) {
                        result.add(confList[idx - 1])
                    } else {
                        result.add(confList[idx])
                    }
                }
            }
        }
        return result
    }

    fun getTransform(src: List<PointF>, dst: List<PointF>): RealMatrix {
        val n = src.size
        val X = Array2DRowRealMatrix(n, 3)
        val Y = Array2DRowRealMatrix(n, 3)
        for (i in 0 until n) {
            X.setEntry(i, 0, src[i].x.toDouble())
            X.setEntry(i, 1, src[i].y.toDouble())
            X.setEntry(i, 2, 1.0)
            Y.setEntry(i, 0, dst[i].x.toDouble())
            Y.setEntry(i, 1, dst[i].y.toDouble())
            Y.setEntry(i, 2, 1.0)
        }
        val Xt = X.transpose()
        val XtX = Xt.multiply(X)
        // LUDecomposition для нахождения обратной матрицы
        val inverse = LUDecomposition(XtX).solver.inverse
        val XtY = Xt.multiply(Y)
        return inverse.multiply(XtY)
    }

    fun applyTransform(pts: List<PointF>, A: RealMatrix): List<PointF> {
        return pts.map { pt ->
            // Вектор-столбец 3x1: [x, y, 1]
            val vec = Array2DRowRealMatrix(arrayOf(
                doubleArrayOf(pt.x.toDouble()),
                doubleArrayOf(pt.y.toDouble()),
                doubleArrayOf(1.0)
            ))
            val res = A.multiply(vec)
            // Извлекаем первые две координаты (x, y)
            PointF(res.getEntry(0, 0).toFloat(), res.getEntry(1, 0).toFloat())
        }
    }

    fun cosineDistance(pose1: FloatArray, pose2: FloatArray): Double {
        var dot = 0.0
        var norm1 = 0.0
        var norm2 = 0.0
        for (i in pose1.indices) {
            dot += pose1[i] * pose2[i]
            norm1 += pose1[i] * pose1[i]
            norm2 += pose2[i] * pose2[i]
        }
        if (norm1 == 0.0 || norm2 == 0.0) return 0.0
        return dot / (sqrt(norm1) * sqrt(norm2))
    }

    fun weightedDistance(pose1: FloatArray, pose2: FloatArray, conf: FloatArray): Double {
        var sum = 0.0
        var sumConf = 0.0
        for (i in pose1.indices) {
            val kpIdx = i / 2
            sum += conf[kpIdx] * abs(pose1[i] - pose2[i])
            sumConf += conf[kpIdx]
        }
        return if (sumConf > 0) sum / sumConf else Double.POSITIVE_INFINITY
    }

    fun pointsToFlatArray(points: Array<PointF>): FloatArray {
        val arr = FloatArray(34)
        for (i in points.indices) {
            arr[2 * i] = points[i].x
            arr[2 * i + 1] = points[i].y
        }
        return arr
    }
}