package com.agroveyra.app.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object ImageUtils {
    fun uriToBitmap(context: Context, uri: Uri): Bitmap {
        context.contentResolver.openInputStream(uri).use { inputStream ->
            if (inputStream == null) {
                throw IOException("Unable to open input stream for URI: $uri")
            }
            return BitmapFactory.decodeStream(inputStream)
                ?: throw IOException("Unable to decode bitmap from URI: $uri")
        }
    }

    fun rotateBitmapIfRequired(bitmap: Bitmap, uri: Uri, context: Context): Bitmap {
        val rotation = context.contentResolver.openInputStream(uri)?.use { inputStream ->
            val exif = ExifInterface(inputStream)
            when (exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90
                ExifInterface.ORIENTATION_ROTATE_180 -> 180
                ExifInterface.ORIENTATION_ROTATE_270 -> 270
                else -> 0
            }
        } ?: 0

        if (rotation == 0) {
            return bitmap
        }

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun saveBitmapToCache(context: Context, bitmap: Bitmap): String {
        val cacheFile = File(context.cacheDir, "agroveyra_${System.currentTimeMillis()}.jpg")
        FileOutputStream(cacheFile).use { outputStream ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
            outputStream.flush()
        }
        return cacheFile.absolutePath
    }

    fun getResizedBitmap(bitmap: Bitmap, maxDimension: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        if (width <= maxDimension && height <= maxDimension) {
            return bitmap
        }

        val scale = if (width > height) {
            maxDimension.toFloat() / width.toFloat()
        } else {
            maxDimension.toFloat() / height.toFloat()
        }

        val newWidth = (width * scale).toInt().coerceAtLeast(1)
        val newHeight = (height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
}

