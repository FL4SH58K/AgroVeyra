package com.agroveyra.app.utils

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.json.JSONArray
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Tensor
import java.io.Closeable
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

class TFLiteHelper(context: Context) : Closeable {

	private val appContext = context.applicationContext
	private var interpreter: Interpreter? = null
	private var classNames: List<String> = emptyList()

	companion object {
		private const val TAG = "TFLiteHelper"
		private const val MODEL_FILE_NAME = "agroveyra_model.tflite"
		private const val CLASS_NAMES_FILE_NAME = "class_names.json"
		private const val INPUT_SIZE = 224
		private const val BYTES_PER_CHANNEL = 4
		private const val NUM_CHANNELS = 3

		/**
		 * Below this top-1 softmax score the classifier is guessing rather than reading the leaf.
		 * Measured on the 16 real field photos (ml/benchmark_preprocessing.py, ml/class_error_report.txt):
		 * 43.8% precision with no gate, 87.5% at 0.70 and 100% at 0.80 - so the UI asks for a retake
		 * instead of presenting a diagnosis it cannot stand behind.
		 */
		const val MIN_CONFIDENCE = 0.70f
	}

	init {
		try {
			val model = loadModelFile(MODEL_FILE_NAME)
			interpreter = Interpreter(model, Interpreter.Options().apply {
				setNumThreads(4)
			})
			classNames = loadClassNames(CLASS_NAMES_FILE_NAME)
			Log.i(TAG, "TFLite model loaded successfully with ${classNames.size} classes")
		} catch (exception: Exception) {
			Log.e(TAG, "Failed to initialize TFLiteHelper", exception)
			interpreter = null
			classNames = emptyList()
		}
	}

	private fun loadModelFile(fileName: String): MappedByteBuffer {
		val assetFileDescriptor = appContext.assets.openFd(fileName)
		FileInputStream(assetFileDescriptor.fileDescriptor).use { inputStream ->
			val fileChannel = inputStream.channel
			return fileChannel.map(
				FileChannel.MapMode.READ_ONLY,
				assetFileDescriptor.startOffset,
				assetFileDescriptor.declaredLength
			)
		}
	}

	private fun loadClassNames(fileName: String): List<String> {
		return appContext.assets.open(fileName).use { inputStream ->
			val json = inputStream.bufferedReader().use { it.readText() }
			val jsonArray = JSONArray(json)
			buildList {
				for (index in 0 until jsonArray.length()) {
					add(jsonArray.getString(index))
				}
			}
		}
	}

	fun classify(bitmap: Bitmap): Pair<String, Float>? {
		val currentInterpreter = interpreter ?: return null

		return try {
			val inferenceStart = SystemClock.elapsedRealtimeNanos()
			val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
			val inputBuffer = convertBitmapToInputBuffer(resizedBitmap, currentInterpreter)
			val outputTensor = currentInterpreter.getOutputTensor(0)
			val predictions = runInference(currentInterpreter, inputBuffer, outputTensor)
			val bestIndex = predictions.indices.maxByOrNull { predictions[it] } ?: return null
			val confidence = predictions[bestIndex]
			val prediction = classNames.getOrNull(bestIndex) ?: return null

			val inferenceTimeMs = (SystemClock.elapsedRealtimeNanos() - inferenceStart) / 1_000_000.0
			Log.d(TAG, "Inference completed in ${"%.2f".format(inferenceTimeMs)} ms")

			prediction to confidence
		} catch (exception: Exception) {
			Log.e(TAG, "Classification failed", exception)
			null
		}
	}

	private fun convertBitmapToInputBuffer(bitmap: Bitmap, interpreter: Interpreter): Any {
		val inputTensor = interpreter.getInputTensor(0)
		val shape = inputTensor.shape()
		val dataType = inputTensor.dataType()

		val floatBuffer = ByteBuffer
			.allocateDirect(INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * BYTES_PER_CHANNEL)
			.order(ByteOrder.nativeOrder())

		val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
		bitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

		for (pixelValue in intValues) {
			val red = ((pixelValue shr 16) and 0xFF) / 255.0f
			val green = ((pixelValue shr 8) and 0xFF) / 255.0f
			val blue = (pixelValue and 0xFF) / 255.0f
			floatBuffer.putFloat(red)
			floatBuffer.putFloat(green)
			floatBuffer.putFloat(blue)
		}

		floatBuffer.rewind()

		return when (dataType) {
			DataType.FLOAT32 -> floatBuffer
			DataType.UINT8, DataType.INT8 -> quantizeBuffer(floatBuffer, inputTensor)
			else -> throw IllegalStateException("Unsupported input tensor type: $dataType with shape ${shape.contentToString()}")
		}
	}

	private fun quantizeBuffer(floatBuffer: ByteBuffer, inputTensor: Tensor): ByteBuffer {
		val quantizedBuffer = ByteBuffer.allocateDirect(floatBuffer.capacity() / BYTES_PER_CHANNEL).order(ByteOrder.nativeOrder())
		val quantizationParams = inputTensor.quantizationParams()
		val scale = quantizationParams.scale
		val zeroPoint = quantizationParams.zeroPoint

		while (floatBuffer.remaining() >= BYTES_PER_CHANNEL) {
			val value = floatBuffer.float
			val quantizedValue = ((value / scale) + zeroPoint).roundToInt()
			quantizedBuffer.put(quantizedValue.toByte())
		}

		quantizedBuffer.rewind()
		floatBuffer.rewind()
		return quantizedBuffer
	}

	private fun runInference(interpreter: Interpreter, inputBuffer: Any, outputTensor: Tensor): FloatArray {
		val outputShape = outputTensor.shape()
		val classCount = outputShape.lastOrNull()?.takeIf { it > 0 } ?: classNames.size.coerceAtLeast(1)

		return when (outputTensor.dataType()) {
			DataType.FLOAT32 -> {
				val outputBuffer = Array(1) { FloatArray(classCount) }
				interpreter.run(inputBuffer, outputBuffer)
				outputBuffer[0]
			}

			DataType.UINT8, DataType.INT8 -> {
				val outputBuffer = ByteBuffer.allocateDirect(classCount).order(ByteOrder.nativeOrder())
				interpreter.run(inputBuffer, outputBuffer)
				outputBuffer.rewind()

				val quantParams = outputTensor.quantizationParams()
				val scale = quantParams.scale
				val zeroPoint = quantParams.zeroPoint
				FloatArray(classCount) { index ->
					val rawValue = outputBuffer.get().toInt()
					val unsignedValue = if (outputTensor.dataType() == DataType.UINT8) rawValue and 0xFF else rawValue
					(unsignedValue - zeroPoint) * scale
				}
			}

			else -> throw IllegalStateException("Unsupported output tensor type: ${outputTensor.dataType()}")
		}
	}

	override fun close() {
		try {
			interpreter?.close()
		} catch (exception: Exception) {
			Log.e(TAG, "Error closing TFLite interpreter", exception)
		} finally {
			interpreter = null
		}
	}
}

