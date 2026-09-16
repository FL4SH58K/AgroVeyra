package com.agroveyra.app

import android.animation.ValueAnimator
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.animation.LinearInterpolator
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.agroveyra.app.databinding.ActivityScanBinding
import com.agroveyra.app.models.PredictionResult
import com.agroveyra.app.utils.ImageUtils
import com.agroveyra.app.utils.TFLiteHelper
import com.agroveyra.app.utils.TreatmentDatabase
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ScanActivity : AppCompatActivity() {

	private lateinit var binding: ActivityScanBinding
	private lateinit var cameraExecutor: ExecutorService

	private var imageCapture: ImageCapture? = null
	private var camera: Camera? = null
	private var pendingCaptureUri: Uri? = null
	private var currentFlashEnabled = false
	private var pulseAnimator: ValueAnimator? = null
	private var cameraProvider: ProcessCameraProvider? = null

	private val requestCameraPermission =
		registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
			if (granted) {
				startCamera()
			} else {
				showPermissionMessage()
			}
		}

	private val galleryPicker =
		registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
			if (uri != null) {
				handleSelectedImage(uri)
			}
		}

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		binding = ActivityScanBinding.inflate(layoutInflater)
		setContentView(binding.root)

		cameraExecutor = Executors.newSingleThreadExecutor()
		setupUi()

		if (hasCameraPermission()) {
			startCamera()
		} else {
			requestCameraPermission.launch(android.Manifest.permission.CAMERA)
		}
	}

	private fun setupUi() {
		binding.backButton.setOnClickListener { finish() }
		binding.captureButton.setOnClickListener { capturePhoto() }
		binding.galleryButton.setOnClickListener { openGallery() }
		binding.flashButton.setOnClickListener { toggleFlash() }
		startFramePulseAnimation()
	}

	private fun hasCameraPermission(): Boolean {
		return ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
	}

	private fun startCamera() {
		val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
		cameraProviderFuture.addListener({
			try {
				cameraProvider = cameraProviderFuture.get()
				bindCameraUseCases()
			} catch (exception: Exception) {
				showError("Unable to start camera.")
			}
		}, ContextCompat.getMainExecutor(this))
	}

	private fun bindCameraUseCases() {
		val provider = cameraProvider ?: return

		val preview = Preview.Builder().build().also {
			it.setSurfaceProvider(binding.previewView.surfaceProvider)
		}

		imageCapture = ImageCapture.Builder()
			.setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
			.build()

		val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

		try {
			provider.unbindAll()
			camera = provider.bindToLifecycle(
				this,
				cameraSelector,
				preview,
				imageCapture
			)
			applyTorchState()
		} catch (exception: Exception) {
			showError("Failed to bind camera use cases.")
		}
	}

	private fun capturePhoto() {
		val capture = imageCapture ?: run {
			showError("Camera is not ready yet.")
			return
		}

		val tempFile = File(cacheDir, "scan_${System.currentTimeMillis()}.jpg")
		val outputOptions = ImageCapture.OutputFileOptions.Builder(tempFile).build()

		showLoading(true)
		capture.takePicture(
			outputOptions,
			ContextCompat.getMainExecutor(this),
			object : ImageCapture.OnImageSavedCallback {
				override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
					lifecycleScope.launch {
						val resultPath = withContext(Dispatchers.IO) {
							tempFile.absolutePath
						}
						pendingCaptureUri = Uri.fromFile(tempFile)
						runLocalInference(tempFile, resultPath)
					}
				}

				override fun onError(exception: ImageCaptureException) {
					showLoading(false)
					showSnackbar("Photo capture failed. Try again.") {
						capturePhoto()
					}
				}
			}
		)
	}

	private fun openGallery() {
		galleryPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
	}

	private fun handleSelectedImage(uri: Uri) {
		lifecycleScope.launch {
			showLoading(true)
			val file = withContext(Dispatchers.IO) {
				val bitmap = ImageUtils.uriToBitmap(this@ScanActivity, uri)
				val rotated = ImageUtils.rotateBitmapIfRequired(bitmap, uri, this@ScanActivity)
				val resized = ImageUtils.getResizedBitmap(rotated, 1280)
				val path = ImageUtils.saveBitmapToCache(this@ScanActivity, resized)
				java.io.File(path)
			}
			runLocalInference(file, file.absolutePath)
		}
	}

	private fun runLocalInference(file: File, capturedPath: String) {
		lifecycleScope.launch {
			showLoading(true, "Analyzing leaf...")
			val result = withContext(Dispatchers.IO) {
				val bitmap = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
				val tfliteHelper = TFLiteHelper(this@ScanActivity)
				val prediction = tfliteHelper.classify(bitmap)
				tfliteHelper.close()
				prediction
			}

			if (result != null) {
				handlePredictionSuccess(result.first, result.second, capturedPath)
			} else {
				handlePredictionFailure("Prediction failed. Model not ready or processing error.", file)
			}
		}
	}

	private fun handlePredictionSuccess(className: String, confidence: Float, capturedPath: String) {
		showLoading(false)

		// Confidence gate (TFLiteHelper.MIN_CONFIDENCE): a low top-1 score means the model is
		// guessing, so ask for a better photo instead of routing to ResultActivity with a guess.
		if (confidence < TFLiteHelper.MIN_CONFIDENCE) {
			showLowConfidenceRetake(confidence)
			return
		}

		val info = TreatmentDatabase.getInfo(this, className)
		
		val confidencePercent = confidence * 100f
		val stageNumber = if (info.isHealthy) 0 else 1 

		val predictionResult = PredictionResult(
			disease = info.disease,
			confidence = confidencePercent,
			displayName = info.displayName,
			crop = info.crop,
			stage = if (info.isHealthy) "Healthy" else "Early Stage",
			stageNumber = stageNumber,
			stagePercentage = if (info.isHealthy) 0f else 20f,
			chemicalTreatment = info.chemicalTreatment,
			organicTreatment = info.organicTreatment,
			prevention = info.prevention,
			urgency = if (info.isHealthy) "N/A" else "Medium",
			isHealthy = info.isHealthy,
			spreadRisk = "",
			spreadMessage = "",
			treatmentUrgency = "",
			capturedImagePath = capturedPath
		)

		val intent = Intent(this, ResultActivity::class.java).apply {
			putExtra(EXTRA_PREDICTION_RESULT, predictionResult)
			putExtra(EXTRA_CAPTURED_IMAGE_PATH, capturedPath)
		}
		startActivity(intent)
	}

	private fun handlePredictionFailure(message: String, file: File) {
		showLoading(false)
		showSnackbar(message) {
			runLocalInference(file, file.absolutePath)
		}
	}

	private fun toggleFlash() {
		currentFlashEnabled = !currentFlashEnabled
		applyTorchState()
	}

	private fun applyTorchState() {
		camera?.cameraControl?.enableTorch(currentFlashEnabled)
	}

	private fun showLoading(show: Boolean, message: String = "") {
		binding.loadingOverlay.isVisible = show
		binding.analyzingProgress.isIndeterminate = show
		if (message.isNotBlank()) {
			binding.analyzingText.text = message
		}
	}

	private fun showPermissionMessage() {
		Snackbar.make(binding.root, "Camera permission is required to scan leaves.", Snackbar.LENGTH_LONG)
			.setAction("Grant") {
				requestCameraPermission.launch(android.Manifest.permission.CAMERA)
			}
			.show()
	}

	private fun showError(message: String) {
		showLoading(false)
		Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
	}

	private fun showSnackbar(message: String, retryAction: (() -> Unit)? = null) {
		val snackbar = Snackbar.make(binding.root, message, Snackbar.LENGTH_INDEFINITE)
		retryAction?.let { action -> snackbar.setAction("Retry") { action() } }
		snackbar.setActionTextColor(ContextCompat.getColor(this, R.color.green_secondary))
		snackbar.show()
	}

	/**
	 * Shown instead of the result screen when the model's top-1 score is below
	 * [TFLiteHelper.MIN_CONFIDENCE]. The disease name is deliberately not displayed: the whole point
	 * of the gate is that the prediction cannot be relied on, so the user is asked to retake the
	 * photo rather than being handed a diagnosis that was wrong roughly half the time in field tests.
	 */
	private fun showLowConfidenceRetake(confidence: Float) {
		val percent = (confidence * 100f).toInt()
		val snackbar = Snackbar.make(
			binding.root,
			"Could not read this leaf clearly (best match only $percent% confident). " +
				"Fill the frame with a single leaf in good light and retake.",
			Snackbar.LENGTH_LONG
		)
		snackbar.setAction("Retake") { capturePhoto() }
		snackbar.setActionTextColor(ContextCompat.getColor(this, R.color.green_secondary))
		snackbar.show()
	}

	private fun startFramePulseAnimation() {
		pulseAnimator?.cancel()
		pulseAnimator = ValueAnimator.ofFloat(0.5f, 1.0f).apply {
			duration = 900L
			repeatMode = ValueAnimator.REVERSE
			repeatCount = ValueAnimator.INFINITE
			interpolator = LinearInterpolator()
			addUpdateListener { animator ->
				val alpha = animator.animatedValue as Float
				binding.pulseHint.alpha = alpha
				binding.frameTopLeftHorizontal.alpha = alpha
				binding.frameTopLeftVertical.alpha = alpha
				binding.frameTopRightHorizontal.alpha = alpha
				binding.frameTopRightVertical.alpha = alpha
				binding.frameBottomLeftHorizontal.alpha = alpha
				binding.frameBottomLeftVertical.alpha = alpha
				binding.frameBottomRightHorizontal.alpha = alpha
				binding.frameBottomRightVertical.alpha = alpha
			}
			start()
		}
	}

	override fun onDestroy() {
		super.onDestroy()
		pulseAnimator?.cancel()
		cameraProvider?.unbindAll()
		cameraExecutor.shutdown()
	}

	companion object {
		const val EXTRA_PREDICTION_RESULT = "extra_prediction_result"
		const val EXTRA_CAPTURED_IMAGE_PATH = "extra_captured_image_path"
	}
}

