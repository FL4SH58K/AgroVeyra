package com.agroveyra.app

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.location.Location
import android.location.LocationManager
import android.media.MediaPlayer
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import androidx.activity.viewModels
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.agroveyra.app.history.HistoryViewModel
import com.agroveyra.app.history.ScanHistoryEntity
import com.agroveyra.app.databinding.ActivityResultBinding
import com.agroveyra.app.models.PredictionResult
import com.agroveyra.app.network.NetworkClient
import com.agroveyra.app.network.OpenWeatherResponse
import com.agroveyra.app.network.ForecastItem
import com.agroveyra.app.network.Result as ApiResult
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ResultActivity : AppCompatActivity() {

	private lateinit var binding: ActivityResultBinding
	private val historyViewModel: HistoryViewModel by viewModels()
	private var currentResult: PredictionResult? = null
	private var tts: TextToSpeech? = null
	private var selectedLanguageCode: String = DEFAULT_LANGUAGE_CODE
	private var currentWeatherResult: OpenWeatherResponse? = null
	private var shimmerAnimator: android.animation.ValueAnimator? = null

	// Guard to ensure we only save once
	private var savedToHistory = false

	private val locationPermissionLauncher =
		registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
			val granted = permissions.values.any { it }
			if (granted) {
				fetchWeatherForCurrentLocation()
			} else {
				showLocationDisabledState()
				saveCurrentResultToHistory()
			}
		}

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		binding = ActivityResultBinding.inflate(layoutInflater)
		setContentView(binding.root)

		currentResult = readPredictionResult(intent)
		if (currentResult == null) {
			finishWithMessage("Missing prediction result.")
			return
		}

		setupUi()
		bindPredictionResult(currentResult!!)
		animateCardsStaggered()
		requestWeatherPermissionAndLoad()
	}

	private fun setupUi() {
		binding.scanAgainButton.setOnClickListener {
			finish()
		}

		binding.voiceActionButton.setOnClickListener {
			showLanguageSelectorDialog()
		}

		binding.treatmentTabs.addOnTabSelectedListener(object : com.google.android.material.tabs.TabLayout.OnTabSelectedListener {
			override fun onTabSelected(tab: com.google.android.material.tabs.TabLayout.Tab?) {
				val result = currentResult ?: return
				when (tab?.position) {
					0 -> renderTreatmentSteps(result.chemicalTreatment)
					1 -> renderTreatmentSteps(result.organicTreatment)
					2 -> renderTreatmentSteps(result.prevention)
				}
			}

			override fun onTabUnselected(tab: com.google.android.material.tabs.TabLayout.Tab?) = Unit
			override fun onTabReselected(tab: com.google.android.material.tabs.TabLayout.Tab?) = Unit
		})

		val languageAdapter = ArrayAdapter(
			this,
			android.R.layout.simple_spinner_item,
			resources.getStringArray(R.array.voice_language_entries).toList()
		).apply {
			setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
		}

		binding.languageSpinner.adapter = languageAdapter
		binding.languageSpinner.setSelection(0, false)
		binding.languageSpinner.setOnItemSelectedListener(object : android.widget.AdapterView.OnItemSelectedListener {
			override fun onItemSelected(
				parent: android.widget.AdapterView<*>,
				view: View?,
				position: Int,
				id: Long,
			) {
				selectedLanguageCode = resources.getStringArray(R.array.voice_language_values)
					.getOrNull(position)
					?: DEFAULT_LANGUAGE_CODE
			}

			override fun onNothingSelected(parent: android.widget.AdapterView<*>) = Unit
		})
	}

	private fun readPredictionResult(intent: Intent): PredictionResult? {
		return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
			intent.getParcelableExtra(EXTRA_PREDICTION_RESULT, PredictionResult::class.java)
		} else {
			@Suppress("DEPRECATION")
			intent.getParcelableExtra(EXTRA_PREDICTION_RESULT)
		}
	}

	private fun bindPredictionResult(result: PredictionResult) {
		binding.diseaseNameText.text = result.displayName.ifBlank { result.disease }
		binding.cropTypeText.text = result.crop
		binding.confidenceProgress.max = 100
		binding.confidenceProgress.progress = result.confidence.toInt().coerceIn(0, 100)
		binding.confidenceText.text = getString(R.string.confidence_percent, result.confidence.toInt())
		applyConfidenceTint(result.confidence)

		val activeStage = result.stageNumber.coerceIn(1, 3)
		binding.infectedAreaText.text = buildInfectedAreaText(result.stagePercentage)
		binding.urgencyBadge.text = result.treatmentUrgency.ifBlank { mapUrgencyText(result.stageNumber) }

		applyResultBadge(result)
		highlightStage(activeStage)
		
		renderTreatmentSteps(result.chemicalTreatment)
		binding.treatmentTabs.getTabAt(0)?.select()

		val imagePath = result.capturedImagePath
		if (imagePath.isNotBlank()) {
			Glide.with(this)
				.load(File(imagePath))
				.centerCrop()
				.into(binding.capturedLeafImage)
		}
	}

	private fun applyConfidenceTint(confidence: Float) {
		val tintColor = when {
			confidence >= 85f -> R.color.green_success
			confidence >= 60f -> R.color.green_warning
			else -> R.color.green_error
		}
		binding.confidenceProgress.progressTintList = ContextCompat.getColorStateList(this, tintColor)
	}

	private fun applyResultBadge(result: PredictionResult) {
		val badgeBackground: Int
		val textColor: Int
		val text: String

		if (result.disease.contains("healthy", ignoreCase = true)) {
			badgeBackground = R.drawable.badge_green
			textColor = R.color.green_primary_variant
			text = getString(R.string.healthy_label)
		} else {
			when {
				result.confidence >= 85f -> {
					badgeBackground = R.drawable.badge_green
					textColor = R.color.green_primary_variant
					text = "HIGH CONFIDENCE"
				}
				result.confidence >= 60f -> {
					badgeBackground = R.drawable.badge_orange
					textColor = Color.parseColor("#B44900")
					text = "MEDIUM CONFIDENCE"
				}
				else -> {
					badgeBackground = R.drawable.badge_yellow
					textColor = Color.parseColor("#8A5A00")
					text = "LOW CONFIDENCE"
				}
			}
		}

		binding.resultBadge.setBackgroundResource(badgeBackground)
		binding.resultBadge.setTextColor(
			if (textColor is Int) textColor else ContextCompat.getColor(this, textColor)
		)
		binding.resultBadge.text = text
	}

	private fun highlightStage(stageNumber: Int) {
		val inactiveColor = ContextCompat.getColor(this, R.color.gauge_grey)
		val activeColor = when (stageNumber.coerceIn(1, 3)) {
			1 -> ContextCompat.getColor(this, R.color.risk_low)
			2 -> ContextCompat.getColor(this, R.color.risk_medium)
			else -> ContextCompat.getColor(this, R.color.risk_high)
		}

		val indicators = listOf(binding.stageIndicator1, binding.stageIndicator2, binding.stageIndicator3)
		val labels = listOf(binding.stageLabel1, binding.stageLabel2, binding.stageLabel3)

		indicators.forEachIndexed { index, view ->
			val isActive = (index + 1) <= stageNumber
			val color = if (isActive) activeColor else inactiveColor
			view.backgroundTintList = android.content.res.ColorStateList.valueOf(color)
			
			if ((index + 1) == stageNumber) {
				labels[index].setTextColor(color)
				labels[index].setTypeface(null, android.graphics.Typeface.BOLD)
			} else {
				labels[index].setTextColor(ContextCompat.getColor(this, R.color.green_text_secondary))
				labels[index].setTypeface(null, android.graphics.Typeface.NORMAL)
			}
		}
	}

	private fun requestWeatherPermissionAndLoad() {
		val permissions = arrayOf(
			Manifest.permission.ACCESS_FINE_LOCATION,
			Manifest.permission.ACCESS_COARSE_LOCATION,
		)

		val hasLocationPermission = permissions.any { permission ->
			ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
		}

		if (hasLocationPermission) {
			fetchWeatherForCurrentLocation()
		} else {
			locationPermissionLauncher.launch(permissions)
		}
	}

	private fun fetchWeatherForCurrentLocation() {
		lifecycleScope.launch {
			showWeatherLoading(true)
			val location = withContext(Dispatchers.IO) { getBestLastKnownLocation() }

			if (location == null) {
				showWeatherLoading(false)
				showLocationUnavailable()
				saveCurrentResultToHistory()
				return@launch
			}

			val diseaseName = currentResult?.disease.orEmpty()
			val apiResult = withContext(Dispatchers.IO) {
				NetworkClient.safeApiCall {
					NetworkClient.apiService.getWeatherForecast(
						lat = location.latitude,
						lon = location.longitude,
						apiKey = BuildConfig.OPENWEATHER_API_KEY
					)
				}
			}

			when (apiResult) {
				is ApiResult.Success -> {
					currentWeatherResult = apiResult.data
					bindWeatherResult(apiResult.data, diseaseName)
					showWeatherLoading(false)
					saveCurrentResultToHistory()
				}

				is ApiResult.Error -> {
					showWeatherLoading(false)
					showWeatherError(apiResult.message)
					saveCurrentResultToHistory()
				}
			}
		}
	}

	private fun bindWeatherResult(response: OpenWeatherResponse, diseaseName: String) {
		val avgHumidity = response.list.take(8).map { it.main.humidity }.average()
		val currentAffectedArea = currentResult?.stagePercentage?.coerceIn(0f, 100f) ?: 0f
		val spreadRisk = when {
			avgHumidity > 80 -> "High"
			avgHumidity > 60 -> "Medium"
			else -> "Low"
		}
		val additionalAreaRange = when (spreadRisk) {
			"High" -> 20..35
			"Medium" -> 10..20
			else -> 0..10
		}
		val spreadMessage = if (currentResult?.isHealthy == true) {
			"No disease detected. Over the next 7 days, weather-based spread risk is low."
		} else {
			"Over the next 7 days, $diseaseName has a $spreadRisk spread risk. " +
				"If untreated, the affected area could increase by about " +
				"${additionalAreaRange.first}-${additionalAreaRange.last} percentage points " +
				"from the current ~${currentAffectedArea.toInt()}%. " +
				"This is a weather-based estimate; recheck the plant in 2-3 days."
		}

		binding.spreadRiskBadge.text = spreadRisk.uppercase(Locale.getDefault())
		binding.spreadMessageText.text = spreadMessage
		binding.gaugeValueText.text = spreadRisk.uppercase(Locale.getDefault())

		val (rotation, colorRes) = when (spreadRisk.lowercase(Locale.getDefault())) {
			"high" -> 60f to R.color.risk_high
			"medium" -> 0f to R.color.risk_medium
			else -> -60f to R.color.risk_low
		}

		binding.gaugeNeedle.animate().rotation(rotation).setDuration(800).start()
		binding.gaugeValueText.setTextColor(ContextCompat.getColor(this, colorRes))

		binding.spreadRiskBadge.setBackgroundResource(
			when (spreadRisk.lowercase(Locale.getDefault())) {
				"high" -> R.drawable.badge_red
				"medium" -> R.drawable.badge_orange
				else -> R.drawable.badge_green
			}
		)
		binding.spreadRiskBadge.setTextColor(
			when (spreadRisk.lowercase(Locale.getDefault())) {
				"high" -> ContextCompat.getColor(this, R.color.green_error)
				"medium" -> Color.parseColor("#B44900")
				else -> ContextCompat.getColor(this, R.color.green_primary_variant)
			}
		)

		renderForecast(response.list.filterIndexed { index, _ -> index % 8 == 0 })

		val treatmentUrgency = when (spreadRisk) {
			"High" -> "Treat within 24 hours"
			"Medium" -> "Treat within 3 days"
			else -> "Monitor closely"
		}

		currentResult = currentResult?.copy(
			spreadRisk = spreadRisk,
			spreadMessage = spreadMessage,
			treatmentUrgency = treatmentUrgency,
		)

		binding.urgencyBadge.text = treatmentUrgency
	}

	private fun renderForecast(forecast: List<ForecastItem>) {
		binding.forecastRowContainer.removeAllViews()

		forecast.forEach { dayForecast ->
			binding.forecastRowContainer.addView(createForecastChip(dayForecast))
		}
	}

	private fun createForecastChip(dayForecast: ForecastItem): View {
		val density = resources.displayMetrics.density

		val card = com.google.android.material.card.MaterialCardView(this).apply {
			layoutParams = LinearLayout.LayoutParams((72 * density).toInt(), (90 * density).toInt()).apply {
				marginEnd = (12 * density).toInt()
			}
			radius = 16 * density
			cardElevation = 0f
			strokeWidth = (1 * density).toInt()
			strokeColor = ContextCompat.getColor(this@ResultActivity, R.color.gauge_grey)
			setCardBackgroundColor(Color.WHITE)
		}

		val content = LinearLayout(this).apply {
			orientation = LinearLayout.VERTICAL
			gravity = android.view.Gravity.CENTER
			setPadding(0, (8 * density).toInt(), 0, (8 * density).toInt())
		}

		val dayLabel = android.widget.TextView(this).apply {
			text = dayLabelFromDate(dayForecast.dtTxt)
			setTextColor(ContextCompat.getColor(this@ResultActivity, R.color.green_text_secondary))
			textSize = 11f
		}

		val icon = android.widget.ImageView(this).apply {
			layoutParams = LinearLayout.LayoutParams((24 * density).toInt(), (24 * density).toInt()).apply {
				topMargin = (4 * density).toInt()
				bottomMargin = (4 * density).toInt()
			}
			setImageResource(android.R.drawable.ic_menu_compass) 
			imageTintList = ContextCompat.getColorStateList(this@ResultActivity, R.color.green_primary)
		}

		val tempLabel = android.widget.TextView(this).apply {
			text = "${dayForecast.main.temp.toInt()}°"
			setTextColor(ContextCompat.getColor(this@ResultActivity, R.color.green_text_primary))
			textSize = 14f
			setTypeface(typeface, android.graphics.Typeface.BOLD)
		}

		content.addView(dayLabel)
		content.addView(icon)
		content.addView(tempLabel)
		card.addView(content)
		return card
	}

	private fun dayLabelFromDate(dateString: String): String {
		return try {
			val parsedDate = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).parse(dateString)
			if (parsedDate != null) {
				SimpleDateFormat("EEE", Locale.getDefault()).format(parsedDate)
			} else {
				dateString.take(3)
			}
		} catch (_: Exception) {
			dateString.take(3)
		}
	}

	private fun showWeatherLoading(isLoading: Boolean) {
		if (isLoading) {
			binding.spreadMessageText.text = getString(R.string.weather_loading)
			binding.weatherShimmerOverlay.isVisible = true
			startShimmer()
		} else {
			stopShimmer()
			binding.weatherShimmerOverlay.isVisible = false
		}
	}

	private fun startShimmer() {
		shimmerAnimator?.cancel()
		binding.weatherShimmerOverlay.post {
			val width = binding.weatherCard.width.toFloat()
			shimmerAnimator = android.animation.ValueAnimator.ofFloat(-width, width).apply {
				duration = 1200
				repeatCount = android.animation.ValueAnimator.INFINITE
				interpolator = android.view.animation.LinearInterpolator()
				addUpdateListener { animator ->
					binding.weatherShimmerOverlay.translationX = animator.animatedValue as Float
				}
				start()
			}
		}
	}

	private fun stopShimmer() {
		shimmerAnimator?.cancel()
		shimmerAnimator = null
	}

	private fun getBestLastKnownLocation(): Location? {
		if (!hasLocationPermission()) return null

		val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
		val providers = locationManager.getProviders(true)

		providers.forEach { provider ->
			try {
				val lastKnown = locationManager.getLastKnownLocation(provider)
				if (lastKnown != null) return lastKnown
			} catch (_: SecurityException) {
				return null
			}
		}

		return null
	}

	private fun hasLocationPermission(): Boolean {
		return ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED ||
			ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
	}

	private fun showLocationUnavailable() {
		binding.spreadRiskBadge.text = "N/A"
		binding.spreadRiskBadge.setBackgroundResource(R.drawable.badge_yellow)
		binding.spreadRiskBadge.setTextColor(Color.parseColor("#8A5A00"))
		binding.spreadMessageText.text = getString(R.string.enable_location_for_spread_prediction)
	}

	private fun showWeatherError(message: String) {
		binding.spreadMessageText.text = message.ifBlank { getString(R.string.weather_api_error) }
		Snackbar.make(binding.root, message.ifBlank { getString(R.string.weather_api_error) }, Snackbar.LENGTH_LONG).show()
	}

	private fun showLocationDisabledState() {
		binding.spreadMessageText.text = getString(R.string.enable_location_for_spread_prediction)
		Snackbar.make(binding.root, getString(R.string.enable_location_for_spread_prediction), Snackbar.LENGTH_LONG).show()
	}

	private fun showLanguageSelectorDialog() {
		val languages = resources.getStringArray(R.array.voice_language_entries)
		val languageValues = resources.getStringArray(R.array.voice_language_values)
		var selectedIndex = binding.languageSpinner.selectedItemPosition.coerceAtLeast(0)

		MaterialAlertDialogBuilder(this)
			.setTitle(getString(R.string.voice_dialog_title))
			.setSingleChoiceItems(languages, selectedIndex) { _, which ->
				selectedIndex = which
			}
			.setPositiveButton(getString(R.string.listen_in_your_language)) { dialog, _ ->
				dialog.dismiss()
				selectedLanguageCode = languageValues.getOrNull(selectedIndex) ?: DEFAULT_LANGUAGE_CODE
				binding.languageSpinner.setSelection(selectedIndex, true)
				speakAdvisory(selectedLanguageCode)
			}
			.setNegativeButton(android.R.string.cancel, null)
			.show()
	}

	private fun speakAdvisory(languageCode: String) {
		val result = currentResult ?: return
		val speakText = buildSpeechText(result)

		binding.voiceActionButton.isEnabled = false
		binding.voiceActionButton.text = getString(R.string.speak_analyzing)

		if (tts == null) {
			tts = TextToSpeech(this) { status ->
				if (status == TextToSpeech.SUCCESS) {
					configureAndSpeak(speakText, languageCode)
				} else {
					showVoiceError("TTS initialization failed.")
				}
			}
		} else {
			configureAndSpeak(speakText, languageCode)
		}
	}

	private fun configureAndSpeak(text: String, languageCode: String) {
		val locale = Locale(languageCode)
		val result = tts?.setLanguage(locale) ?: TextToSpeech.LANG_NOT_SUPPORTED

		if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
			showVoiceError("Language $languageCode is not supported.")
		} else {
			tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "AgroVeyraTts")
			binding.voiceActionButton.isEnabled = true
			binding.voiceActionButton.text = getString(R.string.listen_in_your_language)
		}
	}

	private fun buildSpeechText(result: PredictionResult): String {
		val base = if (result.isHealthy) {
			"Your plant appears healthy. Continue preventive care."
		} else {
			"Detected ${result.displayName}. Stage ${result.stageNumber}. ${result.urgency}."
		}

		return buildString {
			append(base)
			append(" Chemical treatment: ")
			append(result.chemicalTreatment)
			append(" Organic option: ")
			append(result.organicTreatment)
			append(" Prevention: ")
			append(result.prevention)
			if (result.spreadRisk.isNotBlank()) {
				append(" Spread risk: ")
				append(result.spreadRisk)
				append(". ")
				append(result.spreadMessage)
			}
		}
	}

	private fun showVoiceError(message: String) {
		binding.voiceActionButton.isEnabled = true
		binding.voiceActionButton.text = getString(R.string.listen_in_your_language)
		Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG)
			.setAction("Retry") { speakAdvisory(selectedLanguageCode) }
			.show()
	}

	private fun releaseTts() {
		tts?.stop()
		tts?.shutdown()
		tts = null
	}

	private fun saveCurrentResultToHistory() {
		if (savedToHistory) {
			Log.d("ResultActivity", "saveCurrentResultToHistory: already saved, skipping")
			return
		}

		val result = currentResult ?: return
		historyViewModel.insert(result.toHistoryEntity())
		savedToHistory = true
		Log.d("ResultActivity", "saveCurrentResultToHistory: saved scan to history")
		Snackbar.make(binding.root, "Scan saved to history", Snackbar.LENGTH_SHORT).show()
	}

	private fun PredictionResult.toHistoryEntity(): ScanHistoryEntity {
		return ScanHistoryEntity(
			diseaseName = disease,
			displayName = displayName,
			crop = crop,
			confidence = confidence,
			stage = stage,
			isHealthy = isHealthy,
			imagePath = capturedImagePath,
			dateScanned = System.currentTimeMillis(),
			chemicalTreatment = chemicalTreatment,
			organicTreatment = organicTreatment,
			spreadRisk = spreadRisk,
		)
	}

	private fun renderTreatmentSteps(stepsText: String) {
		binding.treatmentStepsContainer.removeAllViews()
		if (stepsText.isBlank()) return

		val steps = stepsText.split(Regex("(?<=\\.)|(?<=\\n)")).filter { it.isNotBlank() }
		val density = resources.displayMetrics.density

		steps.forEachIndexed { index, stepText ->
			val row = LinearLayout(this).apply {
				orientation = LinearLayout.HORIZONTAL
				setPadding(0, (8 * density).toInt(), 0, (8 * density).toInt())
				gravity = android.view.Gravity.TOP
			}

			val badge = android.widget.TextView(this).apply {
				val size = (24 * density).toInt()
				layoutParams = LinearLayout.LayoutParams(size, size).apply {
					marginEnd = (12 * density).toInt()
					topMargin = (2 * density).toInt()
				}
				val stepNum = (index + 1).toString()
				setText(stepNum)
				setTextColor(Color.WHITE)
				textSize = 12f
				setGravity(android.view.Gravity.CENTER)
				setTypeface(null, android.graphics.Typeface.BOLD)
				setBackgroundResource(R.drawable.step_number_badge)
			}

			val stepBody = android.widget.TextView(this).apply {
				layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
				setText(stepText.trim().removePrefix("-").trim())
				setTextColor(ContextCompat.getColor(this@ResultActivity, R.color.green_text_secondary))
				textSize = 14f
			}

			row.addView(badge)
			row.addView(stepBody)
			binding.treatmentStepsContainer.addView(row)
		}
	}

	private fun animateCardsStaggered() {
		val cards = listOf(
			binding.detectionResultCard,
			binding.stageCard,
			binding.weatherCard,
			binding.treatmentCard,
			binding.voiceCard,
		)

		cards.forEachIndexed { index, card ->
			card.alpha = 0f
			card.translationY = 60f
			card.animate()
				.alpha(1f)
				.translationY(0f)
				.setStartDelay(index * 150L)
				.setDuration(450L)
				.setInterpolator(AccelerateDecelerateInterpolator())
				.start()
		}
	}

	private fun buildInfectedAreaText(stagePercentage: Float): String {
		val percent = stagePercentage.toInt().coerceIn(0, 100)
		return "Infected area: ~$percent% of leaf"
	}

	private fun mapUrgencyText(stageNumber: Int): String {
		return when (stageNumber.coerceIn(1, 3)) {
			1 -> "Treat within 7 days"
			2 -> "Treat within 3 days"
			else -> "Treat within 1 day"
		}
	}

	private fun finishWithMessage(message: String) {
		Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
		finish()
	}

	override fun onDestroy() {
		super.onDestroy()
		stopShimmer()
		releaseTts()
	}

	companion object {
		const val EXTRA_PREDICTION_RESULT = "extra_prediction_result"
		private const val DEFAULT_LANGUAGE_CODE = "en"
	}
}

