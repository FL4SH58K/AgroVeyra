package com.agroveyra.app

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.agroveyra.app.databinding.ActivityMainBinding
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

	private lateinit var binding: ActivityMainBinding
	private val preferences: SharedPreferences by lazy {
		getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
	}

	private var pendingPermissionAction: (() -> Unit)? = null

	private val permissionLauncher =
		registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
			val allGranted = result.values.all { it }
			if (allGranted) {
				pendingPermissionAction?.invoke()
				pendingPermissionAction = null
				return@registerForActivityResult
			}

			val deniedPermissions = result.filterValues { granted -> !granted }.keys
			if (deniedPermissions.any { permission ->
					shouldShowRequestPermissionRationale(permission)
				}
			) {
				showPermissionRationaleDialog()
			} else {
				showPermissionSettingsDialog()
			}
		}

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		binding = ActivityMainBinding.inflate(layoutInflater)
		setContentView(binding.root)

		setupClickListeners()
		setupBottomNavigation()
		setBottomNavigationState(NavDestination.HOME)

		lifecycleScope.launch {
			val isFirstLaunch = withContext(Dispatchers.IO) {
				preferences.getBoolean(KEY_FIRST_LAUNCH, true)
			}

			if (isFirstLaunch) {
				withContext(Dispatchers.IO) {
					preferences.edit().putBoolean(KEY_FIRST_LAUNCH, false).apply()
				}
				showOnboardingBottomSheet()
			}
		}
	}

	private fun setupClickListeners() {
		binding.scanButtonCard.setOnClickListener { openScanFlow() }
		binding.recentScansCard.setOnClickListener { openHistoryActivity() }
		binding.diseaseGuideCard.setOnClickListener { showInfoBottomSheet() }
	}

	private fun setupBottomNavigation() {
		binding.homeNavItem.setOnClickListener {
			setBottomNavigationState(NavDestination.HOME)
		}

		binding.scanNavItem.setOnClickListener {
			setBottomNavigationState(NavDestination.SCAN)
			openScanFlow()
		}

		binding.historyNavItem.setOnClickListener {
			setBottomNavigationState(NavDestination.HISTORY)
			openHistoryActivity()
		}

		binding.infoNavItem.setOnClickListener {
			setBottomNavigationState(NavDestination.INFO)
			showInfoBottomSheet()
		}
	}

	private fun openScanFlow() {
		if (hasAllRequiredPermissions()) {
			openScanActivity()
			return
		}

		pendingPermissionAction = ::openScanActivity

		if (shouldShowAnyPermissionRationale()) {
			showPermissionRationaleDialog()
		} else {
			requestRequiredPermissions()
		}
	}

	private fun openScanActivity() {
		startActivity(Intent(this, ScanActivity::class.java))
	}

	private fun openHistoryActivity() {
		startActivity(Intent(this, HistoryActivity::class.java))
	}

	private fun hasAllRequiredPermissions(): Boolean {
		return getRequiredPermissions().all { permission ->
			ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
		}
	}

	private fun shouldShowAnyPermissionRationale(): Boolean {
		return getRequiredPermissions().any { permission ->
			ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED &&
				shouldShowRequestPermissionRationale(permission)
		}
	}

	private fun requestRequiredPermissions() {
		permissionLauncher.launch(getRequiredPermissions().toTypedArray())
	}

	private fun getRequiredPermissions(): List<String> {
		val storagePermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
			Manifest.permission.READ_MEDIA_IMAGES
		} else {
			Manifest.permission.READ_EXTERNAL_STORAGE
		}

		return listOf(Manifest.permission.CAMERA, storagePermission)
	}

	private fun showPermissionRationaleDialog() {
		MaterialAlertDialogBuilder(this)
			.setTitle("Permissions needed")
				.setMessage("AgroVeyra needs camera and photo access to scan leaves and let you choose images from your gallery.")
			.setCancelable(false)
			.setPositiveButton("Grant permissions") { _, _ -> requestRequiredPermissions() }
			.setNegativeButton("Not now") { dialog, _ ->
				dialog.dismiss()
				pendingPermissionAction = null
			}
			.show()
	}

	private fun showPermissionSettingsDialog() {
		MaterialAlertDialogBuilder(this)
			.setTitle("Permissions blocked")
			.setMessage("You have permanently denied a required permission. Open app settings to enable camera and gallery access.")
			.setCancelable(false)
			.setPositiveButton("Open settings") { _, _ -> openAppSettings() }
			.setNegativeButton("Cancel") { dialog, _ ->
				dialog.dismiss()
				pendingPermissionAction = null
			}
			.show()
	}

	private fun openAppSettings() {
		val intent = Intent(
			Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
			Uri.fromParts("package", packageName, null)
		)
		startActivity(intent)
	}

	private fun showOnboardingBottomSheet() {
		val dialog = BottomSheetDialog(this)
		val container = LinearLayout(this).apply {
			orientation = LinearLayout.VERTICAL
			setPadding(dp(20), dp(20), dp(20), dp(24))
		}

		val title = TextView(this).apply {
			text = getString(R.string.app_name)
			textSize = 24f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_primary))
			setTypeface(typeface, android.graphics.Typeface.BOLD)
		}

		val subtitle = TextView(this).apply {
			text = "Quick onboarding"
			textSize = 14f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_secondary))
			setPadding(0, dp(4), 0, dp(16))
		}

		container.addView(title)
		container.addView(subtitle)
		container.addView(createOnboardingStepCard("1", "Scan leaves", "Take a photo of a leaf or upload one from your gallery to get an instant diagnosis."))
		container.addView(createOnboardingStepCard("2", "Get treatment guidance", "See disease severity, weather-based spread risk, and practical treatment advice."))

		val startButton = MaterialButton(this).apply {
			text = "Get started"
			setTextColor(ContextCompat.getColor(context, R.color.green_on_primary))
			setBackgroundColor(ContextCompat.getColor(context, R.color.green_primary))
			cornerRadius = dp(16)
			setOnClickListener {
				dialog.dismiss()
				if (!hasAllRequiredPermissions()) {
					openScanFlow()
				}
			}
		}

		val spacer = View(this).apply {
			layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(12))
		}

		container.addView(spacer)
		container.addView(startButton)
		dialog.setContentView(container)
		dialog.setCancelable(false)
		dialog.show()
	}

	private fun createOnboardingStepCard(step: String, titleText: String, bodyText: String): MaterialCardView {
		val card = MaterialCardView(this).apply {
			radius = dp(12).toFloat()
			cardElevation = dp(4).toFloat()
			setCardBackgroundColor(ContextCompat.getColor(context, R.color.green_surface))
			setContentPadding(dp(16), dp(16), dp(16), dp(16))
			val params = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
			params.bottomMargin = dp(12)
			layoutParams = params
		}

		val row = LinearLayout(this).apply {
			orientation = LinearLayout.HORIZONTAL
			gravity = Gravity.CENTER_VERTICAL
		}

		val stepBadge = TextView(this).apply {
			text = step
			gravity = Gravity.CENTER
			setTextColor(ContextCompat.getColor(context, R.color.green_on_primary))
			setBackgroundColor(ContextCompat.getColor(context, R.color.green_primary))
			setPadding(dp(12), dp(8), dp(12), dp(8))
		}

		val textColumn = LinearLayout(this).apply {
			orientation = LinearLayout.VERTICAL
			setPadding(dp(12), 0, 0, 0)
		}

		val stepTitle = TextView(this).apply {
			text = titleText
			textSize = 16f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_primary))
			setTypeface(typeface, android.graphics.Typeface.BOLD)
		}

		val stepBody = TextView(this).apply {
			text = bodyText
			textSize = 13f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_secondary))
		}

		textColumn.addView(stepTitle)
		textColumn.addView(stepBody)
		row.addView(stepBadge)
		row.addView(textColumn)
		card.addView(row)
		return card
	}

	private fun showInfoBottomSheet() {
		val dialog = BottomSheetDialog(this)
		val content = LinearLayout(this).apply {
			orientation = LinearLayout.VERTICAL
			setPadding(dp(20), dp(20), dp(20), dp(24))
		}

		val title = TextView(this).apply {
			text = "About AgroVeyra"
			textSize = 22f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_primary))
			setTypeface(typeface, android.graphics.Typeface.BOLD)
		}

		val body = TextView(this).apply {
			text = "AgroVeyra uses AI to detect plant diseases, estimate severity, and guide farmers with practical treatment and prevention advice."
			textSize = 14f
			setTextColor(ContextCompat.getColor(context, R.color.green_text_secondary))
			setPadding(0, dp(10), 0, dp(16))
		}

		val closeButton = MaterialButton(this).apply {
			text = "Close"
			setTextColor(ContextCompat.getColor(context, R.color.green_on_primary))
			setBackgroundColor(ContextCompat.getColor(context, R.color.green_primary))
			cornerRadius = dp(16)
			setOnClickListener { dialog.dismiss() }
		}

		content.addView(title)
		content.addView(body)
		content.addView(closeButton)
		dialog.setContentView(content)
		dialog.show()
	}

	private fun setBottomNavigationState(active: NavDestination) {
		updateNavItem(binding.homeNavItem, active == NavDestination.HOME)
		updateNavItem(binding.scanNavItem, active == NavDestination.SCAN)
		updateNavItem(binding.historyNavItem, active == NavDestination.HISTORY)
		updateNavItem(binding.infoNavItem, active == NavDestination.INFO)
	}

	private fun updateNavItem(container: View, isActive: Boolean) {
		val navRow = container as? LinearLayout ?: return
		val icon = navRow.getChildAt(0) as? ImageView
		val label = navRow.getChildAt(1) as? TextView

		val activeColor = ContextCompat.getColor(this, R.color.green_primary)
		val inactiveColor = ContextCompat.getColor(this, R.color.green_text_secondary)

		icon?.imageTintList = ContextCompat.getColorStateList(this, if (isActive) R.color.green_primary else R.color.green_text_secondary)
		label?.setTextColor(if (isActive) activeColor else inactiveColor)
		label?.setTypeface(label.typeface, if (isActive) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
	}

	private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

	private enum class NavDestination {
		HOME,
		SCAN,
		HISTORY,
		INFO,
	}

	companion object {
		private const val PREFS_NAME = "agroveyra_prefs"
		private const val KEY_FIRST_LAUNCH = "first_launch"
	}
}

