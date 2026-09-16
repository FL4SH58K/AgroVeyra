package com.agroveyra.app

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.recyclerview.widget.LinearLayoutManager
import com.agroveyra.app.databinding.ActivityHistoryBinding
import com.agroveyra.app.history.HistoryViewModel
import com.agroveyra.app.history.ScanHistoryAdapter
import com.agroveyra.app.history.ScanHistoryEntity
import com.agroveyra.app.models.PredictionResult
import com.google.android.material.dialog.MaterialAlertDialogBuilder

class HistoryActivity : AppCompatActivity(), ScanHistoryAdapter.Listener {

	private lateinit var binding: ActivityHistoryBinding
	private lateinit var adapter: ScanHistoryAdapter
	private val viewModel: HistoryViewModel by viewModels()

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		binding = ActivityHistoryBinding.inflate(layoutInflater)
		setContentView(binding.root)

		adapter = ScanHistoryAdapter(this)

		setupToolbar()
		setupRecyclerView()
		observeHistory()
		setupEmptyStateAction()
	}

	private fun setupToolbar() {
		binding.historyToolbar.setNavigationOnClickListener { finish() }
	}

	private fun setupRecyclerView() {
		binding.historyRecyclerView.layoutManager = LinearLayoutManager(this)
		binding.historyRecyclerView.adapter = adapter
	}

	private fun observeHistory() {
		viewModel.allScans.observe(this) { scans ->
			val items = scans.orEmpty()
			adapter.submitList(items)
			binding.emptyStateContainer.isVisible = items.isEmpty()
			binding.historyRecyclerView.isVisible = items.isNotEmpty()
		}
	}

	private fun setupEmptyStateAction() {
		binding.startScanningButton.setOnClickListener {
			startActivity(Intent(this, ScanActivity::class.java))
		}
	}

	override fun onScanClicked(item: ScanHistoryEntity) {
		startActivity(
			Intent(this, ResultActivity::class.java).apply {
				putExtra(ResultActivity.EXTRA_PREDICTION_RESULT, item.toPredictionResult())
			}
		)
	}

	override fun onScanLongClicked(item: ScanHistoryEntity) {
		MaterialAlertDialogBuilder(this)
			.setTitle("Delete scan?")
			.setMessage("This scan will be removed from your history.")
			.setPositiveButton("Delete") { _, _ ->
				viewModel.delete(item)
			}
			.setNegativeButton(android.R.string.cancel, null)
			.show()
	}

	private fun ScanHistoryEntity.toPredictionResult(): PredictionResult {
		val stageNumber = when {
			stage.contains("severe", ignoreCase = true) -> 3
			stage.contains("moderate", ignoreCase = true) -> 2
			else -> 1
		}

		val stagePercentage = when (stageNumber) {
			3 -> 75f
			2 -> 50f
			else -> 25f
		}

		val urgency = when (stageNumber) {
			3 -> "Treat within 1 day"
			2 -> "Treat within 3 days"
			else -> "Treat within 7 days"
		}

		return PredictionResult(
			disease = diseaseName,
			confidence = confidence,
			displayName = displayName,
			crop = crop,
			stage = stage,
			stageNumber = stageNumber,
			stagePercentage = stagePercentage,
			chemicalTreatment = chemicalTreatment,
			organicTreatment = organicTreatment,
			prevention = "Continue monitoring and follow standard preventive care.",
			urgency = urgency,
			isHealthy = isHealthy,
			spreadRisk = spreadRisk,
			spreadMessage = if (spreadRisk.isBlank()) "" else "Spread risk: $spreadRisk",
			treatmentUrgency = urgency,
			capturedImagePath = imagePath,
		)
	}
}

