package com.agroveyra.app

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.agroveyra.app.databinding.ActivityPestResultBinding
import com.agroveyra.app.history.HistoryViewModel
import com.agroveyra.app.history.ScanHistoryEntity
import com.agroveyra.app.models.PestCategory
import com.agroveyra.app.utils.PestCategoryDatabase
import com.bumptech.glide.Glide
import java.io.File

/**
 * Shown when the triage classifier routes an image to "pest_damage".
 *
 * Displays the captured leaf and a short questionnaire asking which of the four damage
 * patterns the user sees. Picking a category (or skipping) reveals targeted IPM guidance
 * from pest_category_db.json (chemical + organic treatment + prevention).
 */
class PestResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPestResultBinding
    private val historyViewModel: HistoryViewModel by viewModels()

    private var imagePath: String = ""
    private var confidencePercent: Float = 0f
    private var savedToHistory = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPestResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        imagePath = intent.getStringExtra(EXTRA_CAPTURED_IMAGE_PATH).orEmpty()
        val confidence = intent.getFloatExtra(EXTRA_CONFIDENCE, 0f)
        confidencePercent = confidence * 100f

        loadImage(imagePath)
        bindConfidence(confidence)
        setupActions()
        setupQuestionnaire()
    }

    private fun loadImage(imagePath: String) {
        if (imagePath.isNotBlank()) {
            Glide.with(this)
                .load(File(imagePath))
                .centerCrop()
                .into(binding.capturedLeafImage)
        }
    }

    private fun bindConfidence(confidence: Float) {
        val percent = (confidence * 100f).toInt()
        binding.confidenceText.text = "Pest damage detected ($percent% confident)"
    }

    private fun setupActions() {
        binding.backButton.setOnClickListener { finish() }
        binding.scanAgainButton.setOnClickListener { finish() }
    }

    private fun setupQuestionnaire() {
        val buttons = listOf(
            binding.categoryChewingHoles,
            binding.categoryStipplingYellowing,
            binding.categoryWebbingCurling,
            binding.categoryMiningTrails
        )

        buttons.zip(PestCategoryDatabase.categoryOrder()).forEach { (button, key) ->
            button.setOnClickListener { onCategorySelected(key) }
        }

        binding.skipQuestionnaireButton.setOnClickListener { showGeneralGuidance() }
    }

    private fun onCategorySelected(key: String) {
        val category = PestCategoryDatabase.getCategory(this, key)
        if (category == null) {
            showGeneralGuidance()
            return
        }
        renderGuidance(category)
        saveToHistory(category)
        // Scroll the freshly-revealed guidance into view.
        binding.pestScrollView.post { binding.pestScrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun showGeneralGuidance() {
        binding.guidanceCard.isVisible = true
        binding.guidanceTitle.text = "General pest guidance"
        binding.guidanceSymptoms.text =
            "Follow integrated pest management: identify the pest, monitor regularly, and choose the least-toxic effective control first."
        binding.guidancePests.text = "Unknown"
        binding.chemicalTreatmentText.text =
            "Consult a local agricultural expert for chemical options specific to your crop and region."
        binding.organicTreatmentText.text =
            "Use general organic controls such as neem oil or insecticidal soap, and encourage beneficial insects."
        binding.preventionText.text =
            "Maintain overall plant health, sanitation, and regular scouting."
        saveToHistory(null)
    }

    private fun saveToHistory(category: PestCategory?) {
        if (savedToHistory) return
        savedToHistory = true

        val entity = ScanHistoryEntity(
            diseaseName = "pest_damage",
            displayName = category?.display_name ?: "Pest Damage",
            crop = "Pest Damage",
            confidence = confidencePercent,
            stage = "Pest Damage",
            isHealthy = false,
            imagePath = imagePath,
            dateScanned = System.currentTimeMillis(),
            chemicalTreatment = category?.chemical_treatment
                ?: "Consult a local agricultural expert for chemical options specific to your crop and region.",
            organicTreatment = category?.organic_treatment
                ?: "Use general organic controls such as neem oil or insecticidal soap, and encourage beneficial insects.",
            spreadRisk = category?.urgency ?: "medium"
        )
        historyViewModel.insert(entity)
    }

    private fun renderGuidance(category: PestCategory) {
        binding.guidanceCard.isVisible = true
        binding.guidanceTitle.text = category.display_name
        binding.guidanceSymptoms.text = category.symptoms
        binding.guidancePests.text = category.common_pests.joinToString(", ")
        binding.chemicalTreatmentText.text = category.chemical_treatment
        binding.organicTreatmentText.text = category.organic_treatment
        binding.preventionText.text = category.prevention
    }

    companion object {
        const val EXTRA_CAPTURED_IMAGE_PATH = "extra_captured_image_path"
        const val EXTRA_CONFIDENCE = "extra_confidence"
    }
}
