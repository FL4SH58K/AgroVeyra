package com.agroveyra.app.utils

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

private data class TreatmentRecord(
    val display_name: String = "",
    val crop: String = "",
    val stage_advice: Map<String, String> = emptyMap(),
    val chemical_treatment: String = "",
    val organic_treatment: String = "",
    val prevention: String = "",
    val urgency: String = "medium"
)

object TreatmentDatabase {
    private var records: Map<String, TreatmentRecord>? = null

    fun getInfo(context: Context, className: String): DiseaseInfo {
        val record = load(context)[className]
        if (record == null) {
            return DiseaseMetadata.getInfo(className)
        }

        val healthy = className.contains("healthy", ignoreCase = true)
        val stageGuidance = record.stage_advice["early"].orEmpty()
        val prevention = listOfNotNull(
            stageGuidance.takeIf { it.isNotBlank() }?.let { "Stage-specific guidance: $it" },
            record.prevention.takeIf { it.isNotBlank() }
        ).joinToString("\n")

        return DiseaseInfo(
            disease = record.display_name.ifBlank { className },
            displayName = record.display_name.ifBlank { className.replace("___", " ").replace("_", " ") },
            crop = record.crop.ifBlank { className.substringBefore("___", "Unknown") },
            chemicalTreatment = record.chemical_treatment,
            organicTreatment = record.organic_treatment,
            prevention = prevention,
            isHealthy = healthy
        )
    }

    private fun load(context: Context): Map<String, TreatmentRecord> {
        records?.let { return it }
        val type = object : TypeToken<Map<String, TreatmentRecord>>() {}.type
        val loaded = context.assets.open("treatment_db.json").use { input ->
            input.bufferedReader().use { reader ->
                Gson().fromJson<Map<String, TreatmentRecord>>(reader, type)
            }
        }
        return loaded.also { records = it }
    }
}
