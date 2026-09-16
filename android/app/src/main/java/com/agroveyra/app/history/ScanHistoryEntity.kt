package com.agroveyra.app.history

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "scan_history")
data class ScanHistoryEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val diseaseName: String,
    val displayName: String,
    val crop: String,
    val confidence: Float,
    val stage: String,
    val isHealthy: Boolean,
    val imagePath: String,
    val dateScanned: Long,
    val chemicalTreatment: String,
    val organicTreatment: String,
    val spreadRisk: String,
)
