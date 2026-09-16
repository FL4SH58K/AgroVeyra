package com.agroveyra.app.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "prediction_history")
data class PredictionHistoryEntity(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val disease: String,
    val confidence: Float,
    val displayName: String,
    val crop: String,
    val stage: String,
    val stageNumber: Int,
    val stagePercentage: Float,
    val chemicalTreatment: String,
    val organicTreatment: String,
    val prevention: String,
    val urgency: String,
    val isHealthy: Boolean,
    val spreadRisk: String,
    val spreadMessage: String,
    val treatmentUrgency: String,
    val capturedImagePath: String,
    val createdAt: Long = System.currentTimeMillis()
)

