package com.agroveyra.app.models

import android.os.Parcelable
import com.google.gson.annotations.SerializedName
import kotlinx.parcelize.Parcelize

@Parcelize
data class PredictionResult(
    @SerializedName("disease")
    val disease: String,
    @SerializedName("confidence")
    val confidence: Float,
    @SerializedName("displayName")
    val displayName: String,
    @SerializedName("crop")
    val crop: String,
    @SerializedName("stage")
    val stage: String,
    @SerializedName("stageNumber")
    val stageNumber: Int,
    @SerializedName("stagePercentage")
    val stagePercentage: Float,
    @SerializedName("chemicalTreatment")
    val chemicalTreatment: String,
    @SerializedName("organicTreatment")
    val organicTreatment: String,
    @SerializedName("prevention")
    val prevention: String,
    @SerializedName("urgency")
    val urgency: String,
    @SerializedName("isHealthy")
    val isHealthy: Boolean,
    @SerializedName("spreadRisk")
    val spreadRisk: String,
    @SerializedName("spreadMessage")
    val spreadMessage: String,
    @SerializedName("treatmentUrgency")
    val treatmentUrgency: String,
    @SerializedName("capturedImagePath")
    val capturedImagePath: String
) : Parcelable

