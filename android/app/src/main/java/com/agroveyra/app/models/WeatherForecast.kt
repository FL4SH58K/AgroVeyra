package com.agroveyra.app.models

import com.google.gson.annotations.SerializedName

data class DayForecast(
    @SerializedName("day")
    val day: String,
    @SerializedName("temp")
    val temp: Int,
    @SerializedName("humidity")
    val humidity: Int,
    @SerializedName("condition")
    val condition: String,
    @SerializedName("icon")
    val icon: String
)

data class WeatherResponse(
    @SerializedName("spreadRisk")
    val spreadRisk: String,
    @SerializedName("spreadMessage")
    val spreadMessage: String,
    @SerializedName("forecast")
    val forecast: List<DayForecast>,
    @SerializedName("treatmentUrgency")
    val treatmentUrgency: String
)

