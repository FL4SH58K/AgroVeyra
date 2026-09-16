package com.agroveyra.app.network

import com.google.gson.annotations.SerializedName
import com.agroveyra.app.BuildConfig
import com.agroveyra.app.models.DayForecast
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.HttpException
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.GET
import retrofit2.http.Query
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Retrofit API definition for AgroVeyra backend endpoints.
 */
interface ApiService {

	@GET("data/2.5/forecast")
	suspend fun getWeatherForecast(
		@Query("lat") lat: Double,
		@Query("lon") lon: Double,
		@Query("appid") apiKey: String,
		@Query("units") units: String = "metric"
	): OpenWeatherResponse
}

/**
 * Network singleton that owns the Retrofit instance and safe API helpers.
 */
object NetworkClient {

	const val BASE_URL: String = "https://api.openweathermap.org/"
	private const val TIMEOUT_SECONDS: Long = 30

	private val okHttpClient: OkHttpClient by lazy {
		OkHttpClient.Builder()
			.connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
			.readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
			.writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
			.apply {
				if (BuildConfig.DEBUG) {
					val loggingInterceptor = HttpLoggingInterceptor().apply {
						level = HttpLoggingInterceptor.Level.BODY
					}
					addInterceptor(loggingInterceptor)
				}
			}
			.build()
	}

	private val retrofit: Retrofit by lazy {
		Retrofit.Builder()
			.baseUrl(BASE_URL)
			.client(okHttpClient)
			.addConverterFactory(GsonConverterFactory.create())
			.build()
	}

	val apiService: ApiService by lazy {
		retrofit.create(ApiService::class.java)
	}

	suspend fun <T> safeApiCall(apiCall: suspend () -> T): Result<T> {
		return try {
			Result.Success(apiCall())
		} catch (throwable: Throwable) {
			handleApiError(throwable)
		}
	}

	fun <T> handleApiError(throwable: Throwable): Result<T> {
		val message = when (throwable) {
			is HttpException -> {
				val errorBody = throwable.response()?.errorBody()?.string()?.takeIf { it.isNotBlank() }
				errorBody ?: "Server error (${throwable.code()}). Please try again."
			}
			is IOException -> "Network error. Please check your internet connection."
			else -> throwable.localizedMessage ?: "Unexpected error occurred."
		}

		return Result.Error(message = message, throwable = throwable)
	}
}

/**
 * Coroutine-friendly result wrapper used for API calls.
 */
sealed class Result<out T> {
	data class Success<T>(val data: T) : Result<T>()
	data class Error(
		val message: String,
		val code: Int? = null,
		val throwable: Throwable? = null,
	) : Result<Nothing>()
}

/**
 * OpenWeatherMap Response Data Classes
 */
data class OpenWeatherResponse(
	@SerializedName("list") val list: List<ForecastItem>,
	@SerializedName("city") val city: City
)

data class ForecastItem(
	@SerializedName("dt") val dt: Long,
	@SerializedName("main") val main: MainData,
	@SerializedName("weather") val weather: List<WeatherDescription>,
	@SerializedName("dt_txt") val dtTxt: String
)

data class MainData(
	@SerializedName("temp") val temp: Float,
	@SerializedName("humidity") val humidity: Int
)

data class WeatherDescription(
	@SerializedName("main") val main: String,
	@SerializedName("description") val description: String
)

data class City(
	@SerializedName("name") val name: String
)


