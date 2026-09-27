package com.agroveyra.app.utils

import android.content.Context
import com.agroveyra.app.models.PestCategory
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken

/**
 * Loads the pest-damage questionnaire categories (pest_category_db.json) shipped as an
 * Android asset, mirroring how TreatmentDatabase loads treatment_db.json.
 */
object PestCategoryDatabase {

    private const val PEST_CATEGORY_DB_FILE = "pest_category_db.json"

    /** Canonical questionnaire order (also the order the buttons are laid out). */
    private val categoryOrder = listOf(
        "chewing_holes",
        "stippling_yellowing",
        "webbing_curling",
        "mining_trails"
    )

    private var records: Map<String, PestCategory>? = null

    fun categoryOrder(): List<String> = categoryOrder

    fun getCategory(context: Context, key: String): PestCategory? = load(context)[key]

    fun all(context: Context): Map<String, PestCategory> = load(context)

    private fun load(context: Context): Map<String, PestCategory> {
        records?.let { return it }
        val type = object : TypeToken<Map<String, PestCategory>>() {}.type
        val loaded = context.assets.open(PEST_CATEGORY_DB_FILE).use { input ->
            input.bufferedReader().use { reader ->
                Gson().fromJson<Map<String, PestCategory>>(reader, type)
            }
        }
        return loaded.also { records = it }
    }
}
