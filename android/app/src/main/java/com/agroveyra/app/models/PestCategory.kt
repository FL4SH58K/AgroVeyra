package com.agroveyra.app.models

/**
 * A single pest-damage category from pest_category_db.json.
 * Field names map directly to the JSON keys (Gson default).
 */
data class PestCategory(
    val display_name: String = "",
    val symptoms: String = "",
    val common_pests: List<String> = emptyList(),
    val chemical_treatment: String = "",
    val organic_treatment: String = "",
    val prevention: String = "",
    val urgency: String = "medium"
)
