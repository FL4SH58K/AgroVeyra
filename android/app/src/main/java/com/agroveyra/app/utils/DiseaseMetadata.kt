package com.agroveyra.app.utils

data class DiseaseInfo(
    val disease: String,
    val displayName: String,
    val crop: String,
    val chemicalTreatment: String,
    val organicTreatment: String,
    val prevention: String,
    val isHealthy: Boolean = false
)

object DiseaseMetadata {
    private val metadata = mapOf(
        "Apple___Apple_scab" to DiseaseInfo(
            "Apple Scab", "Apple Scab", "Apple",
            "Apply fungicides such as captan or mancozeb at bud break.",
            "Prune infected branches and use sulfur-based organic fungicides.",
            "Plant resistant varieties and maintain good orchard sanitation.",
            false
        ),
        "Apple___Black_rot" to DiseaseInfo(
            "Black Rot", "Apple Black Rot", "Apple",
            "Use fungicides containing thiophanate-methyl or captan.",
            "Remove and destroy mummified fruit and cankered limbs.",
            "Prune regularly and avoid fruit wounding.",
            false
        ),
        "Apple___Cedar_apple_rust" to DiseaseInfo(
            "Cedar Apple Rust", "Cedar Apple Rust", "Apple",
            "Apply myclobutanil or triadimefon during early spring.",
            "Remove nearby cedar/juniper trees if possible.",
            "Grow resistant cultivars like Liberty or Enterprise.",
            false
        ),
        "Apple___healthy" to DiseaseInfo(
            "Healthy", "Healthy", "Apple",
            "None needed.", "None needed.", "Maintain regular watering and fertilization.",
            true
        ),
        "Potato___Early_blight" to DiseaseInfo(
            "Early Blight", "Potato Early Blight", "Potato",
            "Use chlorothalonil or mancozeb based fungicides.",
            "Apply copper-based sprays and maintain proper spacing.",
            "Rotate crops and avoid overhead irrigation.",
            false
        ),
        "Potato___Late_blight" to DiseaseInfo(
            "Late Blight", "Potato Late Blight", "Potato",
            "Apply systemic fungicides like metalaxyl or propamocarb.",
            "Use compost tea and copper-based organic fungicides.",
            "Use certified disease-free seeds and remove volunteer plants.",
            false
        ),
        "Potato___healthy" to DiseaseInfo(
            "Healthy", "Healthy", "Potato",
            "None needed.", "None needed.", "Monitor for early signs of pests.",
            true
        ),
        "Tomato___Early_blight" to DiseaseInfo(
            "Early Blight", "Tomato Early Blight", "Tomato",
            "Apply fungicides like mancozeb or chlorothalonil every 7-10 days.",
            "Mulch around plants and prune lower leaves to prevent soil splash.",
            "Rotate crops every 3 years and provide good air circulation.",
            false
        ),
        "Tomato___Late_blight" to DiseaseInfo(
            "Late Blight", "Tomato Late Blight", "Tomato",
            "Use fungicides containing chlorothalonil, copper, or maneb.",
            "Apply organic copper sprays and remove infected plants immediately.",
            "Plant resistant varieties and avoid watering from above.",
            false
        ),
        "Tomato___healthy" to DiseaseInfo(
            "Healthy", "Healthy", "Tomato",
            "None needed.", "None needed.", "Maintain consistent soil moisture.",
            true
        )
        // Add more mappings as needed based on class_names.json
    )

    fun getInfo(className: String): DiseaseInfo {
        return metadata[className] ?: DiseaseInfo(
            className.replace("___", " ").replace("_", " "),
            className.split("___").lastOrNull()?.replace("_", " ") ?: className,
            className.split("___").firstOrNull() ?: "Unknown",
            "Consult a local agricultural expert for chemical treatments.",
            "Use general organic pest control methods.",
            "Maintain overall plant health and sanitation.",
            className.contains("healthy", ignoreCase = true)
        )
    }
}

