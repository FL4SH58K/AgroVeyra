package com.agroveyra.app.history

import android.content.Context
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.agroveyra.app.R
import com.agroveyra.app.databinding.ItemScanHistoryBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ScanHistoryAdapter(
    private val listener: Listener,
) : ListAdapter<ScanHistoryEntity, ScanHistoryAdapter.ScanHistoryViewHolder>(DiffCallback) {

    interface Listener {
        fun onScanClicked(item: ScanHistoryEntity)
        fun onScanLongClicked(item: ScanHistoryEntity)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ScanHistoryViewHolder {
        val binding = ItemScanHistoryBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false,
        )
        return ScanHistoryViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ScanHistoryViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ScanHistoryViewHolder(
        private val binding: ItemScanHistoryBinding,
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(item: ScanHistoryEntity) {
            val context = binding.root.context
            binding.historyDiseaseName.text = item.displayName.ifBlank { item.diseaseName }
            binding.historyCropAndDate.text = buildCropAndDate(context, item)
            binding.historyConfidenceText.text = context.getString(R.string.confidence_percent, item.confidence.toInt())
            binding.historySeverityText.text = if (item.isHealthy) {
                context.getString(R.string.healthy_label)
            } else {
                when (severityLevel(item)) {
                    1 -> "Low"
                    2 -> "Medium"
                    else -> "High"
                }
            }

            applySeverityColors(context, item)

            Glide.with(binding.root)
                .load(File(item.imagePath))
                .centerCrop()
                .placeholder(R.drawable.result_hero_gradient)
                .error(R.drawable.result_hero_gradient)
                .into(binding.historyThumbnail)

            binding.root.setOnClickListener { listener.onScanClicked(item) }
            binding.root.setOnLongClickListener {
                listener.onScanLongClicked(item)
                true
            }
        }

        private fun buildCropAndDate(context: Context, item: ScanHistoryEntity): String {
            val formattedDate = SimpleDateFormat("dd MMM yyyy", Locale.getDefault()).format(Date(item.dateScanned))
            return context.getString(R.string.crop_label) + ": " + item.crop + " • " + formattedDate
        }

        private fun severityLevel(item: ScanHistoryEntity): Int {
            if (item.isHealthy) return 0
            return when {
                item.spreadRisk.contains("high", ignoreCase = true) -> 3
                item.spreadRisk.contains("medium", ignoreCase = true) -> 2
                else -> 1
            }
        }

        private fun applySeverityColors(context: Context, item: ScanHistoryEntity) {
            val level = severityLevel(item)
            val badgeColor = when {
                item.isHealthy -> ContextCompat.getColor(context, R.color.risk_low)
                level == 1 -> ContextCompat.getColor(context, R.color.risk_low)
                level == 2 -> ContextCompat.getColor(context, R.color.risk_medium)
                level >= 3 -> ContextCompat.getColor(context, R.color.risk_high)
                else -> ContextCompat.getColor(context, R.color.risk_low)
            }

            binding.historySeverityBadge.setCardBackgroundColor(badgeColor)
            binding.historyStatusDot.backgroundTintList = android.content.res.ColorStateList.valueOf(badgeColor)
        }
    }

    private object DiffCallback : DiffUtil.ItemCallback<ScanHistoryEntity>() {
        override fun areItemsTheSame(oldItem: ScanHistoryEntity, newItem: ScanHistoryEntity): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ScanHistoryEntity, newItem: ScanHistoryEntity): Boolean {
            return oldItem == newItem
        }
    }
}
