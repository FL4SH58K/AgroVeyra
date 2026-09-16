package com.agroveyra.app.history

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class HistoryViewModel(application: Application) : AndroidViewModel(application) {

    private val dao = AppDatabase.getInstance(application).scanHistoryDao()

    val allScans: LiveData<List<ScanHistoryEntity>> = dao.getAllScans()

    fun insert(entity: ScanHistoryEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.insert(entity)
        }
    }

    fun delete(entity: ScanHistoryEntity) {
        viewModelScope.launch(Dispatchers.IO) {
            dao.delete(entity)
        }
    }
}
