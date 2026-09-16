package com.agroveyra.app.history

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query

@Dao
interface ScanHistoryDao {

    @Query("SELECT * FROM scan_history ORDER BY dateScanned DESC")
    fun getAllScans(): LiveData<List<ScanHistoryEntity>>

    @Query("SELECT * FROM scan_history WHERE id = :id LIMIT 1")
    fun getScanById(id: Long): LiveData<ScanHistoryEntity?>

    @Insert
    suspend fun insert(entity: ScanHistoryEntity): Long

    @Delete
    suspend fun delete(entity: ScanHistoryEntity)
}
