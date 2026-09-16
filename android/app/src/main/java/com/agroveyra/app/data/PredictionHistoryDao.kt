package com.agroveyra.app.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface PredictionHistoryDao {

    @Insert
    suspend fun insert(entry: PredictionHistoryEntity)

    @Query("SELECT * FROM prediction_history ORDER BY createdAt DESC")
    suspend fun getAllHistory(): List<PredictionHistoryEntity>
}

