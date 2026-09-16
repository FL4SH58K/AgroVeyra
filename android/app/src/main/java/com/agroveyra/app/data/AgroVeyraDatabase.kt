package com.agroveyra.app.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [PredictionHistoryEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AgroVeyraDatabase : RoomDatabase() {

    abstract fun predictionHistoryDao(): PredictionHistoryDao

    companion object {
        @Volatile
        private var INSTANCE: AgroVeyraDatabase? = null

        fun getInstance(context: Context): AgroVeyraDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    AgroVeyraDatabase::class.java,
                    "agroveyra_database"
                ).build().also { INSTANCE = it }
            }
        }
    }
}

