package com.example.movenet_gym

import com.facebook.react.bridge.*
import kotlinx.coroutines.*

class OpenAIModule(reactContext: ReactApplicationContext)
    : ReactContextBaseJavaModule(reactContext) {

    override fun getName() = "OpenAIModule"

    @ReactMethod
    fun generateRoutineAndDiet(profileJson: String, promise: Promise) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val result = OpenAIService.generateRoutineAndDiet(profileJson)
                withContext(Dispatchers.Main) {
                    promise.resolve(result)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    promise.reject("OPENAI_ERROR", e.message)
                }
            }
        }
    }
}
