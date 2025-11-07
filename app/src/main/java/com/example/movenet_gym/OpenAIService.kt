package com.example.movenet_gym

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import com.example.movenet_gym.BuildConfig

object OpenAIService {
    private val API_KEY = BuildConfig.OPENAI_API_KEY
    private const val BASE_URL = "https://api.openai.com/v1/chat/completions"
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .writeTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()


    suspend fun generateRoutineAndDiet(profileJson: String): String = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject().apply {
                put("model", "gpt-4o-mini")
                put("messages", JSONArray().apply {
                    put(JSONObject().apply {
                        put("role", "system")
                        put(
                            "content",
                            "너는 피트니스 트레이너이자 영양사야. 사용자의 신체 데이터를 JSON으로 받으면 운동 루틴과 식단을 추천해줘. 출력은 JSON만."
                        )
                    })
                    put(JSONObject().apply {
                        put("role", "user")
                        put("content", "사용자 정보: $profileJson")
                    })
                })
            }

            val body =
                json.toString().toRequestBody("application/json".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(BASE_URL)
                .addHeader("Authorization", "Bearer $API_KEY")
                .post(body)
                .build()

            // 🔹 동기 요청 (execute)
            val response = client.newCall(request).execute()

            if (!response.isSuccessful) {
                return@withContext "{\"error\": \"HTTP ${response.code}\"}"
            }

            val data = response.body?.string() ?: return@withContext "{\"error\": \"empty body\"}"

            val result = JSONObject(data)
                .optJSONArray("choices")
                ?.optJSONObject(0)
                ?.optJSONObject("message")
                ?.optString("content")
                ?: "응답 없음"

            result
        } catch (e: Exception) {
            "{\"error\":\"${e.message}\"}"
        }
    }
}
