package com.whisperlivekit.translate.ai

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.Headers
import retrofit2.http.POST
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.google.gson.annotations.SerializedName

data class Message(
    val role: String,
    val content: String
)

data class AgentRequest(
    val model: String = "claude-3-5-sonnet-20240620",
    val messages: List<Message>,
    @SerializedName("max_tokens") val maxTokens: Int = 1000,
    @SerializedName("temperature") val temperature: Double = 0.7
)

data class AgentResponse(
    val content: List<Content>
)

data class Content(
    val type: String,
    val text: String
)

interface ClaudeApi {
    @Headers(
        "x-api-key: YOUR_ANTHROPIC_API_KEY", // Replace with actual key from BuildConfig or secure storage
        "anthropic-version: 2023-06-01"
    )
    @POST("v1/messages")
    suspend fun sendMessage(@Body request: AgentRequest): AgentResponse
}

class ClaudeAgent {
    private val api: ClaudeApi

    init {
        val retrofit = Retrofit.Builder()
            .baseUrl("https://api.anthropic.com/")
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        api = retrofit.create(ClaudeApi::class.java)
    }

    suspend fun refineTranslation(transcription: String, targetLang: String): String = withContext(Dispatchers.IO) {
        val prompt = """
            You are a translation agent inspired by comprehensive agent patterns.
            Refine this Whisper transcription to natural $targetLang, handling ambiguities and context from an audio translation app.
            Transcription: $transcription
            Output only the refined translation.
        """.trimIndent()
        val request = AgentRequest(
            messages = listOf(Message("user", prompt))
        )
        try {
            val response = api.sendMessage(request)
            response.content.firstOrNull()?.text ?: transcription
        } catch (e: Exception) {
            transcription // Fallback on error
        }
    }
}