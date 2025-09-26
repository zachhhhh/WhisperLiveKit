package com.whisperlivekit.translate.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ServerConfigMessage(
    val type: String,
    @SerialName("useAudioWorklet") val useAudioWorklet: Boolean? = null,
    @SerialName("source_language") val sourceLanguage: String? = null,
    @SerialName("target_language") val targetLanguage: String? = null,
)

@Serializable
data class ReadyToStopMessage(
    val type: String,
)

@Serializable
data class FrontLine(
    val speaker: Int = -1,
    val text: String = "",
    val start: String = "",
    val end: String = "",
    val translation: String? = null,
    @SerialName("detected_language") val detectedLanguage: String? = null,
)

@Serializable
data class FrontDataMessage(
    val status: String = "",
    val lines: List<FrontLine> = emptyList(),
    @SerialName("buffer_transcription") val bufferTranscription: String = "",
    @SerialName("buffer_diarization") val bufferDiarization: String = "",
    @SerialName("remaining_time_transcription") val remainingTimeTranscription: Double? = null,
    @SerialName("remaining_time_diarization") val remainingTimeDiarization: Double? = null,
    val error: String? = null,
)

sealed interface SessionEvent {
    data class Config(val message: ServerConfigMessage) : SessionEvent
    data class FrontUpdate(val payload: FrontDataMessage) : SessionEvent
    data object ReadyToStop : SessionEvent
    data class Closed(val reason: String?, val code: Int? = null) : SessionEvent
    data class Failure(val throwable: Throwable) : SessionEvent
}
