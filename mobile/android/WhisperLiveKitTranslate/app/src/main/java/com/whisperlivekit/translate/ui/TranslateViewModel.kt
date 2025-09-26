package com.whisperlivekit.translate.ui

import android.Manifest
import androidx.annotation.RequiresPermission
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.whisperlivekit.translate.audio.AudioStreamer
import com.whisperlivekit.translate.model.Language
import com.whisperlivekit.translate.model.SessionEvent
import com.whisperlivekit.translate.model.TranscriptionLine
import com.whisperlivekit.translate.model.TranslateUiState
import com.whisperlivekit.translate.network.WhisperSessionManager
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class TranslateViewModel : ViewModel() {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    private val audioStreamer = AudioStreamer()
    private val okHttpClient = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BASIC
        })
        .build()

    private val sessionManager = WhisperSessionManager(
        client = okHttpClient,
        json = json,
        audioStreamer = audioStreamer,
    )

    private val _uiState = MutableStateFlow(TranslateUiState())
    val uiState: StateFlow<TranslateUiState> = _uiState.asStateFlow()
    val waveformEnergy: StateFlow<List<Float>> = audioStreamer.bufferEnergy

    private var sessionJob: Job? = null
    private val toggleGuard = AtomicBoolean(false)
    private var lastToggleAt: Long = 0

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun toggleStreaming() {
        val now = System.currentTimeMillis()
        if (toggleGuard.get() || now - lastToggleAt < TOGGLE_DEBOUNCE_MS) {
            return
        }
        lastToggleAt = now
        if (!toggleGuard.compareAndSet(false, true)) {
            return
        }

        if (_uiState.value.isStreaming) {
            stopStreaming()
        } else {
            startStreaming()
        }
    }

    fun updateServerAddress(address: String) {
        _uiState.update { it.copy(serverAddress = address.trim()) }
    }

    fun selectSource(language: Language) {
        _uiState.update { state ->
            state.copy(sourceLanguage = language)
        }
    }

    fun selectTarget(language: Language) {
        _uiState.update { state ->
            state.copy(targetLanguage = language)
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    fun swapLanguages() {
        _uiState.update { state ->
            val newSource = if (state.targetLanguage.isAutoDetect) state.sourceLanguage else state.targetLanguage
            val newTarget = if (state.sourceLanguage.isAutoDetect) state.targetLanguage else state.sourceLanguage
            state.copy(
                sourceLanguage = newSource,
                targetLanguage = newTarget,
            )
        }
    }

    fun onPermissionDenied() {
        _uiState.update { it.copy(isPermissionMissing = true) }
    }

    fun acknowledgePermissionMessage() {
        _uiState.update { it.copy(isPermissionMissing = false) }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startStreaming() {
        try {
            if (sessionJob != null) return

            val config = _uiState.value
            sessionJob = viewModelScope.launch {
                _uiState.update {
                    it.copy(
                        isStreaming = true,
                        isConnected = false,
                        errorMessage = null,
                        statusMessage = "Connecting...",
                    )
                }

                sessionManager.openSession(
                    WhisperSessionManager.SessionConfig(
                        serverAddress = config.serverAddress,
                        sourceLanguage = config.sourceLanguage,
                        targetLanguage = config.targetLanguage,
                    ),
                ).collect { event ->
                    when (event) {
                    is SessionEvent.Config -> {
                        _uiState.update {
                            it.copy(
                                isConnected = true,
                                statusMessage = "Streaming to ${event.message.sourceLanguage ?: it.sourceLanguage.code}",
                            )
                        }
                    }

                    is SessionEvent.FrontUpdate -> {
                        val lines = event.payload.lines.map { line ->
                            TranscriptionLine(
                                speaker = line.speaker,
                                text = line.text,
                                translation = line.translation,
                                start = line.start,
                                end = line.end,
                                detectedLanguage = line.detectedLanguage,
                            )
                        }
                        val translationPreview = lines.mapNotNull { it.translation?.takeIf { text -> text.isNotBlank() } }
                            .joinToString(separator = "\n")
                        _uiState.update {
                            it.copy(
                                transcriptLines = lines,
                                bufferText = event.payload.bufferTranscription,
                                translationPreview = translationPreview,
                                statusMessage = when (event.payload.status) {
                                    "no_audio_detected" -> "Waiting for speech..."
                                    "active_transcription" -> "Listening"
                                    else -> event.payload.status
                                },
                                errorMessage = event.payload.error,
                            )
                        }
                    }

                    SessionEvent.ReadyToStop -> {
                        _uiState.update {
                            it.copy(
                                statusMessage = "Processed",
                                isConnected = false,
                            )
                        }
                    }

                    is SessionEvent.Closed -> {
                        _uiState.update {
                            it.copy(
                                statusMessage = event.reason ?: "Connection closed",
                                isConnected = false,
                            )
                        }
                        stopStreaming()
                    }

                    is SessionEvent.Failure -> {
                        _uiState.update {
                            it.copy(
                                errorMessage = event.throwable.message ?: "Unknown error",
                                statusMessage = "Error",
                            )
                        }
                        stopStreaming()
                    }
                }
                }
            }.apply {
                invokeOnCompletion {
                    stopStreamingInternal()
                }
            }
        } finally {
            toggleGuard.set(false)
        }
    }

    fun stopStreaming() {
        try {
            sessionJob?.cancel()
        } finally {
            toggleGuard.set(false)
        }
    }

    private fun stopStreamingInternal() {
        audioStreamer.stop()
        sessionJob = null
        _uiState.update {
            it.copy(
                isStreaming = false,
                isConnected = false,
            )
        }
        toggleGuard.set(false)
    }

    override fun onCleared() {
        super.onCleared()
        sessionJob?.cancel()
        audioStreamer.stop()
    }

    companion object {
        private const val TOGGLE_DEBOUNCE_MS = 750L
    }
}
